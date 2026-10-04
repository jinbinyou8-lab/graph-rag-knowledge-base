from pymilvus import DataType

from knowledge.processor.import_process.base import  BaseNode,setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.config import get_config
from typing import List,Dict,Optional,Any
from langchain_core.messages import SystemMessage,HumanMessage
from knowledge.utils.llm_client import get_llm_client
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from knowledge.utils.bge_m3_embedding import get_bge_m3_embedding
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.processor.import_process.prompt.item_name_prompt import ITEM_NAME_SYSTEM_PROMPT,ITEM_NAME_USER_PROMPT_TEMPLATE
class ItemNameRecognitionNode(BaseNode):

    name = "item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1. 参数校验
        file_title,  chunks, config = self._validate_inputs(state)

        #2. 构建LLLM的上下文(提取出商品名)
        item_name_context = self._prepare_item_name_context(chunks, config)

        #3. 调用LLM
        item_name = self._recognition_item_name_by_llm(file_title, item_name_context)

        #4. 嵌入商品名(使用本地bge-m3)
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        #5. 存储到milvus向量数据库中
        self._save_to_milvus(file_title, item_name, dense_vector, sparse_vector, config)

        #6. 回填item_name的信息[state/chunk对象]
        self._fill_item_name(item_name, state,chunks)

        return state

    def _validate_inputs(self, state:ImportGraphState):
        self.log_step("step1", "校验输入参数")
        config = get_config()

        #1. 获取state的file_title以及chunks
        file_title = state.get("file_title")
        chunks = state.get("chunks")

        #2. 判断提取到的参数
        if not file_title:
            raise ValidationError("文件标题为空", self.name)

        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunk为空或者无效", self.name)

        item_name_chunk_k = config.item_name_chunk_k

        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError("chunk为空或者无效", self.name)

        #3. 返回
        return file_title, chunks, config

    def _iter_flat_chunks(self, chunks):
        """把 chunks 摊平成 dict 序列。

        兼容两种来源：
        1. 流水线里 DocumentSpliterNode 产出的 chunk —— 每个元素本身就是 dict
        2. MinerU 原始的 *_content_list_v2.json —— 每个元素是"一页"，页内部才是 dict
        """
        for item in chunks or []:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        yield sub

    def _extract_content(self, chunk: Dict[str, Any]) -> str:
        """从一个 chunk 里取出正文文本"""
        content = chunk.get("content")

        #1. 流水线产出的 chunk：content 直接就是字符串
        if isinstance(content, str):
            return content.strip()

        #2. MinerU v2：content 是字典，正文藏在 paragraph_content 里
        if isinstance(content, dict):
            parts = []
            for item in content.get("paragraph_content") or []:
                if isinstance(item, dict) and item.get("content"):
                    parts.append(str(item["content"]))
            if parts:
                return " ".join(parts).strip()

            # 表格 / 图片等其他类型：退回字典内的字符串字段
            for key in ("content", "table_body"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        #3. MinerU v1：正文在 text 字段
        text = chunk.get("text")
        return text.strip() if isinstance(text, str) else ""

    def _prepare_item_name_context(self, chunks:Optional[List[Dict[str,Any]]], config):
        self.log_step("step2", "构建商品名提取的上下文")

        result = []
        #整段上下文的字符数不能超过 config.item_name_chunk_size
        total = 0

        for chunk in self._iter_flat_chunks(chunks):
            #1. 已经取够 item_name_chunk_k 块有效内容就停
            if len(result) >= config.item_name_chunk_k:
                break

            #2. 提取正文(空内容直接跳过, 不占切片名额)
            content = self._extract_content(chunk)
            if not content:
                continue

            spices = f"[切片]- {len(result)+1} - {content}"

            #3. 计算长度
            total += len(spices)
            result.append(spices)

            #4. 是否超过最大长度限制
            if total > config.item_name_chunk_size:
                break

        return  "\n\n".join(result)[:config.item_name_chunk_size]

    def _recognition_item_name_by_llm(self, file_title:str, item_name_context:str):
        #1. 实例化LLM客户端
        llm_client = get_llm_client()

        #2. 构建LLM的提示词(格式化用户提示词模板)
        prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_context)

        #3. 调用模型
        try:
            llm_response = llm_client.invoke(
                [
                    SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                    HumanMessage(content=prompt)
                ]
            )
            #4. 获取模型输出的内容
            content = getattr(llm_response, "content", "").strip()

            #5. 判断
            if not content or content.upper() == "UNKNOWN":
                self.logger.warning(f"LLM无法提取有效的商品名, 安全回退到标题名: {file_title}")
                item_name = file_title
                return item_name
            return content
        except Exception:
            #这里必须打完整堆栈, 否则 400/鉴权 之类的错误会被静默吞掉, 排查时看不到原因
            self.logger.exception(f"LLM调用失败, 安全回退到标题名: {file_title}")
            return file_title

    def _embedding_item_name(self, item_name):
        #1. 获取嵌入模型对象
        embedding_model = get_bge_m3_embedding()

        #2. 嵌入item_name
        embedding_result = embedding_model.encode_documents([item_name])

        #3. 获取稠密和稀疏向量
        dense = embedding_result["dense"][0].tolist()
        start_index = embedding_result['sparse'].indptr[0]
        end_index = embedding_result['sparse'].indptr[1]
        weights = embedding_result['sparse'].data[start_index:end_index].tolist()
        tokenIds = embedding_result['sparse'].indices[start_index:end_index].tolist()
        sparse = dict(zip(tokenIds, weights))
        return dense,sparse

    def _save_to_milvus(self, file_title, item_name, dense_vector, sparse_vector, config):
        self.log_step("step6", "保存到向量数据库中")

        if not dense_vector or not sparse_vector:
            self.logger.warning(f"[{item_name}] 向量生成不完整，跳过入库！")
            return

        try:
            milvus_client = get_milvus_client()
            if milvus_client is None:
                self.logger.error("Milvus 客户端不可用(连接失败), 本次跳过入库！")
                return

            collection_name = config.item_name_collection
            if not collection_name:
                self.logger.error("未配置 ITEM_NAME_COLLECTION, 本次跳过入库！")
                return

            if not milvus_client.has_collection(collection_name=collection_name):
                self._create_item_name_collection(milvus_client, collection_name)

            data = {
                "file_title": file_title,   #文件名字
                "item_name": item_name,     #商品名字
                "dense_vector": dense_vector,   #稠密向量   (list)
                "sparse_vector": sparse_vector  #稀疏向量   (dict:{tokenId:weight})
            }

            result = milvus_client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已成功保存到 Milvus, ID: {result['ids'][0]}")

        except Exception:
            self.logger.exception("Milvus 数据库保存操作彻底失败")

    def _create_item_name_collection(self, client, collection_name):
        self.logger.info(f"正在创建集合: {collection_name}")

        #创建约束
        schema = client.create_schema()

        #1.1 主键字段
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)

        #1.2 标量字段
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)

        #1.3 向量字段
        #1.3.1 稠密向量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        #1.3.2 稀疏向量字段
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        #2. 创建索引
        index_params = client.prepare_index_params()
        #2.1 为稠密向量建立索引
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        #2.1 为稀疏向量建立索引
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        #3. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name(self, item_name:str, state:ImportGraphState, chunks:List[dict[str,Any]]):

        #chunks 可能是"按页分组"的嵌套结构, 统一摊平后再回填
        for chunk in self._iter_flat_chunks(chunks):
            chunk["item_name"] = item_name      #方便下游模型使用参考

        state["item_name"] = item_name           #程序员使用的时候方便



