from dataclasses import dataclass
from pymilvus import DataType,MilvusClient
from knowledge.processor.import_process.base import  BaseNode,setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.config import get_config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Tuple, List, Dict, Any, Sequence, Optional
from knowledge.utils.markdown_utils import MarkdownTableLinearizer
from knowledge.utils.bge_m3_embedding import get_bge_m3_embedding
from knowledge.utils.milvus_util import get_milvus_client
"""
门面+建造者设计模式
门面角色：ImportMilvusNode节点的process（1.数据校验 2.insert() 3.更新sate(返回)）

设计3个类都是和Milvus操作相关

类1：MilvusSchemaBuilder:专门负责对Milvus的约束操作
类2：MilvusIndexBuilder:专门负责对Milvus的索引操作
类3：MilvusInserter:专门负责对Milvus做插入操作


Milvus的约束：
# 1.主键字段约束：唯一性最强
# 2.向量字段约束：唯一性还行
# 3.标量字段约束：灵活【title,file_tile,url,author,page_number】


类4：专门负责管理Milvus标量字段 （对大多数标里的共性字段做提取复用）
"""

@dataclass(frozen=True)     #frozen属性限制住实例元素没办法再进行修改
class ScalarFieldSpec:
    field_name: str
    datatype: DataType
    max_length: Optional[int]=None

# Sequence:表明修饰的对象是一个有序可读的序列
# enable_dynamic_field=True:表示的是schema中没有定义的约束，但是插入数据的时候，有不在schema中没有定义的字段。允许插入进去。切记：不是我定义的，你插入数据的时候可以不传。
_SCALAR_FIELDS:Sequence=(
    ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR,max_length=65535),
    ScalarFieldSpec(field_name="title",datatype=DataType.VARCHAR,max_length=65535),
    ScalarFieldSpec(field_name="parent_title",datatype=DataType.VARCHAR,max_length=65535),
    ScalarFieldSpec(field_name="file_title",datatype=DataType.VARCHAR,max_length=65535),
    ScalarFieldSpec(field_name="item_name",datatype=DataType.VARCHAR,max_length=65535)
)



class _MilvusSchemaBuilder:
    """
    职责: 专门负责构建约束
    """

    @staticmethod
    def bulid(client:MilvusClient, dim: int):
        #1. 构建约束对象(动态映射)
        schema = client.create_schema(enable_dynamic_field=True)

        #2. 构建主键字段约束
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary = True,
            auto_id = True
        )

        #3. 构建向量字段约束
        #3.1 稠密向量约束
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim = dim
        )

        #3.2 稀疏向量约束
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 4. 构建标量字段约束(TODO)
        for scalar_field in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {"field_name": scalar_field.field_name, "datatype": scalar_field.datatype}

            if scalar_field.max_length is not None:
                kwargs['max_length'] = scalar_field.max_length

            schema.add_field(**kwargs)
        return schema

class _MilvusIndexBuilder:
    """
    职责：负责处理Milvus的索引
    """

    @staticmethod
    def build(client: MilvusClient, collection_name: str):
        # 1. 创建索引对象
        index = client.prepare_index_params(collection_name=collection_name)

        # 2. 给向量字段添加索引
        # 2.1 稠密向量字段添加索引
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        # 2.2 稀疏向量字段添加索引
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 3. 返回index
        return index

class _MilvusInserter:
    """
    职责：将数据插入到Milvus 以及 回填chunk_id
    """

    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def insert(self, chunks: List[Dict[str, Any]]):
        # 1. 插入
        inserted_result = self._client.insert(collection_name=self._collection_name, data = chunks)
        inserted_count = inserted_result.get('insert_count')

        ids = inserted_result.get('ids')

        # 2. 回填id
        self._fill_chunk_ids(chunks, ids)

        return chunks

    def _fill_chunk_ids(self, chunks: List[Dict[str, Any]], ids: List[Any]):
        for chunk, id in zip(chunks, ids):
            chunk["chunk_id"] = id


class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 参数校验
        validated_chunks, dim, config = self._validate_get_inputs(state)

        # 2. 获取milvus客户端
        milvus_client = get_milvus_client()

        # 3. 判断milvus客户端
        if milvus_client is None:
            return state

        # 4. 获取集合名字
        collection = config.chunks_collection or "chunks_collection"

        # 5.确保集合存在（判断集合是否有、没有 创建新的【Schema index】）
        self._ensure_has_collection(milvus_client, collection, dim)

        #6.插入
        inserter = _MilvusInserter(client=milvus_client, collection_name=collection)

        final_chunks = inserter.insert(chunks=validated_chunks)

        #7. 更新state
        state["chunks"] = final_chunks

        return state
        

    def _validate_get_inputs(self, state: ImportGraphState):
        self.log_step(step_name="step1", message="参数校验")
        config = get_config()
        # 1. 获取chunks
        chunks = state.get('chunks')

        # 2. 校验是否为空
        if not chunks:
            raise ValidationError("待入库的切块chunk不存在", self.name)

        # 3. 校验是否有混合向量
        validated_chunks = []
        for chunk in chunks:

            if chunk.get('dense_vector') and chunk.get('sparse_vector'):
                validated_chunks.append(chunk)
            else:
                self.logger.error("待入库的切块chunk的混合向量不存在")

        # 4. 判断有效集合
        if not validated_chunks:
            raise ValidationError("入库的chunk都无效", self.name)

        # 5. 获取向量维度
        dim = len(validated_chunks[0].get('dense_vector'))
        self.logger.info(f"导入Milvus向量数据库的有效块: {len(validated_chunks)}, 且chunk的向量维度{dim}")

        return validated_chunks, dim, config

    def _ensure_has_collection(self, milvus_client:MilvusClient, collection_name:str, dim:int, delete_flag:bool=True):


        #1. 是否要删除集合
        if delete_flag == True and milvus_client.has_collection(collection_name=collection_name):
            milvus_client.drop_collection(collection_name=collection_name)

        #2. 判断集合是否存在
        if milvus_client.has_collection(collection_name=collection_name):
            self.logger.info(f"{collection_name}集合已经存在")
            return

        #3. 创建约束
        schema = _MilvusSchemaBuilder.bulid(milvus_client, dim)

        #4. 创建索引
        index = _MilvusIndexBuilder.build(milvus_client, collection_name)

        #5. 创建集合
        milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index)



