from knowledge.processor.import_process.base import  BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.config import get_config
from typing import Tuple,List,Dict,Any
from knowledge.utils.bge_m3_embedding import get_bge_m3_embedding



class BgeEmbeddingChunksNode(BaseNode):
    """
    1. 获取所有的chunks拼接想要的向量内容
    2. 批量嵌入 chunk的 (embdedding.content:item_name + chunk.get('content'))
    3. 将所有chunk嵌入后的向量值, 存储到列表中, 返回给下一个节点使用
    """
    name = 'bge_embedding_chunks_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1. 数据校验
        validated_chunks,config = self._validte_get_inputs(state)

        #2. 获取批量嵌入的阈值
        embedding_batch_chunk_size = getattr(config, "embedding_batch_size", 16)

        #3. 准备批量嵌入(pipline)
        total_length = len(validated_chunks)
        final_chunks = []

        for i in range(0,total_length, embedding_batch_chunk_size):
            batch = validated_chunks[i:i+embedding_batch_chunk_size]
            #拼接要嵌入的内容, 向量嵌入的内容, 把嵌入的向量注入到chunk当中
            batch_chunks = self._process_batch_chunks(batch, i, total_length)
            final_chunks.extend(batch_chunks)
        
        #4. 返回状态
        state["chunks"] = final_chunks
        return state

    def _validte_get_inputs(self, state:ImportGraphState):
        self.log_step("step1", "参数校验")
        config = get_config()

        #1. 获取chunks
        chunks = state.get("chunks")

        #2. 校验chunks
        if not chunks or not isinstance(chunks, list):
            raise ValidationError(f"chunks为空或者无效",self.name)

        #3. 返回chunks
        return chunks,config

    def _process_batch_chunks(self, batch:List[Dict[str, Any]], star_index, total_length:int):
        self.log_step("step2",f"分批进行嵌入,目前的批次是{star_index+1}")
        embedding_contents = []

        for _, chunk in enumerate(batch):
            #1.1 提取content
            content = chunk.get("content")

            #1.2 提取item_name
            item_name = chunk.get("item_name")

            #1.3 拼接要嵌入的最终内容
            embedding_content = f"{item_name}\n{content}"

            embedding_contents.append(embedding_content)


        #2. 批量嵌入
        bge_3_model = get_bge_m3_embedding()
        embedding_result = bge_3_model.encode_documents(documents=embedding_contents)

        # 3. 循环处理所有chunk的向量以及注入到每一个chunk中########重点处理过程
        for index, chunk in enumerate(batch):
            # 3.1 获取稠密向量
            dense_vector = embedding_result['dense'][index].tolist()

            # 3.2 解构csr矩阵&获取稀疏向量
            csr_array = embedding_result['sparse']
            # a) 行索引
            ind_ptr = csr_array.indptr

            # b) 获取行索引的起始值
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index + 1]

            # c) 获取token_id
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()

            # d) 获取权重
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()

            sparse_vector = dict(zip(token_id, weight))

            #3.3 注入
            chunk["dense_vector"] = dense_vector
            chunk["sparse_vector"] = sparse_vector

        return batch