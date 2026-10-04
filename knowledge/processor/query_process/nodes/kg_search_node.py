"""
node_query_kg — 知识图谱查询节点。
类结构（与导入侧 kg_graph_node.py 的 Writer 模式对称）：
_EntityExtractor    LLM 实体抽取 + JSON 解析
_EntityAligner      Milvus ENTITY_NAME_COLLECTION 实体对齐
_Neo4jGraphReader   Neo4j 种子节点 / 一跳关系 / chunk 反查
_ChunkBackfiller    Milvus CHUNKS_COLLECTION chunk 回填
KGQueryNode         主编排器（组装上述四个组件，执行 pipeline）

node_query_kg()     LangGraph 节点入口函数（薄包装）
"""
import json
import logging
import re
import os
from dotenv import  load_dotenv
load_dotenv()
from typing import Dict, List, Any, Tuple, Union

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode, T
from  knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.utils.llm_client import get_llm_client
from knowledge.processor.query_process.prompts.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage,HumanMessage
from knowledge.utils.milvus_util import get_milvus_client,create_hybrid_search_requests,execute_hybrid_search_query,fetch_chunks_by_chunk_ids
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model,generate_hybrid_embeddings
from knowledge.utils.neo4j_util import get_neo4j_driver
# ALLOWED_ENTITY_LABELS_CN: str = (
#     "设备(Device)、部件(Part)、操作(Operation)、步骤(Step)、"
#     "警告(Warning)、条件(Condition)、工具(Tool)"
# )

_ENTITY_NAME_MAX_LENGTH = 15
_DEFAULT_ENTITY_NAME_ALIGN=0.5
ItemEntityPair = Dict[str, Any]
EntitySeedNode = Dict[str, Any]
OneHopRelation = Dict[str, Any]

SEED_NODE_WEIGHT=2.0
NER_NODE_WEIGHT=1.0



# Neo4j的Cypher语句
_CYPHER_EXACT_SEEDS="""
MATCH (n:Entity)
WHERE n.item_name=$item_name AND n.name=$name
RETURN n.item_name as item_name,n.name as name
LIMIT 1
"""

# toLower()小写
_CYPHER_FUZZY_SEEDS="""
MATCH (n:Entity)
WHERE toLower(n.name) CONTAINS toLower($entity_name)
    AND n.item_name = $item_name
RETURN n.name AS name, n.item_name AS item_name
LIMIT $limit
"""

# 查询种子节点的一跳关系
_CYPHER_ONE_HOP_RELATIONS="""

MATCH (seed:Entity {name:$name, item_name:$item_name})-[r]-(nbr:Entity)
WHERE type(r) <> 'MENTIONED_IN' AND nbr.item_name=$item_name

RETURN
  CASE WHEN startNode(r)=seed THEN seed.name ELSE nbr.name END AS head,
  type(r) as rel,
  CASE WHEN startNode(r)=seed THEN nbr.name ELSE seed.name END as tail

limit $limit
"""

# 根据带权重的节点查询chunk_id
_CYPHER_LOOKUP_CHUNK="""
UNWIND $weighted_nodes as n

MATCH (e:Entity{name : n.entity_name,item_name : n.item_name})-[r:MENTIONED_IN]->(c:Chunk{item_name : n.item_name})

WITH c,sum(n.weight) AS score, count(e) AS cnt

RETURN c.id AS chunk_id, c.item_name AS item_name, score, cnt

ORDER BY score DESC, cnt DESC,chunk_id DESC

LIMIT $limit
"""



# Neo4j的常量
_MAX_FUZZY_SEEDS: int = 3



def _clean_parse_llm_content(llm_response_content:str)->List[str]:
    """
    职责：清洗以及解析LLM输出
    Args:
        llm_response_content:

    Returns:
        List[str]:清洗后的实体名
    """
    # 1. 判断LLM输出内容是否为空
    if not llm_response_content:
        return []

    # 2. 清洗json代码围栏
    text = re.sub(r"^```(?:json)?\s*", "", llm_response_content)
    re_sub = re.sub(r"\s*```$", "", text)

    #3.  反序列化解析
    deserialized_result = json.loads(re_sub)

    # 4. 获取提取的实体名
    entities_name = deserialized_result.get('entities', [])

    # 4.1 提取实体名的校验（为空）
    if not entities_name:
        return []
    # 4.2 提取实体名的校验（有效类型）
    if not isinstance(entities_name, list):
        return []

    # 4.3 遍历所有实体名
    seen = set()
    entities_name_result = []
    for entity_name in entities_name:
        # 1. 判断是否为空
        if not entities_name:
            continue
        # 2. 判断是否有效类型
        if not isinstance(entity_name, str):
            continue

        # 3. 实体名是否过长
        truncated_entity_name = truncate_entity_name_length(entity_name)

        # 4. 去重保序【顺序：防御性】
        if truncated_entity_name not in seen:
            seen.add(truncated_entity_name)
            entities_name_result.append(truncated_entity_name)

    return entities_name_result

def _clean_seed_rows(rows: List[Dict[str, Any]]):
    """
    职责：清洗查询种子节点的数据
    Args:
        rows: 查询到的结果记录

    Returns:
        干净的结果记录
    """
    # 1. 判断查询记录是否存在
    if not rows:
        return []
    clean_seeds_result = []
    # 2. 遍历 rows:
    for row in rows:
        # 2.1 获取item_name
        item_name = row.get('item_name', '').strip()
        # 2.2 获取entity_name
        entity_name = row.get('name', '').strip()
        # 2.3 判断
        if not item_name or not entity_name:
            continue
        # 2.3 封装一下
        clean_seeds_result.append({
            "item_name": item_name,
            "entity_name": entity_name
        })

    return clean_seeds_result


def truncate_entity_name_length(entity_name: str) -> str:
    name = entity_name.strip()
    return name[:_ENTITY_NAME_MAX_LENGTH] if len(name) > _ENTITY_NAME_MAX_LENGTH else name


def _item_name_filter_expr(item_names: List[str]) -> str:
    quoted = ", ".join(f"'{item_name}'" for item_name in item_names)
    return f"item_name in [{quoted}]"

class _ChunkBackFiller:
    """
    职责：
    1、根据Neo4J返回的chunk信息（entity{"chunk_id"}） 获取到chunk_ids
    2、根据chunk_ids 查询milvus 获取到chunks(批量操作，返回的chunk没有所谓的顺序，只负责将chunk_id的对象给你)：导致根据分数降序的chunk_id就失去了作用
    3、构建一个chunk_id和chunk对象的字典映射表，将milvus返回的chunk_id和chunk对象存储进来
    4、遍历原有分数降序的chunk_id列表，然后在从该映射表中获取对应的chunk(保证原有根据分数降序的chunk_id顺序能利用起来)
    5. 更新到state当中
    """



    def __init__(self, collection_name:str):
        self._collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def back_fill(self, chunk_nodes_sorted: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        职责：
        1. 获取所有chunk_id
        2. 根据批量的chunk_id查询milvus
        3. 构建chunk_id和chunk对象的映射表（内存操作 快且没有oom风险（batch））
        4. 遍历原有顺序的chunk_id列表, 接着去到映射表中获取对应的chunk

        Args:
            chunk_nodes_sorted:

        Returns:

        """
        # 1. 判断chunk_nodes_sorted是否存在
        if not chunk_nodes_sorted:
            return []

        # 2. 获取chunk_ids
        chunk_ids: List[Union[str, int]] = self._collect_chunk_ids(chunk_nodes_sorted)

        # 3. 查询milvus 获取chunk对象
        try:
            chunks: List[Dict[str, Any]] = fetch_chunks_by_chunk_ids(
                collection_name=self._collection_name,
                chunk_ids=chunk_ids,
                output_fields=['chunk_id', 'content', 'title', 'item_name'],
                batch_size=30
            )
            if not chunks:
                return []
        except Exception as e:
            self.logger.error(f"根据chunk_id批量查询chunk对象失败：{str(e)}")
            return []

        # 4. 构建chunk_id和chunk对象的映射表
        chunk_id_map = {str(chunk.get('chunk_id')): chunk for chunk in chunks if chunk.get('chunk_id') is not None}

        # 5. 根据真实顺序的chunk_id从映射表查询真正的chunk对象
        return [{"entity":chunk_id_map.get(str(chunk_id))} for chunk_id in chunk_ids]

    def _collect_chunk_ids(self, chunk_nodes_sorted: List[Dict[str, Any]]) -> List[Union[str, int]]:
        """
        Args:
            chunk_nodes_sorted:

        Returns:

        """
        chunk_ids = []
        # 1. 遍历chunk_ids
        for chunk_node in chunk_nodes_sorted:

            # 2. 判断chunk_id
            if not chunk_node:
                continue

            # 3. 获取entity
            entity= chunk_node.get('entity', '')
            if not entity:
                continue
            # 4. 获取chunk_id
            chunk_id = entity.get('chunk_id')
            if not chunk_id:
                continue

            # 5. chunk_id转换
            chunk_id_str = str(chunk_id)
            try:
                chunk_id_int = int(chunk_id_str)
                chunk_ids.append(chunk_id_int)
            except (ValueError, TypeError):
                # 出现转换失败的异常很小
                chunk_ids.append(chunk_id_str)

        return chunk_ids


class _EntityExtractor:
    """
    实体提取器：
    责任： 利用LLM从查询问题中提取实体
    prompt: 设计
    """

    def extract(self, user_query: str) -> List[str]:
        """
        根据用户问题提取实体名
        Args:
            user_query: 用户问题

        Returns:
            List[str]: 提取后的实体名

        """
        #1. 获取llm的客户端
        llm_client = get_llm_client(json_mode=True)

        #2. 判断
        if llm_client is None:
            return []

        #3. 获取提示词
        #3.1 系统提示词
        entities_name_extract_system_prompt = ENTITY_EXTRACT_SYSTEM_PROMPT.format(MAX_ENTITY_NAME_LENGTH=_ENTITY_NAME_MAX_LENGTH)


        #4. 调用LLM
        #4.1 发送请求
        llm_response = llm_client.invoke(
            [
                SystemMessage(content=entities_name_extract_system_prompt),
                HumanMessage(content=f"用户问题:{user_query}")
            ]
        )

        #4.2. 获取模型的结果
        llm_response_content = llm_response.content.strip() or ""

        #4.3 清洗与解析
        entities_name = _clean_parse_llm_content(llm_response_content)

        return entities_name



class _EntityAligner:
    """
    实体对齐器：
    责任： 根据LLM提取到的实体名 查询Milvus，获取真正的实体名（对齐后的实体名、能够查询neo4j(查询节点使用)）
    """

    def __init__(self, collection_name: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._collection_name = collection_name

    def align(self, entity_names: List[str], item_names: List[str]) -> Dict[str, Any]:
        """

        Args:
            entity_names:
            item_names:

        Returns:
        Dict[str,Any]: 该字典准备封装两个key.

        第一个key:entities_aligned:[] 所有对齐后的实体名

        第二key:entityments[]: 所有对齐后的实体信息[source_id ,distance,origin,aligned,content]
        """

        fallback_result = {"entities_aligned_name": [], "entities_aligned_elements": []}
        # 1. 判断实体名是否有
        if not entity_names:
            return fallback_result

        # 2. 获取嵌入模型
        embedding_model = get_beg_m3_embedding_model()
        if embedding_model is None:
            self._logger.error("嵌入模型不存在")
            return fallback_result

        # 3. 获取milvus客户端
        milvus_client = get_milvus_client()
        if milvus_client is None:
            self._logger.error("Milvus客户端不存在")
            return fallback_result

        # 4. 向量化实体名
        embedding_result =generate_hybrid_embeddings(embedding_model=embedding_model, embedding_documents=entity_names)

        # 5.1 获取嵌入后的稠密向量对象（二维数组）
        embedding_result_dense = embedding_result['dense']
        # 5.2 获取嵌入后的稀疏向量对象（二维数组）
        embedding_result_sparse = embedding_result['sparse']

        # 6. 获取item_name的表达式
        item_name_filtered_expr = _item_name_filter_expr(item_names)

        # 7. 遍历所有的实体名字
        aligned_entities_name: List[str] = []  # 存放所有实体名字
        aligned_entity_elements: List[Dict[str, Any]] = []  # 存放所有实体的详细信息
        seen = set()
            # 7.1 对齐一个实体的名字
        for index, entity_name in enumerate(entity_names):

            align_one_result = self._align_one(milvus_client,
                            self._collection_name,
                            item_name_filtered_expr,
                            embedding_result_dense,
                            embedding_result_sparse,
                            index,entity_name)
            # 7.2 统一按 list 消费（一个实体可能命中多个商品）
            align_one_items = align_one_result if isinstance(align_one_result, list) else [align_one_result]
            for align_one_item in align_one_items:
                aligned_entity_name = align_one_item.get('aligned')
                # 防御：过滤空串，避免 "" 被塞进结果集
                if aligned_entity_name and aligned_entity_name not in seen:
                    seen.add(aligned_entity_name)
                    aligned_entities_name.append(aligned_entity_name)
                aligned_entity_elements.append(align_one_item)

        return {
            "entities_aligned_name": aligned_entities_name,
            "entities_aligned_elements": aligned_entity_elements
        }

    def _align_one(self, milvus_client, _collection_name, item_name_filtered_expr, embedding_result_dense,
                   embedding_result_sparse, index, entity_name):
        dense_vector = embedding_result_dense[index]
        sparse_vector = embedding_result_sparse[index]

        # 1. 判断实体名的稠密和稀释向量
        if not dense_vector or not sparse_vector:
            return [{"original": entity_name, "aligned": "", "context": "" , "reason": "vector values is not exist "}]
        # 2. 创建混合搜索请求
        hybrid_search_requests = create_hybrid_search_requests(dense_vector=dense_vector,
                                                               sparse_vector=sparse_vector,
                                                               expr=item_name_filtered_expr, limit=5)
        # 3. 执行混合搜索请求
        reps = execute_hybrid_search_query(milvus_client=milvus_client,
                                           collection_name=_collection_name,
                                           search_requests=hybrid_search_requests,
                                           ranker_weights=( 0.4,0.6),
                                           norm_score=True,
                                           limit=5,
                                           output_fields=["source_chunk_id", "item_name", "context", "entity_name"])

        # 4. 解析结果
        hits = reps[0] if reps else []
        if not hits:
            return [{"original": entity_name, "aligned": "", "score": "", "reason": "no_hit"}]

        # 4.1 按 item_name 分组，每组取最高分
        best_by_item: Dict[str, Dict] = {}
        for hit in hits:
            # a) 获取实体
            entity = hit.get("entity")
            # b) 从实体中获取
            item_name = entity.get("item_name").strip()
            # c) 只保留每个 item_name 下的第一个（即最高分）
            if item_name not in best_by_item:
                best_by_item[item_name] = hit

        # 4.2 是否有最好的item_name
        if not best_by_item:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "no_valid_item_name"}]

        # 4.3 item_name 分组输出结果，过滤低于阈值的
        results: List[Dict[str, Any]] = []
        for item, best in best_by_item.items():
            # a) 获取最好的那一个分数
            score = best.get("distance")
            # b) 判断分数值
            if float(score) < float(_DEFAULT_ENTITY_NAME_ALIGN):
                continue
            # c) 获取实体信息
            ent = best.get("entity")

            # d）将不同商品下最好的实体名添加到结果集中
            results.append({
                "original": entity_name,
                "aligned": ent.get("entity_name"),
                "score": score,
                "item_name": item,
                "source_chunk_id": ent.get("source_chunk_id"),
                "reason": "top1_per_item",
            })
        # 4.4 全部低于阈值时返回未命中
        if not results:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "all_below_threshold"}]

        return results


    def _pick_best_entity_name(self, search_entities_name_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        职责： 从返回的5个实体名字中留下一个实体名
        Args:
            param:

        Returns:

        """
        # 1. 判断是否检索到了
        if not search_entities_name_result:
            return None

        # 2. 获取第一个
        first_entity = search_entities_name_result[0]
        if not first_entity:
            return None

        # 3. 获取第一个实体名的分数值
        first_entity_name_score = first_entity.get('distance')

        # 4. 判断是否超过阈值
        if not first_entity_name_score:
            return None

        # 5. 返回的实体名是第一个且分数超过阈值的（对齐策略）
        return first_entity if first_entity_name_score > _DEFAULT_ENTITY_NAME_ALIGN else None

class _Neo4jGraphReader:
    """
    职责：所有对Neo4j的读操作
    1. 种子节点的查询（1.1 精确查询 1.2 降级走模糊查询兜底 ）
    2. 查询种子节点一跳关系（双向：种子节点指向另外的节点，以及另外的节点指向种子节点） 保留完整的关系
    3. 根据所有的节点（种子节点以及邻居节点）方向查询chunk(item_name, id)
    4. 根据所有的chunk_id 查询milvus得到所有的chunk
    """

    def __init__(self,  database,
                 KG_MAX_SEED_CANDIDATES:int,
                 KG_MAX_TOTAL_SEEDS:int,
                 KG_MAX_TRIPLES_PER_SEED: int,
                 KG_MAX_TOTAL_TRIPLES: int,
                 KG_MAX_TOTAL_CHUNKS:int
                 ):
        self._database = database
        self._KG_MAX_SEED_CANDIDATES = KG_MAX_SEED_CANDIDATES
        self._KG_MAX_TOTAL_SEEDS = KG_MAX_TOTAL_SEEDS
        self._KG_MAX_TRIPLES_PER_SEED = KG_MAX_TRIPLES_PER_SEED
        self._KG_MAX_TOTAL_TRIPLES = KG_MAX_TOTAL_TRIPLES
        self._KG_MAX_TOTAL_CHUNKS = KG_MAX_TOTAL_CHUNKS
        self._logger = logging.getLogger(self.__class__.__name__)

    def _session(self):
        # 1.获取驱动
        neo4j_driver = get_neo4j_driver()

        # 2. 判断驱动是否存在
        if neo4j_driver is None:
            raise RuntimeError("Neo4J驱动获取失败")

        # 3. 返回session对象
        return neo4j_driver.session(database=self._database)

    def find_seed_nodes(self, pairs: List[ItemEntityPair]) -> List[EntitySeedNode]:
        """
        职责：根据item_name 以及entity_name 查询种子节点
        策略：精确查询，只返回一条 模糊查询，返回三条
        Args:
            pairs:  _build_item_entity_pairs方法返回的商品名和实体名的pair对

        Returns:
            所有商品名下所有实体名对应的种子节点
        """
        # 1. pair对是否存在
        if not pairs:
            return []
        final_seeds_result: List[EntitySeedNode] = []
        # 2. 遍历所有pair对
        for pair in pairs:
            # 2.1 获取item_name
            item_name = pair.get('item_name', '').strip()
            # 2.2 获取entity_name
            entity_name = pair.get('entity_name', '').strip()
            # 2.3 过滤掉无效
            if not item_name or not entity_name:
                continue
            # 2.4 执行cypher语句（1）精确查询 2）可能要模糊查询）
            try:
                with self._session() as session:

                    candidates_seed_nodes = self._execute_seed_nodes(session, item_name, entity_name, self._KG_MAX_SEED_CANDIDATES)

                    # # 1.精确查询
                    # exact_rows = session.execute_read(
                    #     lambda tx: tx.run(
                    #         _CYPHER_EXACT_SEEDS, item_name=item_name, name=entity_name
                    #     ).data()
                    # )
                    # if exact_rows:
                    #     final_seeds_result.extend(_clean_seed_rows(exact_rows))
                    #     continue
                    # # 2. 模糊查询
                    # fuzzy_rows = session.execute_read(
                    #     lambda tx: tx.run(
                    #         _CYPHER_FUZZY_SEEDS, item_name=item_name, entity_name=entity_name, limit=_MAX_FUZZY_SEEDS
                    #     ).data()
                    # )
                    final_seeds_result.extend(candidates_seed_nodes)
                    if len(final_seeds_result) > self._KG_MAX_TOTAL_SEEDS:
                        final_seeds_result = final_seeds_result[:self._KG_MAX_TOTAL_SEEDS]
                        break

            except Exception as e:
                self._logger.error(f"获取种子节点失败,原因 :{str(e)}")
                return []

        return final_seeds_result

    def _execute_seed_nodes(self, session, item_name, entity_name, _KG_MAX_SEED_CANDIDATES):


        # 1.精确查询
        exact_rows = session.execute_read(
            lambda tx: tx.run(
                _CYPHER_EXACT_SEEDS, item_name=item_name, name=entity_name
            ).data()
        )
        if exact_rows:
            return _clean_seed_rows(exact_rows)
        # 2. 模糊查询
        fuzzy_rows = session.execute_read(
            lambda tx: tx.run(
                _CYPHER_FUZZY_SEEDS, item_name=item_name, entity_name=entity_name, limit=_KG_MAX_SEED_CANDIDATES
            ).data()
        )
        return _clean_seed_rows(fuzzy_rows)

    def find_one_hop_relations(self, seed_nodes: List[EntitySeedNode]):
        """
        职责：💡根据种子节点查询一跳的关系（双向），并且过滤掉MENTIONED_IN 关系的节点
        注意: 1. 去重（不允许同一条边出现多次）只能出现一次。 2. 图谱中存储的节点和关系结构是什么 查询的时候一定要和存储的我结构保证一致 3. 邻居节点可以是你在一跳范围内指向的节点也可以别人在一跳范围内找到的节点
        比如: A->B(类型: 认识) A->B(类型: 认识) B->A(类型: 认识)

        Args:
            seed_nodes: find_seed_nodes: 所有种子节点(所有商品的种子节点)

        Returns:List[OneHopNode]

        """
        # 1. 判断种子节点
        if not seed_nodes:
            return []
        seen = set()
        one_hop_relations_final_result = []
        # 2. 遍历所有的种子节点
        for seed_node in seed_nodes:
            # 2.1 提取item_name
            item_name = seed_node.get('item_name', "")
            # 2.2 提取entity_name
            seed_name = seed_node.get('entity_name', "")
            # 2.3 判断是否都存在
            if not item_name or not seed_name:
                continue

            # 2.4 执行Cypher语句
            try:
                with self._session() as session:
                    # a) 查询种子节点的一跳关系
                    seed_one_hop_relations: List[OneHopRelation] = self._execute_one_hop_relations(session, item_name,
                                                                                                   seed_name,
                                                                                                   self._KG_MAX_TRIPLES_PER_SEED)
                    if not seed_one_hop_relations:
                        continue

                    # b）遍历种子节点所有的关系
                    for seed_one_hop_relation in seed_one_hop_relations:
                        # b.1 获取头
                        head = seed_one_hop_relation.get('head')
                        # b.2 获取rel
                        rel = seed_one_hop_relation.get('rel')
                        # b.3 获取tail
                        tail = seed_one_hop_relation.get('tail')
                        # b.4 获取item_name
                        item_name = seed_one_hop_relation.get('item_name')

                        # b.4 去重（同一条边不能重复出现）同一个商品下，不运行有重复的 不同商品下不能叫重复的边
                        # 场景：A节点是种子节点 邻居也是种子节点（A节点作为种子查询邻居节点的时候已经把他们的关系查找到了）所以当在以邻居节点为种子查询的时候，就会出现重复的边。因此要过滤掉
                        # 去重key
                        key = (item_name, head, rel, tail)

                        if key not in seen:
                            seen.add(key)
                            one_hop_relations_final_result.append(seed_one_hop_relation)

                    # c）截取 种子节点的关系，防止超过LLM窗口阈值
                    if len(one_hop_relations_final_result) > self._KG_MAX_TOTAL_TRIPLES:
                        one_hop_relations_final_result = one_hop_relations_final_result[:self._KG_MAX_TOTAL_TRIPLES]
                        break
                    # d) 返回
            except Exception as e:
                self._logger.error(f"查询 {seed_name} 种子节点的一跳关系失败: {str(e)}")
                return []
        return one_hop_relations_final_result

    def _execute_one_hop_relations(self, session, item_name, seed_name, _KG_MAX_TRIPLES_PER_SEED):
        """

        Args:
        session: neo4j驱动
        item_name: 商品名
        seed_name: 种子节点名字
        kg_max_triples_per_seed:种子节点最大的关系数

    Returns:
        List[OneHopRelation]:种子节点的关系

        """

        # 1. 根据session执行查询方法
        one_hop_relations = session.execute_read(
            lambda tx: tx.run(
                _CYPHER_ONE_HOP_RELATIONS, item_name=item_name, name=seed_name, limit=_KG_MAX_TRIPLES_PER_SEED
            ).data()
        )

        # 2. 解析结构
        if not one_hop_relations:
            return []

        # 3. 遍历所有的一条关系
        one_hop_relations_result = []
        for one_hop_relation in one_hop_relations:
            # 3.1 提取head
            head = one_hop_relation.get('head', '').strip()
            # 3.2 提取rel
            rel = one_hop_relation.get('rel', '').strip()
            # 3.3 提取tail
            tail = one_hop_relation.get('tail', '').strip()

            # 3.4 判断是否存在关系链
            if not (head and rel and tail):
                continue

            # 3.5 将关系链添加到最终结果中
            one_hop_relations_result.append({
                "head": head,
                "rel": rel,
                "tail": tail,
                "item_name": item_name
            })

        return one_hop_relations_result

    def deal_node_weight(self, seed_nodes: List[EntitySeedNode], one_hop_relations: List[OneHopRelation]) -> \
    List[Dict[str, Any]]:
        """
        职责：
        为种子节点设置权重weight:高=2.0
        为邻居节点设置权重weigh:低=1.0
        自定义的分数（权重求和）值越高、出现的次数越多，该chunk越重要
        排序：根据分数排、次数排 、 chunk_id顺序排
        Args:
            seed_nodes: 种子节点
            one_hop_relations:一跳关系

        Returns:
            带分数的chunk_id
        """
        # 1. 判断seed_nodes 是否存在
        if not seed_nodes:
            return []

        # 2. 判断 one_hop_relations 是否存在
        if not one_hop_relations:
            return []

        # 存放所有节点（种子节点和邻居节点的权重）
        weight_map: [Tuple[str, str], float] = {}
        seen = set()

        # 3. 遍历所有的种子节点
        for seed_node in seed_nodes:
            # 3.1 获取item_name
            item_name = seed_node.get('item_name')
            # 3.2 获取节点名
            seed_name = seed_node.get('entity_name')

            key = (item_name, seed_name)
            if key not in seen:
                seen.add(key)
                weight_map[key] = SEED_NODE_WEIGHT

        # 4. 遍历一跳三元组
        for one_hop_relation in one_hop_relations:

            # 4.1 获取head
            head = one_hop_relation.get('head')
            # 4.2 获取tail
            tail = one_hop_relation.get('tail')

            # 4.3 获取商品的名字
            item_name = one_hop_relation.get('item_name')

            # 4.4 为邻居节点赋值权重
            if head and (item_name, head) not in weight_map:
                weight_map[(item_name, head)] = NER_NODE_WEIGHT

            if tail and (item_name, tail) not in weight_map:
                weight_map[(item_name, tail)] = NER_NODE_WEIGHT
        return [{"item_name": it, "entity_name": en, "weight": w}
                for (it, en), w in weight_map.items()]

    def find_nodes_chunk_id(self, weighted_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据带权重的节点查询chunk_id
        并且基于权重和、次数、进行排序

        Args:
            weighted_nodes:带权重的节点

        Returns:
            查询到的chunk_id
        """
        # 1. 执行Cypher语句
        try:
            with self._session() as session:
                sorted_node_chunk_id = session.execute_read(lambda tx: tx.run(
                    _CYPHER_LOOKUP_CHUNK, weighted_nodes=weighted_nodes, limit=self._KG_MAX_TOTAL_CHUNKS
                ).data())

        except Exception as e:
            self._logger.error(f"反查chunk_id失败,原因:{str(e)}")
            return []

        hits = []

        # 2. 处理结果
        for chunk_row in sorted_node_chunk_id:
            chunk_id = chunk_row.get('chunk_id', "").strip()
            item_name = chunk_row.get('item_name', "").strip()
            score = chunk_row.get('score')

            if chunk_id and item_name:
                hits.append({
                    "id": None,
                    "distance": float(score or 0.0),
                    "entity": {"chunk_id": str(chunk_id), "item_name": str(item_name)}
                })

        return hits


def _build_item_entity_pairs(aligned_entities_info: List[Dict[str, Any]]) -> List[ItemEntityPair]:
    """
    职责：从对齐后的实体详情中获取商品名+实体名的pair对
    去重：本质同一个商品名下的实体名只留一个，不同商品名下的实体名都留
    Args:
        aligned_entities_info: 对齐后的实体详情列表

    Returns:
        商品+实体名的pairs
    """
    # 1. 判断对齐后的实体详情是否存在
    if not aligned_entities_info:
        return []

    seen = set()
    item_entity_pairs = []
    # 2. 遍历对齐后的实体详情
    for aligned_entity_info in aligned_entities_info:
        # 2.1 获取商品名item_name
        item_name = aligned_entity_info.get('item_name', "").strip()
        # 2.2 获取对齐后的实体名
        aligned_entity_name = aligned_entity_info.get('aligned', "").strip()
        # 2.3 商品名&实体名都存在
        if not (item_name and aligned_entity_name):
            continue
        # 2.4 去重
        key = (item_name, aligned_entity_name)
        if key not in seen:
            seen.add(key)
            item_entity_pairs.append({
                "item_name": item_name,
                "entity_name": aligned_entity_name
            })

    return item_entity_pairs


def _one_hop_triples_to_texts(triples: List[OneHopRelation]) -> List[str]:
    if not triples:
        return []
    docs: List[str] = []
    for tr in triples:
        it = (tr.get("item_name") or "").strip()
        h = (tr.get("head") or "").strip()
        r = (tr.get("rel") or "").strip()
        t = (tr.get("tail") or "").strip()
        if not (h and r and t):
            continue
        docs.append(f"[{it}] {h} -({r})-> {t}" if it else f"{h} -({r})-> {t}")
    return docs


def _strip_item_names(query: str, item_names: List[str]) -> str:
    toks = set()
    for name in item_names or []:
        name = (name or "").strip()
        if not name:
            continue
        toks.add(name)
        toks.update(t for t in name.split() if len(t) >= 2)                      # RS / PRO / RS-12 / 数字万用表
        toks.update(t for t in re.split(r"[-_/·、,，。()（）\s]+", name) if len(t) >= 2)  # 再切一次连字符
    cleaned = query
    for t in sorted(toks, key=len, reverse=True):        # 长 token 优先，避免 RS 先吃掉 RS-12
        cleaned = cleaned.replace(t, "")
        pat = r"[\s\-_/·、,，。()（）]*".join(re.escape(c) for c in t if not c.isspace())
        if len(pat) > 1:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\s\-]+", " ", cleaned).strip(" ,，。?？!！")
    return cleaned or query



class KnowledgeGraphSearchNode(BaseNode):
    """
    知识图谱查询主编排器。

    职责：
    - 组装四个服务组件（Extractor / Aligner / GraphReader / Backfiller）
    - 按 pipeline 顺序编排调用

    Pipeline:
    抽取实体 → 对齐实体 → Neo4j查询 → 回填chunk
    """

    name = "kg_search_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 参数校验
        validated_query, validated_item_names = self._validate_inputs(state)

        # 2. 执行流水线
        kg_result = self._run_pipeline(validated_query, validated_item_names)

        # 3. 更新state
        #    只返回本节点修改的字段：本节点与 vector/hyde/mcp 三路并行，
        #    返回整个 state 会触发 InvalidUpdateError("At key 'session_id': ...")
        return {
            "kg_chunks": kg_result.get("kg_chunks"),
            "kg_triples": kg_result.get("kg_triples"),
        }

    def _validate_inputs(self, state: QueryGraphState):
        # 1. 获取参数
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")

        # 2. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 3. 从重写的问题中踢掉商品名
        user_query = _strip_item_names(rewritten_query, item_names)

        # 4. 返回
        return user_query, item_names

    def _run_pipeline(self, validated_query: str, validated_item_names: List[str])->Dict[str, Any]:
        # 1. 初始化组件
        entity_extractor = _EntityExtractor()
        entity_aligner = _EntityAligner(collection_name=self.config.entity_name_collection)
        neo4j_graph_reader = _Neo4jGraphReader(database="neo4j",KG_MAX_SEED_CANDIDATES=int(os.getenv("KG_MAX_SEED_CANDIDATES","3")),
                                               KG_MAX_TOTAL_SEEDS=int(os.getenv("KG_MAX_TOTAL_SEEDS","30")),
                                               KG_MAX_TRIPLES_PER_SEED = int(os.getenv("KG_MAX_TRIPLES_PER_SEED","100")),
                                               KG_MAX_TOTAL_TRIPLES = int(os.getenv("KG_MAX_TOTAL_TRIPLES","100")),
                                               KG_MAX_TOTAL_CHUNKS = int(os.getenv("KG_MAX_TOTAL_CHUNKS","100"))
                                               )
        chunk_back_filler = _ChunkBackFiller(collection_name="chunks_collection")

        # 2. 利用提取器提取实体(核心的实体名字留下，就可以通过该实体节点找和该节点有关系的节点)
        entities_name = entity_extractor.extract(user_query=validated_query)
        entities_name_aligned: Dict[str, Any] = entity_aligner.align(entities_name, item_names=validated_item_names)

        # 2.1 获取所有对齐后的实体名(业务逻辑不使用)
        aligned_entities_name = entities_name_aligned.get('entities_aligned_name')

        # 2.2 获取所有对齐后的实体详情（结构信息细粒）
        aligned_entities_info = entities_name_aligned.get('entities_aligned_elements')

        # 3. 构建商品名+实体名的pair对
        item_entity_pairs: List[ItemEntityPair] = _build_item_entity_pairs(aligned_entities_info)

        # 4. Neo4j操作
        # 4.1 根商品名和实体名的pairs 查询种子节点
        seed_nodes: List[EntitySeedNode] = neo4j_graph_reader.find_seed_nodes(item_entity_pairs)
        # 4.2 根据种子节点查询一跳关系
        one_hop_relations: List[OneHopRelation] = neo4j_graph_reader.find_one_hop_relations(seed_nodes)
        # 4.3 根据种子节点(查询到的)以及一跳关系【种子节点/邻居节点】
        weighted_nodes = neo4j_graph_reader.deal_node_weight(seed_nodes, one_hop_relations)
        # 4.4 根据带权重的节点反查chunk,并且基于权重给chunk排序（权重排【sum】降序/次数排降序/chunk_id升序）
        chunk_nodes_sorted = neo4j_graph_reader.find_nodes_chunk_id(weighted_nodes)
        # 5.Milvus的操作(回填)
        kg_chunks = chunk_back_filler.back_fill(chunk_nodes_sorted)
        # 6.将一跳关系转换成能够理解真实图谱结构
        triples_docs = _one_hop_triples_to_texts(one_hop_relations)
        # 7. 汇总知识图谱节点的所有信息
        return {
            "kg_chunks": kg_chunks,  # 回填后的切片文本 → 送入 RRF
            "kg_triples": triples_docs,  # 关系文本描述 → 送入答案生成 prompt
            "kg_seed_nodes": seed_nodes,
            "kg_triples_raw": one_hop_relations,
            "kg_entities": entities_name,
            "kg_aligned_entities": aligned_entities_name,
            "kg_alignments": aligned_entities_info,
        }


if __name__ == '__main__':
    kg_search_node = KnowledgeGraphSearchNode()

    state = {
        "rewritten_query": "RS-12数字万用表如何测量直流电压",
        "item_names": ["RS PRO RS-12 数字万用表"]
    }
    result = kg_search_node.process(state)


    for i in state["kg_triples"]:
        print(i,end="\n")



