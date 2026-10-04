"""查询流程依赖注入。"""

from functools import lru_cache

from knowledge.processor.query_process.services.query_service import QueryService


@lru_cache
def get_query_service() -> QueryService:
    """获取查询业务服务单例。"""
    return QueryService()
