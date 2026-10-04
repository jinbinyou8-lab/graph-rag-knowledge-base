"""查询相关 Schema 定义。

来源：shopkeeper_brain 仓库 knowledge/schema/query_schema.py，
按本项目目录约定落到 processor/query_process/shcema/ 下，
并补充前端 chat.html 实际消费的字段（done_list / error / task_id）。
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")
    is_stream: bool = Field(False, description="是否流式返回")


class QueryResponse(BaseModel):
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    answer: str = Field("", description="生成的答案")
    # 以下两项为前端 chat.html 非流式分支所消费，缺失时前端会退化显示
    done_list: List[str] = Field(default_factory=list, description="已完成的节点（中文名）")
    error: str = Field("", description="错误信息，无错误时为空")


class StreamSubmitResponse(BaseModel):
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    task_id: str = Field(..., description="任务ID，前端用此 ID 建立 SSE 连接")


class TaskStatusResponse(BaseModel):
    """任务状态（前端轮询 /status/{task_id} 使用）"""

    task_id: str = Field("", description="任务ID")
    status: str = Field("", description="任务状态：processing/completed/failed")
    answer: str = Field("", description="已生成的答案")
    error: str = Field("", description="错误信息")
    done_list: List[str] = Field(default_factory=list, description="已完成节点")
    running_list: List[str] = Field(default_factory=list, description="进行中节点")


class HistoryItem(BaseModel):
    id: str = Field("", alias="_id")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    item_names: List[str] = Field(default_factory=list)
    ts: Optional[float] = None


class HistoryResponse(BaseModel):
    session_id: str
    items: List[HistoryItem]
