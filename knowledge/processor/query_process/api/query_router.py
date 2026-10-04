"""查询路由（FastAPI 应用入口，默认端口 8001）。

来源：shopkeeper_brain 仓库 knowledge/api/query_router.py，
按本项目目录约定落到 processor/query_process/api/ 下，并做三处适配：
1. 导入路径改为 knowledge.processor.query_process.*；
2. 非流式查询放到线程池执行（run_in_threadpool），避免阻塞事件循环
   —— 否则 mcp_search_node 内的 asyncio.run 会直接抛
   "asyncio.run() cannot be called from a running event loop"；
3. 补上 "GET /" 与 "GET /status/{task_id}"，前者方便直接打开聊天页，
   后者是 chat.html 轮询函数 poll() 所调用的接口。
"""

import os
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from knowledge.processor.query_process.core.paths import get_front_page_dir
from knowledge.processor.query_process.core.deps import get_query_service
from knowledge.processor.query_process.shcema.query_schema import (
    QueryRequest, QueryResponse, StreamSubmitResponse, TaskStatusResponse,
)
from knowledge.processor.query_process.services.query_service import QueryService
from knowledge.utils.sse_util import sse_generator
from knowledge.processor.query_process.base import setup_logging


def create_app() -> FastAPI:
    app = FastAPI(title="Query Service", description="知识库查询服务")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))
    register_routes(app)
    return app


def register_routes(app: FastAPI):

    @app.get("/")
    async def index():
        """根路径直接打开聊天页，省去手输 /chat.html。"""
        return FileResponse(os.path.join(get_front_page_dir(), "chat.html"))

    @app.get("/chat.html")
    async def chat_page():
        return FileResponse(os.path.join(get_front_page_dir(), "chat.html"))

    @app.post("/query")
    async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        service: QueryService = Depends(get_query_service),
    ):
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()
        service.submit_query(task_id, request.is_stream)

        if request.is_stream:
            background_tasks.add_task(
                service.run_query_graph, task_id, session_id, request.query, True
            )
            return StreamSubmitResponse(
                message="Query submitted", session_id=session_id, task_id=task_id
            )

        # 同步阻塞的图执行放到线程池，别卡住事件循环
        await run_in_threadpool(
            service.run_query_graph, task_id, session_id, request.query, False
        )
        return QueryResponse(
            message="处理完成",
            session_id=session_id,
            answer=service.get_answer(task_id),
            done_list=service.get_status(task_id)["done_list"],
            error=service.get_error(task_id),
        )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        return StreamingResponse(
            sse_generator(task_id, request), media_type="text/event-stream",
        )

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status(
        task_id: str,
        service: QueryService = Depends(get_query_service),
    ) -> TaskStatusResponse:
        return TaskStatusResponse(**service.get_status(task_id))

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/history/{session_id}")
    async def get_history(
        session_id: str, limit: int = 50,
        service: QueryService = Depends(get_query_service),
    ):
        try:
            items = await run_in_threadpool(service.get_history, session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
        session_id: str,
        service: QueryService = Depends(get_query_service),
    ):
        count = await run_in_threadpool(service.clear_history, session_id)
        return {"message": "History cleared", "deleted_count": count}


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8001)
