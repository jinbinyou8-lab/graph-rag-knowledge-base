import os.path

import uvicorn
from fastapi import FastAPI,File,UploadFile
from fastapi import Depends,BackgroundTasks
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from knowledge.processor.import_process.core.deps import get_task_service, get_import_file_service
from knowledge.processor.import_process.services.import_file_service import ImportFileService
from knowledge.processor.import_process.core.paths import get_front_page_dir, get_local_base_dir
from fastapi.responses import FileResponse

from knowledge.processor.import_process.services.task_service import TaskService
from knowledge.processor.import_process.shcema.upload_schema import UploadResponse
from knowledge.processor.import_process.shcema.task_schema import TaskStatusResponse
from knowledge.processor.import_process.services.import_file_service import ImportFileService
def create_app()->FastAPI:
    """
    创建FastAPI实例
    Returns:

    """
    #1. 创建FastAPI实例
    app = FastAPI(description="知识库的导入")

    #2. 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],    #允许任意的源
        allow_credentials=True, #允许cookie中携带任意的自定义参数
        allow_methods=["*"],    #允许任意的请求方式
        allow_headers=["*"],    #允许前端带上任意请求头（token、自定义 header）
    )

    # 3. 将静态资源的目录挂在到app实例上
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    # 4. 注册路由(接收前端发送的各种方式的请求)
    register_router(app)

    return app

def register_router(app:FastAPI):
    #1. 处理导入页面访问请求
    @app.get("/import")
    async def import_root():
        return FileResponse(path=os.path.join(get_front_page_dir(),'import.html'))

    #2. 上传请求
    @app.post("/upload", response_model=UploadResponse)
    async def upload_file_endpoint(backgrounf_tasks:BackgroundTasks,file: UploadFile=File(...),
                                   service:ImportFileService = Depends(get_import_file_service)):
        #1. 上传文件(本地/minio)
        task_id, file_dir, import_file_path = service.process_upload_file(file)

        #2. 运行后台任务(跑graph节点)
        backgrounf_tasks.add_task(service.run_import_graph,task_id, file_dir, import_file_path)

        #3. 返回
        return UploadResponse(message="文件上传成功",task_id=task_id)

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status_endpoint(
            task_id: str,
            task_service: TaskService = Depends(get_task_service)
    ) -> TaskStatusResponse:
        task_info = task_service.get_task_info(task_id)
        return TaskStatusResponse(**task_info)





if __name__ == "__main__":
    """
    启动web服务器(fastapi实例)
    """
    uvicorn.run(app=create_app(),port = 8000,host="0.0.0.0")