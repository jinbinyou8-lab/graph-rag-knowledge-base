from functools import lru_cache

from knowledge.processor.import_process.services.task_service import TaskService
from knowledge.processor.import_process.services.import_file_service import ImportFileService

@lru_cache  #会存缓存
def get_task_service():
    task_service = TaskService()
    return task_service

@lru_cache
def get_import_file_service():
    import_file_service = ImportFileService(get_task_service())
    return import_file_service
