#内置库
import json
from pathlib import Path

from knowledge.processor.import_process.base import  BaseNode,setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError
class EntryNode(BaseNode):
    """
    实体节点
    位置: 整个导入流程的位置(第一位)
    作用: 对上传的文件类型做判断(.pdf文件/.md文件)
    """

    name = "entry"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        处理文件类型的检测
        Args:
            state:ImportGraphState

        Returns:ImportGraphState  该节点处理之后的节点状态

        """

        #1. 获取导入文件的路径以及文件所在的目录
        self.log_step("Step1","{获取文件路径}")
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")

        #2.简单校验一下文件路径以及所在目录
        self.log_step("Step2","{检查文件路径}")
        if not file_dir or not import_file_path:
            raise ValidationError("文件目录或者文件路径不存在",self.name)

        #3.使用标准的Path对象操作文件逻辑
        path = Path(import_file_path)

        #4. 获取上传文件的后缀 path.suffix直接拿到文件的后缀名
        suffix = path.suffix.lower()#转成小写

        #5. 判断文件后缀
        if suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        elif suffix == '.md':
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        else :
            self.logger.debug(f"文件类型{suffix}不支持")
            raise ValidationError(f"文件类型{suffix}不支持")

        #6. 获取文件的标题名,使用path.stem
        file_title = path.stem
        state['file_title'] = file_title

        return state

