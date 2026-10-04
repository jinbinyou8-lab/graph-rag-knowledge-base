import os
import subprocess
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import  BaseNode,setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError,FileProcessingError,PdfConversionError

class Pdf_To_Md_Node(BaseNode):
    """"
    pdf转node节点
    """
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state:

        Returns:

        """
        #1. 参数校验
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        #2. 利用MinerU工具解析pdf->md
        processed_code = self._execute_mineru(import_file_path, file_dir_path)
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF失败", self.name)

        #3. 获取md的path
        md_path = self._get_md_paths(import_file_path, file_dir_path)

        #4. 更新state 字典的md_path
        state["md_path"] = md_path

        #5. 返回state
        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """

        Args:
            state:该节点接收到的状态

        Returns:

        """
        self.log_step("step1", "对状态的路径输入参数做校验")

        # 1. 获取输入pdf文件路径
        import_file_path = state.get("import_file_path", "")
        # 2. 获取解析后的输出目录
        file_dir = state.get("file_dir", "")
        # 3. 校验输入的文件路径
        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)
        # 4. 用Path标准化
        import_file_path_opj = Path(import_file_path)

        # 5. 校验是否是一个真实的路径
        if not import_file_path_opj.exists():
            raise FileProcessingError("解析的文件路径不存在", self.name)

        # 6. 判断输出目录是否为空
        if not file_dir:
            # 默认目录兜底
            file_dir = import_file_path_opj.parent  # 只有经历过标准化的路径才能拥有parent

        # 7. 标准输出目录
        file_dir_path_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径:{import_file_path}")
        self.logger.info(f"输出的目录:{file_dir}")

        # 8. 返回 输出文件以及输出目录的标准path
        return import_file_path_opj, file_dir_path_obj

    def _execute_mineru(self, import_file_path, file_dir_path) -> int:
        """

        Args:
            self:
            import_file_path:解析后的文件路径
            file_dir_path:解析后的文件目录

        Returns:
            mineru -p <input_path> -o <output_path>ut_path>
        """
        # 执行命令行
        self.log_step("step2", "执行MinerU解析PDF")
        # 1. 构建命令行   mineru -p <input_path> -o <output_path>ut_path>
        cmd = [
            "mineru",
            "-p",
            str(import_file_path),
            "-o",
            str(file_dir_path),
            "--source",
            "local"
        ]

        import time
        process_start_time = time.time()

        # 2.构造子进程的环境变量
        # MinerU 会在本地临时起一个 mineru-api 服务(127.0.0.1:随机端口), 然后用 httpx 轮询任务状态。
        # httpx 默认 trust_env=True, 会经由 urllib.request.getproxies() 读取系统代理;
        # 本机开着 Clash 系统代理(127.0.0.1:7897)时, 连发往 127.0.0.1 的请求都会被交给代理,
        # 于是报 Failed to query task status ... 404 Not Found, MinerU 以非0退出。
        # 给子进程设置 NO_PROXY 即可让本地地址直连(此时 getproxies() 不再回退到注册表)。
        child_env = os.environ.copy()
        child_env["NO_PROXY"] = "127.0.0.1,localhost,::1"
        child_env["no_proxy"] = "127.0.0.1,localhost,::1"

        # 3.执行子进程, 子进程能够自动读取到主进程的环境变量
        proc = subprocess.Popen(
            args=cmd,
            env=child_env,
            stdout=subprocess.PIPE,  # 主进程开一根管子,专门用来读取子进程的输出(错误日志+正常日志都从这根管子出来)
            stderr=subprocess.STDOUT,  # 把子进程的错误日志并到上面那根 stdout 管子里,两者合并成一路输出
            errors="replace",  # 保证遇到一些错误没办法解释的时候不会崩,用一些会的内容进行替换   比如,输出 ?
            text=True,  # 输出字符串
            encoding="utf-8",
            bufsize=1  # 按行缓存区, 只要一行满了就换行
        )

        # 4. 获取日志信息, stdout是流式产生,所以需要迭代输出
        for outlog in proc.stdout:
            self.logger.info(f"执行Mineru产生的日志:{outlog}")

        # 5. 等待子进程处理完(主进程等待子进程做完
        # ),做完会返回处理码0,如果不是0就没有处理完
        processed_code = proc.wait()

        process_end_time = time.time()

        if processed_code == 0:
            self.logger.info(f"MinerU成功解析PDF文件:{import_file_path.name}, 耗时:{process_end_time - process_start_time:.2f}") #因为路径经过path标准化之后就可以调用.name属性
        else:
            self.logger.error(f"MinerU解析PDF文件失败: {import_file_path.name}")
        # 5. 返回状态码
        return processed_code

    def _get_md_paths(self, import_file_path: Path, file_dir_path: Path) -> str:

        file_name = import_file_path.stem


        return str(file_dir_path / file_name / "hybrid_auto" / f"{file_name}.md")







if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = Pdf_To_Md_Node()

    test_entry_state = {
        "file_dir": r"F:\agent_project_knowledgestore\shopkeeper_brain\knowledge\processor\import_process",
        "import_file_path": r"C:\Users\尤锦滨\Desktop\面向AI Agent的零信任安全机制设计.pdf"
    }

    pdf_to_md_node(test_entry_state)