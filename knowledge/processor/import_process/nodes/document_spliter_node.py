from knowledge.processor.import_process.base import  BaseNode,setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.config import get_config
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Tuple,List,Dict,Any
from knowledge.utils.markdown_utils import MarkdownTableLinearizer
import re
import json
from pathlib import Path

class DocumentSpliterNode(BaseNode):

    name = "document_split_name"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        #加载---->打散(1. 嵌入模型语义更准确 2. 注入元数据 3. 多路召回 4. 性能,成本高 ----->减少llm的幻觉, 提高我们的检索质量)---->组合

        #1. 获取参数
        md_content, file_title, max_content_length, min_content_length = self._get_inputs(state)

        #2. 根据标题进行切割
        sections, has_title = self._split_by_headings(md_content, file_title)

        # 3. 处理(切分,合并)
        # 3.1 section内容过长, 继续进行二次切割
        # 3.2 section内容过短, 看看能不能合并(如果不能合并不合并, 反之则合并)
        final_chunks = self._split_and_merge(sections, max_content_length, min_content_length)

        #4. 组装
        chunks = self._assembel_chunk(final_chunks)

        #5. 更新state: chunks
        state['chunks'] = chunks

        return state


    def _get_inputs(self, state: ImportGraphState)->Tuple[str,str,int,int]:

        self.log_step("step1","获取参数")

        config = get_config()

        #1. 获取md_content
        md_content = state.get("md_content")

        #2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        #3. 获取文件标题
        file_title = state.get("file_title")

        #4. 校验最大最小值
        if config.max_content_length<=0 or config.min_content_length<=0 or config.max_content_length<=config.min_content_length:
            raise ValueError(f"长度校验失败")
        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content:str, file_title: str) -> Tuple[List[dict], bool]:

        self.log_step("step2","对md进行切分")

        """
        根据MD的标题(1-6)进行切分
        Args:
            md_content: md内容
            file_title: 文档的名字

        Returns:
            Tuple[List[dict], bool]
                List[dict]: sections
                bool: md 文档是否有标题
                {
                "title": "# 第一章",
                "body": "正文内容...",
                "file_title": "万用表",
                "parent_title": "# 第一章"
                }
        """
        self.log_step("step1", "根据标题进行切分")
        #1. 定义变量
        in_fence = False
        body_lines = []
        current_level = 0
        current_title = ""
        hierarchy = [""] * 7    # 七个长度 但是第一个(0号位置)不用

        #2. 定义正则表达式(group1: 标题的语法符号#[最少1个# 最多6个#])
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        #3.切分
        content_lines = md_content.split("\n")

        def _flush(prev_title, prev_level, prev_body_lines):

            self.log_step("step3","对按行切分的md进行刷新并返回标题区间")

            """
            结算上一个标题的内容。
            注意：这里传入的是上一个标题和它的正文，而不是当前的！
            """
            body = "\n".join(prev_body_lines)
            if not prev_title and not body:
                return None  # 空数据不生成 section

            # 找上级标题（这里用文件标题兜底）//只是为了兜底的
            parent_title = file_title
            # 如果存在父标题，就记录父标题名字（注意，这里不要覆盖为 file_title，要拿hierarchy里的字符串）
            for i in range(prev_level - 1, 0, -1):
                if hierarchy[i]:
                    parent_title = hierarchy[i]  # 取上一级真正的标题
                    break

            # 如果没有任何父标题，父标题就是自己（或者根据业务需求定制）
            if not parent_title:
                parent_title = prev_title if prev_title else file_title

            return {
                "title": prev_title,  # 这里才是真正的标题
                "body": body,  # 属于该标题的正文
                "file_title": file_title,
                "parent_title": parent_title
            }

        sections = []

        for content_line in md_content.split("\n"):
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):
                in_fence = not in_fence
                continue

            if in_fence:
                body_lines.append(content_line)  # 内容照样收进 body
                continue  # 只是跳过下面的标题识别 ← 门在这里

            match = heading_re.match(content_line)

            if match:
                # === 遇到新标题，先结算上一个标题 ===
                # 把 old_title 和 old_body 传给 _flush
                section = _flush(current_title, current_level, body_lines)
                if section:
                    sections.append(section)  # 添加到结果列表里

                # === 开始处理新标题 ===
                level = len(match.group(1))
                current_title = content_line
                current_level = level
                hierarchy[level] = current_title
                body_lines = []  # 清空正文，准备收集新标题的内容

                # 清除更深的层级
                for i in range(level + 1, 7):
                    hierarchy[i] = ""
            else:
                body_lines.append(content_line)

        # === 循环结束，结算最后一个标题 ===
        section = _flush(current_title, current_level, body_lines)
        if section:
            sections.append(section)

        has_title = any(s["title"] for s in sections)  # 判断是否有标题
        return sections, has_title

    def _split_and_merge(self, sections:List[Dict[str, Any]], max_content_length:int, min_content_length:int):
        """

        Args:
            sections: 根据一级标题切分后的所有section(章节)块
            max_content_length: 每一个section内容[title+body]的最大长度
            min_content_length: 每一个section内容[title+body]的最小长度

        Returns:
                List[section]
        """
        self.log_step("step4","切分长内容以及合并短内容")

        #1. 切分
        current_sections = []
        for section in sections:
            current_sections.extend(self.split_long_section(section, max_content_length))

        #2. 合并
        final_sections = self.merge_short_section(current_sections, min_content_length)

        #3. 返回
        return final_sections

    def split_long_section(self, section:Dict[str, Any], max_content_length: int):
        self.log_step("step5","进行长内容的切分")
        #1. 获取section对象属性
        title = section.get("title")        #不可能为空
        body = section.get("body")
        file_title = section.get("file_title")
        parent_title = section.get("parent_title")

        #2. 判断表格(检查一下有没有表格,如果要表格需要进行处理,调用markdown_utils)
        if "<table>" in body:
            self.logger.info("检查到了表格........")
            body = MarkdownTableLinearizer.process(body)

        #3. 对标题进行校验(有可能标题过长,所以要对标题设置一个阈值)
        TITLE_MAX_LENGTH = 50   #定义最大的长度是50
        if len(title) > TITLE_MAX_LENGTH:
            self.logger.warning(f"文件{file_title}对应的{title}过长")
            title = title[:50]

        #4. 拼接title前缀
        title_prefix = f"{title}\n\n"

        #5. 计算总长度(标题前缀的长度 + 切分块内容的长度)
        total_length = len(title_prefix) + len(body)

        #6. 判断当前section是不是小于等于最大的限制
        if total_length <= max_content_length:
            return [section]

        #7. 计算body可用的长度
        body_length = max_content_length - len(title_prefix)

        if body_length <= 0:
            return [section]

        #8. 切分   # 8.1 对谁切【body】 # 8.2 用谁切[1)手写 2)langchain提供的切分器:TextSplitter/TokenSplitter/LenghtSplitter/）:chunkoverlap:块与块之间的重叠/递归切分器
        #8.1 定义递归的文档切分器对象
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=body_length,
                                                       chunk_overlap=0,
                                                       separators=["\n\n", "\n", " ", ",", ""],
                                                       keep_separator=False #默认是True就是在文本中会保留这些分隔符,选用False就不会保留
                                                       )

        #8.2 进行切割
        texts = text_splitter.split_text(body)

        #8.3 判断切分器返回的长度[如果是0,说明body没有内容,如果是1,说明body里的内容的总数没有超过最大阈值, 如果大于1说明body内文本很多,所以我们要分别进行存储]
        if len(texts)<=1:
            return [section]

        sub_section = []
        for index, text in enumerate(texts):
            sub_section.append({
                "title": title + "-"+f"{index+1}",
                "body": text,
                "file_title":file_title,
                "parent_title":parent_title,
                "part":f"{index+1}" #用来记录是切分的第几块
            })

        return sub_section

    def merge_short_section(self, current_sections:List[Dict[str,Any]], min_content_length:int):
        """
        贪心累加算法
        俩个局限性:
            1. 可能会撑爆阈值
            2. 会有孤儿小块
        Args:
            current_sections:
            min_content_length:

        Returns:

        """
        self.log_step("step4","进行短内容的合并")
        current_section = current_sections[0]
        fina_sections = []  #最终的箱子

        for next_section in current_sections[1:]:

            #同源(在这里就是要在相同的父标题下)
            same_parent = (current_section["parent_title"] == next_section["parent_title"])
            if same_parent and len(current_section['body']) < min_content_length:
                #body的合并(更新当前section_body)
                current_section['body'] = (
                    current_section.get("body").rstrip() + "\n\n" + next_section.get("body").lstrip()
                )
                #简单的能涵盖住合并起来的内容的标题
                current_section["title"] = current_section["parent_title"]

                current_section['part'] = 0     #保证只要有合并的可能,我们都给第一个的part赋值为0,方便后续的合并,因为后续合并的时候不在意你的part是多少,会有一个新的part来计数


            else :
                #1.将原来current_section进行封箱
                fina_sections.append(current_section)

                #2. 更新next_section
                current_section = next_section

        #最后一个人(封装起来)
        fina_sections.append(current_section)

        #对所有section的part进行处理(为每一个父节点设置对应的part计数器)
        part_counter = {}
        result = []
        for fina_section in fina_sections:
            if "part" in fina_section:
                #获取section的父标题
                parent_title = fina_section.get("parent_title")

                #给计数器赋值
                part_counter[parent_title] = part_counter.get(parent_title, 0) +1

                #获取计数器的值
                new_part = part_counter[parent_title]

                fina_section['part'] = new_part
                fina_section['title'] = fina_section["title"] + '-' + f"{new_part}"
            result.append(fina_section)
        return result

    def _assembel_chunk(self, final_chunks:List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        最终组合好的chunk
        Args:
            final_chunks:

        Returns:

        """
        chunks = []
        for chunk in final_chunks:
            #1. 获取chunk的信息
            title = chunk.get("title")
            file_title = chunk.get("file_title")
            parent_title = chunk.get("parent_title")
            body = chunk.get("body")
            content = f"{title}\n\n{body}"

            assemble_chunk = {
                "title":title,
                "file_title":file_title,
                "parent_title":parent_title,
                "content":content
            }

            #2. 判断是否有part,有的话要加入
            if "part" in chunk:
                assemble_chunk["part"] = chunk.get("part")

            chunks.append(assemble_chunk)
        return chunks

