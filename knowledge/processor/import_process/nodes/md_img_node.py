import logging
import os
import re
from pathlib import Path
from typing import Tuple,List
from knowledge.processor.import_process.base import  BaseNode,setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError
from knowledge.processor.import_process.config import get_config
from openai import OpenAI
from knowledge.utils.minio_util import get_minio_client
from minio import Minio

class MarkDownImageNode(BaseNode):
    """
    处理MarkDown图片节点类
    """
    name = "md_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state:上一个节点处理之后的state的最新状态

        Returns:当前节点处理之后的state的最新状态(处理md_content=process_md)

        """
        #1. 传入配置对象
        config = get_config()
        #2. 处理文件路径(返回值:1.md的内容, 2.md的path, 3. 图片目录)
        md_content, md_path_obj, image_dir = self._get_img_md_content(state)
        if not image_dir.exists():
            #如果没有图片目录,图片就不需要处理了,直接更新state的md_content
            self.logger.warning(f"文件{md_path_obj.name}暂无图片需要处理")
            state['md_content'] = md_content
            return state

        #3. 扫描并处理图片(最复杂的部分)
        target_images_context = self._scan_images_and_context( image_dir, md_content,config)

        #4. 用VLM给图片生成描述(摘要)
        images_summaries = self._extract_img_summary(md_path_obj.stem, target_images_context, config)

        #5. 符合函数
        #5.1 本地图片上传到minio(图片在minio上的地址)----> remote_url(图片远程dizhi)
        #5.2 替换md中的图片的本地url, 以及vlm生成的摘要

        new_md_content = self._upload_img_and_update_md(md_path_obj.stem,md_content, images_summaries, target_images_context, config)

        #6. 更新state
        state["md_content"] = new_md_content

        #7. 返回更新后的状态
        return state



    def _get_img_md_content(self, state:ImportGraphState) -> Tuple[str, Path, Path]:
        """

        Args:
            state:上一个节点处理之后的state的最新状态

        Returns:
            md_content: md的内容
            md_path: md的路径
            image_dir: 图片目录
        """

        self.log_step("step1","读取md内容以及构建图片目录")
        #1. 获取md_path
        md_path = state.get('md_path','')

        #2. 判断路径是否有内容
        if not md_path:
            raise ValidationError("md不存在",self.name)

        #3. 标准化处理路径
        md_path_obj = Path(md_path)

        #4. 判断路径是否有效
        if not md_path_obj.exists():
            raise FileProcessingError("md路径不是有效路径",self.name)

        #5. 读取md内容
        with open(md_path_obj,"r",encoding="utf-8") as f:
            md_content = f.read()   #文件全部读完

        #6. 构建图片目录
        image_dir = md_path_obj.parent / "images"

        #7. 返回
        return md_content, md_path_obj, image_dir

    def _scan_images_and_context(self, image_dir: Path, md_content, config) -> List[Tuple[str,str,Tuple[str,str,str]]]:
        """
        扫面并处理图片
        返回所有有效图片的丰富信息(image_name, image_path, 图片的上下文)
        图片的上下文主要是帮助llm来理解图片     找上文和下文的策略,通过max_char_number(max_total) 丢失语义 混合策略(max_total)
        最终获取上下文的策略: 1.先找到当前图片最近的一个标题(#,##,###,####,#####)(定位到标题的位置以及标题的内容)
        2. 从图片内容的上一行开始向上找一直找到最近的下一个标题的下一行
        3. 根据开始索引和结束索引定位到俩个索引间的内容
        4.利用最大字符数和区间来选择最终留下多少

        Args:
            image_dir:  图片目录
            md_content: md内容
            config: 配置信息
        Returns:

        """
        self.log_step("step2", f"扫面图片文件目录{image_dir}")
        #1. 遍历图片文件目录
        target_images_context = []

        for img_name in os.listdir(image_dir):
            file_text = os.path.splitext(img_name)[1]      #os.path.splitext()获得的是一个元组,第一个元素是名称,第二个是后缀
            #1.1 如果文件不是有效文件,直接跳过
            if file_text not in config.image_extensions:
                continue    #就直接开始处理下一个文件

            #1.2 构建image_path 转成字符串
            img_path = str(image_dir / img_name)

            #1.3 构建图片的上下文
            img_context = self._find_img_context_with_limit(md_content, img_name, config.img_content_length)
            if not img_context:
                self.logger.warning("MD文件中暂未提取到有效的图片")
                continue
            #1.4 提取到当前图片的唯一上下文内容
            primary_img_context = img_context[0]    # 因为有概率会重复,所以仅仅取第一个值

            #1.5 存储到列表中
            target_images_context.append((img_name, img_path, primary_img_context))
        self.logger.info(f"找到{len(target_images_context)}有效的图片")
        return target_images_context

    def _find_img_context_with_limit(self,md_content:str, img_name:str, max_chars:int =200) -> List[Tuple[str,str,str]]:     #这部分是难点
        """
        从MD文档中提取处图片上下文信息
        思路: 使用正则来查找md的位置
        Args:
            md_content: 要操作的md
            img_name:   要定位的图片名字
            max_chars: 最大字符数
        Returns:
        """
        """
        # 1. 定义正则的规则(从md找到图片) 标准的图片在md中的语法结构: ![图片的描述](images/aaa.jpg"提示")
        # 1.1 第一部分
        # r:python不要在对正则中的字符做转义了
        # ! md语法
        # [ (需要正则转义): 在正则中[代表的字符集 (a-z A-Z 0-9 + /)
        # . 任意字符
        # * (+) 任意字符出现的数量 0个或者多个  (+) 至少要有一个
        # ? (非贪婪模式: 佛系)  不加? 就是贪婪模式 (加班 积极向上)
        # ]
        # () (需要正则转义):在正则中()代表捕获组
        # img_name=text.jpg  :金标准: .后置的名字一定做escape处理
        """
        #1. 定义正则的规则(从md找到图片)
        re_pattern = re.compile(r"!\[.*?\]\((.*?" + re.escape(img_name) + r".*?)\)")

        #2. 从md中定位图片位置(按行切分md的内容, 遍历每一行, 看是否满足图片的正则规则)
        md_lines = md_content.split("\n")
        img_context = []
        for line_idx, line in enumerate(md_lines):  #enumerate的返回值是(索引, 元素值)

            #a. 如果找到的这一行不是图片就继续找下一行
            if not re_pattern.search(line):
                continue

            #b. 找到了图片位置     a.1 然后就要定位这张图片最近的标题     a.2 再提取上文内容 再提取下文内容

            ####找图片的上文
            #b.1 找标题    (re.match(r"^#{1,6}\s+", lines[i]):) 标题的正则表达式
            head_title = "" #初始的标题内容
            head_index = -1 #初始的标题索引
            for i in range(line_idx - 1, -1, -1):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    head_title = md_lines[i]
                    head_index = i
                    break
            #b.2  定义要截取上文的索引(就是大标题的下一行与图片的上一行之间的区间)为什么下文不用减去1,因为在切片的右边是取不到的
            pre_content_start_index = head_index + 1
            pre_content_end_index = line_idx
            pre_content = md_lines[pre_content_start_index:pre_content_end_index]
            img_pre_context = self._extract_img_context_with_limit(pre_content,max_chars,direction = "front")

            ####找图片的下文(下文不用记录标题内容)
            md_len = len(md_lines)
            section_index = md_len
            for i in range(line_idx+1, md_len):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    section_index = i
                    break

            #   定义要截取的下文的索引
            post_content_start_index = line_idx + 1
            post_content_end_index = section_index
            post_content = md_lines[post_content_start_index:post_content_end_index]

            img_post_context = self._extract_img_context_with_limit(post_content,max_chars,direction = "end")

            img_context.append((head_title, img_pre_context, img_post_context))     #如果会重复用一个图片或者上下文可能会有重复的三元组
        return img_context

    def _extract_img_context_with_limit(self, extract_content:list, max_chars:int, direction: str):
        """
        提取图片到上下标题(最近)之间的内容(段落内容)
        direction: front: 自下而上
        direction: end: 自上而下
        如何从给定的内容中找段落
        策略: md中的段落 \n分割     补充:行与行之间有俩个空格
        Args:
            extract_content: 提取到的内容
            max_chars:  最大的字符数
            direction:  方向选择
        Returns:
        """
        #1. 切分内容的段落


        current_paragraph = []  #存储当前遍历到的内容
        final_paragraph = []    #存储最终遍历到的段落(多个段落)

        #2. 遍历每一行 收集段落
        for line in extract_content:
            clean_strip = line.strip()
            if not clean_strip:     #自然而然的段落分割
                if current_paragraph:
                    final_paragraph.append("\n".join(current_paragraph))    #"\n"是这些段落链接起来的方式
                    current_paragraph = []
            else:
                if re.match(r"^!\[.*?\]\(.*?\)$", clean_strip): #遇到其它图片,就把前面遍历到的行形成一个段落
                    if current_paragraph:
                        final_paragraph.append("\n".join(current_paragraph))  # "\n"是这些段落链接起来的方式
                        current_paragraph = []
                    continue
                current_paragraph.append(line)
        #处理最后一段且没有空行也要加到里面不能忘记
        if current_paragraph:
            final_paragraph.append("\n".join(current_paragraph))


        #从下向上取值
        if direction == "front":
            final_paragraph.reverse()


        #3. 判断final中收集到的段落字符的长度是否超过了阈值,(截取)
        total = 0
        selected = []
        for para in final_paragraph:
            para_len = len(para)

            if total + para_len > max_chars and selected:
                break
            selected.append(para)
            total += para_len

        #如果是向下往上取值,最后还要反转回来
        if direction == "front":
            selected.reverse()

        #4. 返回selected
        return "\n\n".join(selected)

    def _extract_img_summary(self,document_title: str, target_images_context: List[Tuple[str,str,Tuple[str,str,str]]], config):
        """
        为所有图片生成图片摘要 VLM(视觉语言模型)
        Args:
            document_title: (文件) 文档名字
            target_images_context:  所有图片信息
            config:

        Returns: Dict{"图片名字1":"摘要1","图片名字2":"摘要2"}
        """
        self.log_step("step3", "准备提取图片摘要")
        summeries = {}

        #1. 构造OpenAI客户端
        try:
            client = OpenAI(api_key = os.getenv("DEEPSEEK_API_KEY"),
                            base_url = os.getenv("DEEPSEEK_BASE_URL")
                            )
        except Exception as e:
            logging.error(f"VLM客户端创建失败")
            return summeries

        #2. 发送请求(提取摘要)

        for img_name, img_path, images_context in target_images_context:
            summary = self._get_img_summary(config, client, document_title, img_path, images_context)
            summeries[img_name] = summary

        #3. 返回映射表
        return summeries

    def _get_img_summary(self, config, client, document_title: str, img_path:str, images_context:Tuple[str,str,str])->str:
        #1. 解包images_context构建上下文
        section_title, pre_context, post_context = images_context

        #2. 判断上下文
        context_parts = []
        if section_title:
            context_parts.append(section_title)
        if pre_context:
            context_parts.append(pre_context)
        if post_context:
            context_parts.append(post_context)
        #3. 构建上下文
        final_context = "\n".join(context_parts) if context_parts else "暂无上下文可用"

        #4. 读取图片文件
        local_img_content = ""
        try:
            import base64
            with open(img_path, "rb") as  f:    #因为是字节流没办法直接接收,需要使用工具
                local_img_content = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return "暂无图片"

        #5. 发送请求
        try:
            response = client.chat.completions.create(
                model="deepseek-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                                为我的MarkDown文档中的图片生成一个简短的中文标题.
                                背景信息:
                                1. 所属文档标题:{document_title}
                                2. 图片上下文: {final_context}
                                请结合图片视觉内容和上述的上下文信息, 用中文简短总结这张图片的内容.
                                生成一个极端的中文标题(不要包含"照片"二字)
    """
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{local_img_content}"}
                            }
                        ]
                    }
                ]
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img_path}: {e}")
            return "暂无图片描述"

    def _upload_img_and_update_md(self, document_name, md_content, images_summaries, target_images_context, config):
        """
        上传图片到minio以及替换md中的图片url和摘要
        Args:
            md_content:
            images_summaries:
            target_images_context:

        Returns:

        """

        remote_urls = {}

        #1. 构建MinIO客户端
        minio_client = get_minio_client()

        if minio_client is None:
            self.logger.warning(f"无法将本地图片上传到minio")
            return md_content    #拿不到客户端,原样返回md,不做替换,避免写入假地址

        #2. 遍历图片信息列表
        for img_name, img_path, _ in target_images_context:
            #2.1 构建对象的名字
            object_name = f"{document_name}/{img_name}"
            try :
                # 2.2 开始上传(本地文件 -> minio, 用 fput_object)
                minio_client.fput_object(
                    config.minio_bucket,
                    object_name,
                    img_path
                )
                #2.3 手动拼接远程地址
                remote_url = config.get_minio_base_url() + "/" + config.minio_bucket + "/" + object_name
                self.logger.info(f"{img_name}图片上传到minio成功")
                remote_urls[img_name] = remote_url

            except Exception as e:
                self.logger.warning(f"{img_name}上传minio失败: {e}")
                remote_urls[img_name] = "http://minio_mock" + document_name + "/" + img_name #可选

        self.logger.info(f"成功上传{len(remote_urls)}图片到minio")

        #3. 替换(摘要和图片地址)
        new_md_content = md_content
        for img_name, images_summary in images_summaries.items():
            #3.1 提取远程地址
            remote_url = remote_urls.get(img_name)

            if not remote_url:
                continue    #没有就不进行替换

            #3.2 替换url和摘要
            replace_pattern = re.compile(
                r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)",
                re.IGNORECASE
            )
            new_md_content = replace_pattern.sub(f"![{images_summary}]({remote_url})",new_md_content)

        #4. 全部图片替换完之后再返回(原来写在了for里面,只会替换第一张,并且汇总为空时会返回None)
        return new_md_content





#测试
if __name__ == "__main__":
    setup_logging()
    img_md_node = MarkDownImageNode()

    state = {
        "md_path": r"F:\agent_project_knowledgestore\shopkeeper_brain\knowledge\processor\import_process\万用表的使用\hybrid_auto\万用表的使用.md"
    }

    res = img_md_node.process(state)
    print(res)



