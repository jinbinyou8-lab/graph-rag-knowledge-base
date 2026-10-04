import json

from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv
load_dotenv()

def get_llm_client(model_name:str = "deepseek-flash", json_mode:bool = False):
    """
    创建 DeepSeek 客户端

    Args:
        model_name: 模型名
        json_mode: 是否强制模型输出 JSON。
                   注意: DeepSeek 要求开启 json_object 时, 提示词里必须出现
                   "json" 这个单词, 否则会直接返回 400。
                   本项目的商品名识别要求"纯净输出", 所以默认关闭。

    Returns: llm客户端对象

    """

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    client = ChatDeepSeek(
        model=model_name,
        model_kwargs=model_kwargs
    )

    return client


