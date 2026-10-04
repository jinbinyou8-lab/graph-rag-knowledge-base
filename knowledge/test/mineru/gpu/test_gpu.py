import torch

"""
Mineru支持GPU加速
"""

print(f"是否支持GPU:{torch.cuda.is_available()}")
print(f"设备名:{torch.cuda.get_device_name()}")

# 1.默认找gpu加速
# 2.默认把用到和解析相关的模型都缓存到本地的c盘用户目录的.cache目录下
# 2.1 修改方式有两种： 1）自己从.cache目录中拷贝到指定的目录中 2）执行这个命令之前，通过环境变量设置指定的模型下载目录 HF_HOME/MODELSCOPE_CACHE
# 2.2 下载模型的命令：mineru-models-download    注意：会自动生成mineru.json

# 3.默认使用解析后端是混合模式（hybrid:pineline+vlm）
# mineru -p <input_path> -o <output_path>ut_path>





# mineru -p C:\\Users\\尤锦滨\\Desktop\\面向AI Agent的零信任安全机制设计.pdf -o F:\\agent_project_knowledgestore\\shopkeeper_brain\\knowledge\\processor\\output_temp_dir

