# embeddings.py
import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# 加载环境变量（确保读取 .env 中的路径）
load_dotenv()

@lru_cache(maxsize=1)
def get_embeddings(model_name: str = None, device: str = "cpu"):
    """
    获取统一的 Embedding 实例（单例工厂模式）
    :param model_name: 模型名称或本地路径，默认从 .env 读取
    :param device: 运行设备 'cpu' 或 'cuda'
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "./models/text2vec-base-chinese")

    # 统一封装参数，可加 trust_remote_code 或 tokenizer 参数，只改这里
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )