# core/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 路径
    data_path: str = "data/job.csv"
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "./models/text2vec-base-chinese"

    # 检索
    top_k: int = 5
    bm25_weight: float = 0.3  # RRF 融合权重
    vector_weight: float = 0.7

    # LLM（仅 qa_cli 需要）
    dashscope_api_key: str = ""
    qwen_model: str = "qwen-plus"
    qwen_llm_model: str = "qwen-plus"
    qwen_embedding_model: str = "text-embedding-v3"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 日志
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
