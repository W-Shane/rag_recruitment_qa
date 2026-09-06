#build_vector_db.py
import os
import logging
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embeddings import get_embeddings

# 加载环境变量
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    data_path = os.getenv("DATA_PATH", "data/job.csv")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")

    logger.info(f"使用Embedding模型: {embedding_model_name}")
    # embeddings = HuggingFaceEmbeddings
    embeddings = get_embeddings(
        model_name=embedding_model_name,
        # model_kwargs={"device": "cpu"}
        device = "cpu" # 直接传设备名
    )

    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在：{data_path}")
        return

    # df = pd.read_csv(data_path)
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='gbk')
    documents = []

    for _, row in df.iterrows():
        content = f"""职位名称：{row.get('职位名称','')}
公司：{row.get('公司名称','')}
薪资范围：{row.get('薪资范围','')}
地点：{row.get('地点','')}
工作经验：{row.get('工作经验','')}
学历要求：{row.get('学历要求','')}
岗位标签：{row.get('岗位标签','')}
公司类型：{row.get('公司类型','')}
公司规模：{row.get('公司规模','')}
"""
        doc = Document(
            page_content=content.strip(),
            metadata={
                "职位名称": row.get("职位名称",""),
                "公司名称": row.get("公司名称",""),
                "薪资范围": row.get("薪资范围",""),
                "地点": row.get("地点","")
            }
        )
        documents.append(doc)

    logger.info(f"一共加载 {len(documents)} 条岗位数据")

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="recruit_job"
    )
    logger.info(f"向量库构建完成，保存路径：{persist_dir}")

if __name__ == "__main__":
    main()
