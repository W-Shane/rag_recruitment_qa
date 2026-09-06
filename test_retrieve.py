#test_retrieve.py
import os
import logging
from dotenv import load_dotenv
from embeddings import get_embeddings
from langchain_chroma import Chroma

load_dotenv()
logging.basicConfig(level=logging.INFO)

# EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL", "./models/text2vec-base-chinese")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", 3))

_vectorstore = None

# 使用与构建时完全相同的 Embedding 类
def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings(device="cpu")
        _vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="recruit_job"
        )
    return _vectorstore

def search_jobs(query: str, top_k: int = TOP_K):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search_with_score(query, k=top_k)
    return [
        {"content": doc.page_content, "metadata": doc.metadata, "score": score}
        for doc, score in docs
    ]

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 离线岗位语义搜索工具（完全本地，无需大模型API）")
    print("输入 q 退出程序")
    print("=" * 60)

    while True:
        user_input = input("\n请输入搜索内容：").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            print("程序退出。")
            break
        if not user_input:
            continue

        results = search_jobs(user_input)
        if not results:
            print("未找到匹配岗位")
            continue

        for idx, item in enumerate(results, start=1):
            print(f"\n------【结果 {idx}】（相似度得分: {item['score']:.4f}）------")
            # 打印元数据
            meta = item["metadata"]
            print(f"公司: {meta.get('公司名称', 'N/A')}")
            print(f"职位: {meta.get('职位名称', 'N/A')}")
            print(f"薪资: {meta.get('薪资范围', 'N/A')}")
            print(f"地点: {meta.get('地点', 'N/A')}")
            # 也可显示完整内容（可选）
            # print(f"\n完整描述:\n{item['content']}")
