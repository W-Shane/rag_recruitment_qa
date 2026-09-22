# scripts/test_retrieve.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retriever import HybridRetriever
from core.config import settings

retriever = HybridRetriever()

def search_jobs(query: str):
    docs = retriever.retrieve(query)
    return [{"content": d.page_content, "metadata": d.metadata, "score": s} for d, s in docs]

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
