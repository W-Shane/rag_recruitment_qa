import os
import sys
import json
import logging
import jieba
from collections import Counter

# 把项目根目录加入 sys.path，保证能 import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.retriever import HybridRetriever
from core.config import settings

# ============ 日志 ============
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 停用词（过滤无意义助词，不参与关键词统计）
STOP_WORDS = {"的", "了", "是", "在", "和", "与", "及", "也", "都", "有", "就", "等", "可以", "需要"}

def extract_keywords(text: str):
    """对答案分词，提取有效关键词，过滤停用词"""
    words = jieba.lcut(text)
    keywords = [w for w in words if len(w) >1 and w not in STOP_WORDS]
    # 新增这一行打印，看关键词
    logger.info(f"提取关键词：{keywords}")
    # 去重
    return list(set(keywords))

def calc_context_recall(ground_truth:str, retrieved_docs:list[str]) -> float:
    """
    手写版本：计算召回率
    ground_truth：标准答案
    retrieved_docs：检索返回的文档文本列表
    """
    gt_keywords = extract_keywords(ground_truth)
    if len(gt_keywords) == 0:
        return 0.0

    # 把所有检索文档合并成一段文本
    all_retrieved_text = "\n".join(retrieved_docs)
    hit_count = 0
    for kw in gt_keywords:
        if kw in all_retrieved_text:
            hit_count +=1
    recall = hit_count / len(gt_keywords)
    return round(recall,4)

# ============ 兼容多种 retriever 返回格式 ============
def extract_content(doc):
    """
    从任意格式的检索结果中提取文本内容。
    支持：
      - langchain Document 对象（有 .page_content 属性）
      - tuple: (Document, score) 或 (content_str, metadata_dict)
      - dict: {'content': '...'} 或 {'page_content': '...'}
      - str: 直接返回
      - 嵌套 tuple: 递归处理
    """
    if doc is None:
        return ""
    if hasattr(doc, "page_content"):
        return doc.page_content
    if isinstance(doc, tuple):
        if len(doc) == 0:
            return ""
        return extract_content(doc[0])
    if isinstance(doc, dict):
        return doc.get("content") or doc.get("page_content") or ""
    if isinstance(doc, str):
        return doc
    logger.warning(f"未知的文档类型: {type(doc)}, 内容: {doc}")
    return ""


def main():
    # ============ 1. 加载 QA 数据集 ============
    qa_path = os.path.join(os.path.dirname(__file__), "qa_dataset.json")
    if not os.path.exists(qa_path):
        logger.error(f"评测数据集不存在: {qa_path}")
        return

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    logger.info(f"加载了 {len(qa_pairs)} 条 QA 对")

    questions = [item["question"] for item in qa_pairs]
    ground_truths = [item["answer"] for item in qa_pairs]

    # ============ 2. 初始化检索器 & 生成 contexts ============
    logger.info("初始化 HybridRetriever ...")
    retriever = HybridRetriever()

    all_recall = []
    result_detail = []

    for i, q in enumerate(questions, start=1):
        gt = ground_truths[i-1]
        try:
            docs = retriever.retrieve(q)
        except Exception as e:
            logger.error(f"检索第 {i} 个问题失败: {q}\n错误: {e}")
            all_recall.append(0.0)
            result_detail.append({"question":q,"recall":0.0,"note":"检索失败"})
            continue

        if not docs:
            logger.warning(f"第 {i} 个问题没有检索到任何文档: {q}")
            all_recall.append(0.0)
            result_detail.append({"question":q,"recall":0.0,"note":"无检索结果"})
            continue

        extracted = [extract_content(d) for d in docs]
        extracted = [t for t in extracted if t.strip()]

        recall = calc_context_recall(gt, extracted)
        all_recall.append(recall)
        result_detail.append({
            "question": q,
            "ground_truth":gt,
            "recall": recall
        })

        logger.info(f"第 {i}/{len(questions)} 个问题完成检索，召回率={recall}")

    # ============ 汇总结果 ============
    avg_recall = sum(all_recall)/len(all_recall)
    logger.info(f"\n==== 评测汇总 ====")
    logger.info(f"样本总数：{len(all_recall)}")
    logger.info(f"平均Context Recall：{round(avg_recall,4)}")

    # ============ 保存报告 ============
    report_path = os.path.join(os.path.dirname(__file__), "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"样本总数：{len(all_recall)}\n")
        f.write(f"平均Context Recall：{round(avg_recall,4)}\n\n")
        for item in result_detail:
            f.write(f"问题：{item['question']}\n")
            f.write(f"召回率：{item['recall']}\n")
            f.write("-"*40+"\n")
    logger.info(f"评测报告已保存: {report_path}")


if __name__ == "__main__":
    main()
