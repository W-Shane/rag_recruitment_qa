# core/retriever.py
import os
import json
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.embeddings import get_embeddings
from core.config import settings
import jieba


class HybridRetriever:
    def __init__(self):
        # 1. 加载向量库
        self.embeddings = get_embeddings(device="cpu")
        self.vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self.embeddings,
            collection_name="recruit_job"
        )

        # 2. 加载所有文档用于 BM25
        self.docs = self._load_all_docs()

        # 空库保护，防止ZeroDivisionError
        if not self.docs:
            raise ValueError("向量库中没有任何文档！请先执行build_vector_db构建知识库")

        self.bm25 = BM25Okapi([jieba.lcut(doc.page_content) for doc in self.docs])
        # 建立id→document映射，过滤掉id为None的文档
        self.id2doc = {}
        for doc in self.docs:
            doc_id = doc.metadata.get("id")
            if doc_id is not None:
                self.id2doc[doc_id] = doc

    def _load_all_docs(self) -> List[Document]:
        """从 Chroma 读取全部文档，组装成完整Document对象"""
        all_data = self.vectorstore.get()
        docs_text = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])

        doc_list = []
        for text, meta, doc_id in zip(docs_text, metadatas, ids):
            doc_list.append(
                Document(page_content=text, metadata={**(meta or {}), "id": doc_id})
            )
        return doc_list

    def _reciprocal_rank_fusion(self, vector_results: List[Document], bm25_results: List[Document], k=60):
        """RRF 倒数排序融合算法，返回排序后的doc_id列表，过滤id=None"""
        scores = {}
        for rank, doc in enumerate(vector_results):
            doc_id = doc.metadata.get("id")
            if doc_id is None:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + settings.vector_weight / (k + rank + 1)
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.metadata.get("id")
            if doc_id is None:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + settings.bm25_weight / (k + rank + 1)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return sorted_ids

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        top_k = top_k or settings.top_k

        # 1.向量检索，返回Document对象
        vector_docs: List[Document] = self.vectorstore.similarity_search(query, k=top_k * 2)

        # 2.BM25关键词检索，得到Document对象列表
        query_tokens = jieba.lcut(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
        bm25_docs = [self.docs[idx] for idx in bm25_indices]

        # 3.执行RRF融合排序
        sorted_doc_ids = self._reciprocal_rank_fusion(vector_docs, bm25_docs)
        selected_ids = sorted_doc_ids[:top_k]

        # 根据id取回document，构造返回结果（简单递减模拟分数）
        out = []
        for idx, doc_id in enumerate(selected_ids):
            doc = self.id2doc.get(doc_id)
            if doc is None:
                continue
            score = round(1.0/(idx+1),4)
            out.append((doc, score))
        return out
