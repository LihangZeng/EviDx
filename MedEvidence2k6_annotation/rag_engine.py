# consultant/rag_engine.py
import os
import json
import glob
from typing import List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from openai import OpenAI 
from rank_bm25 import BM25Okapi  

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
STATPEARLS_COLLECTION = os.getenv("QDRANT_COLLECTION", "stat_ptnchld")

EMBED_API_KEY = os.getenv("EMBED_API_KEY", "EMPTY")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:8000/v1")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "qwen3-embedding")

TEXTBOOK_DIR = os.getenv("TEXTBOOK_DIR", "")

class TextbookBM25Retriever:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents = [] 
        self.tokenized_corpus = []
        self.bm25 = None
        self._load_and_index()

    def _load_and_index(self):
        print(f"[RAG] Loading Textbooks from {self.data_dir}...")
        if not self.data_dir or not os.path.isdir(self.data_dir):
            print("[RAG Warning] TEXTBOOK_DIR is not configured or does not exist.")
            return
        jsonl_files = glob.glob(os.path.join(self.data_dir, "*.jsonl"))
        
        for file_path in jsonl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if "contents" in item:
                            self.documents.append(item)
                            self.tokenized_corpus.append(item["contents"].lower().split())
            except Exception as e:
                print(f"[RAG Warning] Failed to load {file_path}: {e}")

        if self.tokenized_corpus:
            print(f"[RAG] Building BM25 Index for {len(self.documents)} textbook chunks...")
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            print("[RAG] BM25 Index built.")
        else:
            print("[RAG Error] No textbook data found.")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        if float(np.max(scores)) <= 0:
            return []

        top_k = min(top_k, len(self.documents))
        top_n_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_n_indices:
            doc = self.documents[idx]
            results.append({
                "content": doc.get("contents", doc.get("content", "")),
                "title": doc.get("title", "Unknown Textbook"),
                "source_type": "Textbook (BM25)",
                "score": float(scores[idx]),
                "id": doc.get("id", str(idx))
            })
        return results

class RAGEngine:
    def __init__(self):
        try:
            self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            print(f"[Consultant] Qdrant connected ({STATPEARLS_COLLECTION}).")
        except Exception as e:
            print(f"[Consultant] Warning: Qdrant Error: {e}")
            self.qdrant_client = None

        try:
            self.embed_client = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_BASE_URL)
        except Exception as e:
            self.embed_client = None
        self.textbook_retriever = TextbookBM25Retriever(TEXTBOOK_DIR)

    def get_embedding(self, text: str) -> List[float]:
        if not self.embed_client: return []
        try:
            text = text.replace("\n", " ")
            resp = self.embed_client.embeddings.create(model=EMBED_MODEL_NAME, input=[text])
            return resp.data[0].embedding
        except Exception as e:
            print(f"[RAG Warning] Embedding failed: {e}")
            return []

    def _search_qdrant(self, query: str, top_k: int) -> List[Dict]:
        if not self.qdrant_client: return []
        vector = self.get_embedding(query)
        if not vector: return []

        try:
            hits = self.qdrant_client.query_points(
                collection_name=STATPEARLS_COLLECTION,
                query=vector,
                limit=top_k, 
                with_payload=True
            ).points
            
            results = []
            for hit in hits:
                payload = hit.payload
                content = payload.get("parent_context") or payload.get("child_content") or payload.get("text") or ""
                results.append({
                    "content": content,
                    "title": payload.get("doc_title", "Unknown"),
                    "source_type": "StatPearls (Vector)",
                    "score": float(hit.score),
                    "id": str(hit.id)
                })
            return results
        except Exception as e:
            print(f"[RAG Error] Qdrant search failed: {e}")
            return []

    def _rrf_fusion(self, list_a: List[Dict], list_b: List[Dict], k=60) -> List[Dict]:
        fused_scores = {}
        doc_map = {}

        def _safe_str(x) -> str:
            if x is None:
                return ""
            return x if isinstance(x, str) else str(x)

        def _doc_key(doc: Dict) -> str:
            title = _safe_str(doc.get("title", ""))
            content = _safe_str(doc.get("content", ""))
            source = _safe_str(doc.get("source_type", ""))
            doc_id = _safe_str(doc.get("id", ""))
            return f"{source}:{doc_id}" if doc_id else f"{source}:{title}:{content[:50]}"

        for rank, doc in enumerate(list_a):
            doc_id = _doc_key(doc)
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1 / (k + rank + 1)

        for rank, doc in enumerate(list_b):
            doc_id = _doc_key(doc)
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1 / (k + rank + 1)

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return [doc_map[uid] for uid in sorted_ids]

    def search_hybrid(self, query: str, top_k: int = 3) -> str:
        vector_results = self._search_qdrant(query, top_k=top_k) 
        bm25_results = self.textbook_retriever.search(query, top_k=top_k) 
        
        fused_results = self._rrf_fusion(vector_results, bm25_results)

        def _safe_str(x) -> str:
            if x is None:
                return ""
            return x if isinstance(x, str) else str(x)

        valid_results = [
            d for d in fused_results
            if _safe_str(d.get("content", "")).strip()
        ]
        final_results = valid_results[:top_k]

        context_parts = []
        ref_id = 0
        for doc in final_results:
            source_type = _safe_str(doc.get("source_type", "Unknown Source"))
            title = _safe_str(doc.get("title", "Untitled"))
            content = _safe_str(doc.get("content", ""))
            ref_id += 1

            part = (
                f"--- [Ref {ref_id}] Source: {source_type} ---\n"
                f"TITLE: {title}\n"
                f"CONTENT:\n{content}\n"
            )
            context_parts.append(part)

        if not context_parts:
            return "NO_EVIDENCE_RETRIEVED: BM25 and vector search returned no usable content."

        return "\n".join(context_parts)
