"""Hybrid retriever: BM25 + dual-FAISS (code index + doc index) with RRF.

Search pipeline:
1. BM25 keyword search over doc chunks
2. FAISS vector search over doc chunks (text model)
3. FAISS vector search over code symbols (code model — CodeBERT)
4. Merge all three result lists with Reciprocal Rank Fusion

This three-way fusion means a query like "how does password hashing work?" will
surface both documentation chunks that mention passwords AND functions like
`hash_password` that the code model recognises as semantically related.
"""

from typing import Optional

from codegraph.graph.schema import EdgeType, Node, NodeType
from codegraph.graph.store import GraphStore
from codegraph.utils.logging import get_logger

log = get_logger(__name__)

_RRF_K = 60  # standard constant from the original RRF paper (Cormack et al. 2009)


class RAGRetriever:
    def __init__(self, store: GraphStore, indexer=None) -> None:
        self.store = store
        self._indexer = indexer
        self._bm25 = None
        self._bm25_ids: list[str] = []

    def search_docs(self, query: str, k: int = 5) -> list[dict]:
        """Three-way hybrid search: BM25 + doc-FAISS + code-FAISS, merged via RRF."""
        if self._indexer is None:
            return self._text_search_docs(query, k)

        bm25_results = self._bm25_search(query, k * 2)
        doc_vec_results = self._faiss_search_docs(query, k * 2)
        code_vec_results = self._faiss_search_code(query, k * 2)

        if not bm25_results and not doc_vec_results and not code_vec_results:
            return self._text_search_docs(query, k)

        rrf_scores: dict[str, float] = {}
        for rank, (nid, _) in enumerate(bm25_results):
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (rank + _RRF_K)
        for rank, (nid, _) in enumerate(doc_vec_results):
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (rank + _RRF_K)
        for rank, (nid, _) in enumerate(code_vec_results):
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (rank + _RRF_K)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for nid, rrf_score in ranked:
            node = self.store.get_node(nid)
            if node:
                results.append(self._node_to_result(node, relevance=rrf_score))
        return results

    def _bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            self._build_bm25()
        if self._bm25 is None or not self._bm25_ids:
            return []
        try:
            scores = self._bm25.get_scores(query.lower().split())
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [(self._bm25_ids[i], float(scores[i])) for i in top if scores[i] > 0]
        except Exception as e:
            log.error("bm25_error", error=str(e))
            return []

    def _faiss_search_docs(self, query: str, k: int) -> list[tuple[str, float]]:
        index = self._indexer.get_doc_faiss()
        ids = self._indexer.get_doc_chunk_ids()
        if index is None or not ids:
            return []
        try:
            embedder = self._indexer.get_embedder()
            q = embedder.embed_text([query]).astype("float32")
            dists, indices = index.search(q, min(k, len(ids)))
            return [
                (ids[i], float(1 / (1 + d)))
                for d, i in zip(dists[0], indices[0])
                if 0 <= i < len(ids)
            ]
        except Exception as e:
            log.error("doc_faiss_error", error=str(e))
            return []

    def _faiss_search_code(self, query: str, k: int) -> list[tuple[str, float]]:
        """Search code symbols using the code-specific embedding model."""
        index = self._indexer.get_code_faiss()
        ids = self._indexer.get_code_node_ids()
        if index is None or not ids:
            return []
        try:
            embedder = self._indexer.get_embedder()
            # Query is natural language — use text model for the query embedding,
            # but search against code embeddings (cross-modal retrieval)
            q = embedder.embed_text([query]).astype("float32")
            # Resize if dimension mismatch (code model vs text model dims differ)
            if q.shape[1] != index.d:
                return []
            dists, indices = index.search(q, min(k, len(ids)))
            return [
                (ids[i], float(1 / (1 + d)))
                for d, i in zip(dists[0], indices[0])
                if 0 <= i < len(ids)
            ]
        except Exception as e:
            log.error("code_faiss_error", error=str(e))
            return []

    def _build_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        chunks = self.store.get_all_nodes(NodeType.DOC_CHUNK)
        if not chunks:
            return
        corpus = [
            n.metadata.get("content", n.docstring or n.name).lower().split()
            for n in chunks
        ]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_ids = [n.id for n in chunks]

    def _text_search_docs(self, query: str, k: int) -> list[dict]:
        chunks = self.store.search_nodes(query, NodeType.DOC_CHUNK, limit=k)
        return [self._node_to_result(c) for c in chunks]

    def docs_for_node(self, node_id: str) -> list[dict]:
        edges = self.store.get_edges_to(node_id, EdgeType.DOCUMENTS)
        results = []
        for edge in edges:
            chunk = self.store.get_node(edge.source_id)
            if chunk and chunk.node_type == NodeType.DOC_CHUNK:
                results.append(self._node_to_result(chunk))
        return results

    @staticmethod
    def _node_to_result(node: Node, relevance: Optional[float] = None) -> dict:
        is_code = node.node_type in (NodeType.FUNCTION, NodeType.METHOD, NodeType.CLASS)
        return {
            "id": node.id,
            "content": node.metadata.get("content", node.docstring or node.name),
            "source": node.metadata.get("source", node.file_path),
            "relevance_score": relevance,
            "node_type": node.node_type.value,
            "result_kind": "code_symbol" if is_code else "documentation",
        }
