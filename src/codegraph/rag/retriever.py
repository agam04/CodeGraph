from typing import Optional

from codegraph.graph.schema import EdgeType, Node, NodeType
from codegraph.graph.store import GraphStore
from codegraph.utils.logging import get_logger

log = get_logger(__name__)

# Reciprocal Rank Fusion constant — 60 is the standard value from the original RRF paper
_RRF_K = 60


class RAGRetriever:
    def __init__(self, store: GraphStore, indexer=None) -> None:
        self.store = store
        self._indexer = indexer
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_corpus: list[list[str]] = []

    def _build_bm25(self) -> None:
        """Build a BM25 index over all DOC_CHUNK nodes for keyword search."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            log.warning("rank_bm25_not_installed")
            return

        chunks = self.store.get_all_nodes(NodeType.DOC_CHUNK)
        if not chunks:
            return

        corpus = []
        ids = []
        for chunk in chunks:
            text = chunk.metadata.get("content", chunk.docstring or chunk.name)
            corpus.append(text.lower().split())
            ids.append(chunk.id)

        self._bm25 = BM25Okapi(corpus)
        self._bm25_ids = ids
        self._bm25_corpus = corpus

    def search_docs(self, query: str, k: int = 5) -> list[dict]:
        """Hybrid search: BM25 + FAISS vector similarity, merged with Reciprocal Rank Fusion."""
        if self._indexer is None:
            return self._text_search_docs(query, k)

        faiss_index = self._indexer.get_faiss_index()
        chunk_ids = self._indexer.get_chunk_ids()
        bm25_results = self._bm25_search(query, k * 2)
        vector_results = self._faiss_search(query, k * 2, faiss_index, chunk_ids)

        if not bm25_results and not vector_results:
            return self._text_search_docs(query, k)

        # Reciprocal Rank Fusion: score = Σ 1/(rank_i + k) across all result lists
        rrf_scores: dict[str, float] = {}
        for rank, (chunk_id, _) in enumerate(bm25_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rank + _RRF_K)
        for rank, (chunk_id, _) in enumerate(vector_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rank + _RRF_K)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for chunk_id, rrf_score in ranked:
            node = self.store.get_node(chunk_id)
            if node:
                results.append(self._node_to_chunk(node, relevance=rrf_score))
        return results

    def _bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            self._build_bm25()
        if self._bm25 is None or not self._bm25_ids:
            return []
        try:
            tokenized = query.lower().split()
            scores = self._bm25.get_scores(tokenized)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [(self._bm25_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
        except Exception as e:
            log.error("bm25_search_error", error=str(e))
            return []

    def _faiss_search(self, query: str, k: int, faiss_index, chunk_ids: list[str]) -> list[tuple[str, float]]:
        if faiss_index is None or not chunk_ids:
            return []
        try:
            import numpy as np
            embedder = self._indexer._get_embedder()
            if embedder is None:
                return []
            q_emb = embedder.encode([query], show_progress_bar=False)
            q_emb = np.array(q_emb, dtype="float32")
            distances, indices = faiss_index.search(q_emb, min(k, len(chunk_ids)))
            return [
                (chunk_ids[idx], float(1 / (1 + dist)))
                for dist, idx in zip(distances[0], indices[0])
                if 0 <= idx < len(chunk_ids)
            ]
        except Exception as e:
            log.error("faiss_search_error", error=str(e))
            return []

    def _text_search_docs(self, query: str, k: int) -> list[dict]:
        chunks = self.store.search_nodes(query, NodeType.DOC_CHUNK, limit=k)
        return [self._node_to_chunk(c) for c in chunks]

    def docs_for_node(self, node_id: str) -> list[dict]:
        edges = self.store.get_edges_to(node_id, EdgeType.DOCUMENTS)
        results = []
        for edge in edges:
            chunk = self.store.get_node(edge.source_id)
            if chunk and chunk.node_type == NodeType.DOC_CHUNK:
                results.append(self._node_to_chunk(chunk))
        return results

    @staticmethod
    def _node_to_chunk(node: Node, relevance: Optional[float] = None) -> dict:
        return {
            "id": node.id,
            "content": node.metadata.get("content", node.docstring or node.name),
            "source": node.metadata.get("source", node.file_path),
            "relevance_score": relevance,
        }
