from typing import Optional

from codegraph.graph.schema import EdgeType, Node, NodeType
from codegraph.graph.store import GraphStore
from codegraph.utils.logging import get_logger

log = get_logger(__name__)


class RAGRetriever:
    def __init__(self, store: GraphStore, indexer=None) -> None:
        self.store = store
        self._indexer = indexer

    def search_docs(self, query: str, k: int = 5) -> list[dict]:
        if self._indexer is None:
            return self._text_search_docs(query, k)

        faiss_index = self._indexer.get_faiss_index()
        chunk_ids = self._indexer.get_chunk_ids()

        if faiss_index is None or not chunk_ids:
            return self._text_search_docs(query, k)

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            embedder = self._indexer._get_embedder()
            if embedder is None:
                return self._text_search_docs(query, k)

            q_emb = embedder.encode([query], show_progress_bar=False)
            q_emb = np.array(q_emb, dtype="float32")
            distances, indices = faiss_index.search(q_emb, min(k, len(chunk_ids)))

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(chunk_ids):
                    continue
                node = self.store.get_node(chunk_ids[idx])
                if node:
                    results.append(self._node_to_chunk(node, relevance=float(1 / (1 + dist))))
            return results
        except Exception as e:
            log.error("faiss_search_error", error=str(e))
            return self._text_search_docs(query, k)

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
