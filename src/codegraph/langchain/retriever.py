"""LangChain retriever backed by codegraph.

Implements LangChain's BaseRetriever interface so codegraph can be dropped
into any LangChain chain, agent, or RAG pipeline as a knowledge source.

The retriever runs the three-way hybrid search (BM25 + doc-FAISS + code-FAISS)
and returns LangChain Document objects with rich metadata so downstream LLMs
can cite sources and line numbers.
"""

from typing import Any

from codegraph.graph.schema import NodeType
from codegraph.graph.store import GraphStore
from codegraph.rag.retriever import RAGRetriever
from codegraph.utils.logging import get_logger

log = get_logger(__name__)


def _build_documents(results: list[dict], code_nodes: list[dict]) -> list:
    """Convert codegraph results to LangChain Document objects."""
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document  # type: ignore[no-reattr]

    docs = []
    for r in results:
        docs.append(Document(
            page_content=r.get("content", ""),
            metadata={
                "source": r.get("source", ""),
                "relevance_score": r.get("relevance_score"),
                "result_kind": r.get("result_kind", "documentation"),
                "node_type": r.get("node_type", ""),
            },
        ))
    for n in code_nodes:
        content = (
            f"Function: {n.get('qualified_name', n.get('name', ''))}\n"
            f"Signature: {n.get('signature', '')}\n"
            f"Docstring: {n.get('docstring', '') or n.get('generated_docstring', 'No docstring.')}\n"
            f"File: {n.get('file', '')} (line {n.get('start_line', '')})"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source": n.get("file", ""),
                "qualified_name": n.get("qualified_name", ""),
                "start_line": n.get("start_line", 0),
                "result_kind": "code_symbol",
                "node_type": n.get("type", "function"),
            },
        ))
    return docs


class CodeGraphRetriever:
    """LangChain-compatible retriever backed by codegraph's hybrid search.

    Supports both the new `langchain-core` interface and the older
    `langchain` package via duck-typing.

    Usage:
        retriever = CodeGraphRetriever(store=store, rag=retriever, k=5)
        docs = retriever.invoke("how does authentication work?")
    """

    def __init__(
        self,
        store: GraphStore,
        rag: RAGRetriever,
        k: int = 5,
        include_code_symbols: bool = True,
        code_symbols_limit: int = 3,
    ) -> None:
        self.store = store
        self.rag = rag
        self.k = k
        self.include_code_symbols = include_code_symbols
        self.code_symbols_limit = code_symbols_limit

    def _get_relevant_documents(self, query: str) -> list:
        # Hybrid search over docs
        doc_results = self.rag.search_docs(query, k=self.k)

        # Also surface directly-matched code symbols
        code_nodes = []
        if self.include_code_symbols:
            matches = self.store.search_nodes(query, node_type=None, limit=self.code_symbols_limit)
            code_nodes = [
                {
                    "name": n.name,
                    "qualified_name": n.qualified_name,
                    "type": n.node_type.value,
                    "signature": n.signature,
                    "docstring": n.docstring,
                    "generated_docstring": n.metadata.get("generated_docstring"),
                    "file": n.file_path,
                    "start_line": n.start_line,
                }
                for n in matches
                if n.node_type in (NodeType.FUNCTION, NodeType.METHOD, NodeType.CLASS)
            ]

        return _build_documents(doc_results, code_nodes)

    # LangChain new-style invoke
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> list:
        query = input if isinstance(input, str) else str(input)
        return self._get_relevant_documents(query)

    # LangChain old-style get_relevant_documents
    def get_relevant_documents(self, query: str) -> list:
        return self._get_relevant_documents(query)

    # Allow use as a plain callable
    def __call__(self, query: str) -> list:
        return self._get_relevant_documents(query)

    # Pydantic-style attribute so LangChain chains can inspect it
    @property
    def search_kwargs(self) -> dict:
        return {"k": self.k}
