import re
from pathlib import Path
from typing import Optional

from codegraph.graph.schema import Edge, EdgeType, Node, NodeType
from codegraph.graph.store import GraphStore
from codegraph.utils.hashing import hash_content, node_id
from codegraph.utils.logging import get_logger

log = get_logger(__name__)

_DOC_DIRS = {"docs", "documentation", "wiki", "doc"}
_DOC_EXTS = {".md", ".rst", ".txt"}


class DocIndexer:
    def __init__(self, store: GraphStore, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedder = None
        self._faiss_index = None
        self._chunk_ids: list[str] = []

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except ImportError:
                log.warning("sentence_transformers_not_installed")
        return self._embedder

    def index_repo(self, root_path: Path) -> int:
        docs = self._find_docs(root_path)
        chunks_indexed = 0
        for doc_path in docs:
            try:
                content = doc_path.read_text(errors="replace")
                chunks = self._chunk(content)
                source = str(doc_path.relative_to(root_path))
                for i, chunk in enumerate(chunks):
                    chunk_qname = f"doc::{source}::{i}"
                    chunk_id = node_id(str(doc_path), chunk_qname)
                    chunk_node = Node(
                        id=chunk_id,
                        node_type=NodeType.DOC_CHUNK,
                        name=f"{doc_path.name}#{i}",
                        qualified_name=chunk_qname,
                        file_path=str(doc_path),
                        start_line=0,
                        end_line=0,
                        language="text",
                        docstring=chunk[:200],
                        source_hash=hash_content(chunk),
                        metadata={"source": source, "chunk_index": i, "content": chunk},
                    )
                    self.store.upsert_node(chunk_node)
                    chunks_indexed += 1
                    # Link to code nodes mentioned in chunk
                    self._link_to_code(chunk_id, chunk)
            except Exception as e:
                log.error("doc_index_error", path=str(doc_path), error=str(e))

        self.store.commit()

        # Build FAISS index
        self._build_faiss()
        log.info("docs_indexed", chunks=chunks_indexed)
        return chunks_indexed

    def _find_docs(self, root: Path) -> list[Path]:
        docs: list[Path] = []
        # Root README
        for name in ("README.md", "README.rst", "README.txt"):
            p = root / name
            if p.exists():
                docs.append(p)
        # Doc directories
        for d in root.iterdir():
            if d.is_dir() and d.name.lower() in _DOC_DIRS:
                for f in d.rglob("*"):
                    if f.is_file() and f.suffix in _DOC_EXTS:
                        docs.append(f)
        # All .md files in repo
        for f in root.rglob("*.md"):
            if f not in docs:
                docs.append(f)
        return docs

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            i += self.chunk_size - self.chunk_overlap
        return chunks or [text]

    def _link_to_code(self, chunk_id: str, text: str) -> None:
        # Find backtick-enclosed names like `authenticate()` or `MyClass`
        pattern = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)\(?[^`]*`")
        for match in pattern.finditer(text):
            name = match.group(1).split(".")[0]
            node = self.store.get_node_by_name(name)
            if node:
                self.store.upsert_edges([Edge(chunk_id, node.id, EdgeType.DOCUMENTS)])

    def _build_faiss(self) -> None:
        embedder = self._get_embedder()
        if embedder is None:
            return
        try:
            import faiss
            import numpy as np
            chunks = self.store.get_all_nodes(NodeType.DOC_CHUNK)
            if not chunks:
                return
            texts = [n.metadata.get("content", n.docstring or n.name) for n in chunks]
            self._chunk_ids = [n.id for n in chunks]
            embeddings = embedder.encode(texts, show_progress_bar=False)
            embeddings = np.array(embeddings, dtype="float32")
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            self._faiss_index = index
            log.info("faiss_index_built", vectors=len(texts), dim=dim)
        except ImportError:
            log.warning("faiss_not_installed")
        except Exception as e:
            log.error("faiss_build_error", error=str(e))

    def get_faiss_index(self):
        return self._faiss_index

    def get_chunk_ids(self) -> list[str]:
        return self._chunk_ids
