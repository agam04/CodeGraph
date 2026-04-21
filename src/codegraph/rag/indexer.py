"""RAG indexer: chunks + embeds documentation and code symbols.

Uses a dual-model strategy:
- Code symbols (functions, classes): CodeBERT embeddings — better semantic
  search for "what does this function do?" queries against source code.
- Documentation (README, /docs, docstrings): all-MiniLM-L6-v2 — optimised
  for natural language similarity.

Both indices are stored in memory and used together in the retriever via
Reciprocal Rank Fusion.
"""

import re
from pathlib import Path


from codegraph.graph.schema import Edge, EdgeType, Node, NodeType
from codegraph.graph.store import GraphStore
from codegraph.rag.embedders import CodeAwareEmbedder
from codegraph.utils.hashing import hash_content, node_id
from codegraph.utils.logging import get_logger

log = get_logger(__name__)

_DOC_DIRS = {"docs", "documentation", "wiki", "doc"}
_DOC_EXTS = {".md", ".rst", ".txt"}


class DocIndexer:
    def __init__(
        self,
        store: GraphStore,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        code_model: str = "microsoft/codebert-base",
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedder = CodeAwareEmbedder(
            code_model_name=code_model,
            text_model_name=text_model,
        )

        # Separate FAISS indices for docs (text model) and code (code model)
        self._doc_faiss = None
        self._doc_chunk_ids: list[str] = []

        self._code_faiss = None
        self._code_node_ids: list[str] = []

    # ── Public ─────────────────────────────────────────────────────────────────

    def index_repo(self, root_path: Path) -> int:
        """Index all documentation in the repo and embed code symbols."""
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
                    self._link_to_code(chunk_id, chunk)
            except Exception as e:
                log.error("doc_index_error", path=str(doc_path), error=str(e))

        self.store.commit()
        self._build_doc_faiss()
        self._build_code_faiss()

        log.info("docs_indexed", chunks=chunks_indexed)
        return chunks_indexed

    def get_doc_faiss(self):
        return self._doc_faiss

    def get_doc_chunk_ids(self) -> list[str]:
        return self._doc_chunk_ids

    def get_code_faiss(self):
        return self._code_faiss

    def get_code_node_ids(self) -> list[str]:
        return self._code_node_ids

    def get_embedder(self) -> CodeAwareEmbedder:
        return self._embedder

    # kept for backward compat with tests/retriever that call _get_embedder()
    def _get_embedder(self) -> CodeAwareEmbedder:
        return self._embedder

    # kept for retriever backward compat
    def get_faiss_index(self):
        return self._doc_faiss

    def get_chunk_ids(self) -> list[str]:
        return self._doc_chunk_ids

    # ── Private ────────────────────────────────────────────────────────────────

    def _find_docs(self, root: Path) -> list[Path]:
        docs: list[Path] = []
        for name in ("README.md", "README.rst", "README.txt"):
            p = root / name
            if p.exists():
                docs.append(p)
        for d in root.iterdir():
            if d.is_dir() and d.name.lower() in _DOC_DIRS:
                for f in d.rglob("*"):
                    if f.is_file() and f.suffix in _DOC_EXTS:
                        docs.append(f)
        for f in root.rglob("*.md"):
            if f not in docs:
                docs.append(f)
        return docs

    def _chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunks.append(" ".join(words[i : i + self.chunk_size]))
            i += self.chunk_size - self.chunk_overlap
        return chunks or [text]

    def _link_to_code(self, chunk_id: str, text: str) -> None:
        pattern = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)\(?[^`]*`")
        for match in pattern.finditer(text):
            name = match.group(1).split(".")[0]
            node = self.store.get_node_by_name(name)
            if node:
                self.store.upsert_edges([Edge(chunk_id, node.id, EdgeType.DOCUMENTS)])

    def _build_doc_faiss(self) -> None:
        """Build FAISS index for doc chunks using the text embedding model."""
        try:
            import faiss
            chunks = self.store.get_all_nodes(NodeType.DOC_CHUNK)
            if not chunks:
                return
            texts = [n.metadata.get("content", n.docstring or n.name) for n in chunks]
            self._doc_chunk_ids = [n.id for n in chunks]
            embeddings = self._embedder.embed_text(texts)
            embeddings = embeddings.astype("float32")
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            self._doc_faiss = index
            log.info("doc_faiss_built", vectors=len(texts), model="text")
        except ImportError:
            log.warning("faiss_not_installed")
        except Exception as e:
            log.error("doc_faiss_error", error=str(e))

    def _build_code_faiss(self) -> None:
        """Build FAISS index for code symbols using the code embedding model.

        This is the key differentiator: searching for 'password hashing' will
        surface functions that hash passwords even if none mention it in their
        docstring, because CodeBERT was trained on code/docstring alignment.
        """
        try:
            import faiss
            from codegraph.graph.schema import NodeType as NT
            code_nodes = (
                self.store.get_all_nodes(NT.FUNCTION)
                + self.store.get_all_nodes(NT.METHOD)
                + self.store.get_all_nodes(NT.CLASS)
            )
            if not code_nodes:
                return

            # Embed the signature + docstring as the code representation
            texts = []
            for n in code_nodes:
                sig = n.signature or ""
                doc = n.docstring or ""
                texts.append(f"{n.qualified_name} {sig} {doc}".strip())

            self._code_node_ids = [n.id for n in code_nodes]
            embeddings = self._embedder.embed_code(texts)
            embeddings = embeddings.astype("float32")
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            self._code_faiss = index
            log.info("code_faiss_built", vectors=len(texts), model="code")
        except ImportError:
            log.warning("faiss_not_installed")
        except Exception as e:
            log.error("code_faiss_error", error=str(e))
