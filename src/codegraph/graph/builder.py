import time
from pathlib import Path
from typing import Optional

import pathspec

from codegraph.analyzers.js_analyzer import JSAnalyzer
from codegraph.analyzers.python_analyzer import PythonAnalyzer
from codegraph.config import CodegraphConfig
from codegraph.graph.schema import BuildStats
from codegraph.graph.store import GraphStore
from codegraph.utils.hashing import hash_file
from codegraph.utils.logging import get_logger

log = get_logger(__name__)

_ANALYZERS = {
    ".py": lambda: PythonAnalyzer(),
    ".js": lambda: JSAnalyzer("javascript"),
    ".jsx": lambda: JSAnalyzer("javascript"),
    ".ts": lambda: JSAnalyzer("typescript"),
    ".tsx": lambda: JSAnalyzer("tsx"),
}


class GraphBuilder:
    def __init__(self, root_path: Path, store: GraphStore, config: Optional[CodegraphConfig] = None) -> None:
        self.root_path = root_path.resolve()
        self.store = store
        self.config = config or CodegraphConfig()
        self._analyzers = {ext: factory() for ext, factory in _ANALYZERS.items()}
        self._gitignore_spec: Optional[pathspec.PathSpec] = None
        if self.config.respect_gitignore:
            self._gitignore_spec = self._load_gitignore()

    def _load_gitignore(self) -> Optional[pathspec.PathSpec]:
        gitignore = self.root_path / ".gitignore"
        if gitignore.exists():
            patterns = gitignore.read_text(errors="replace").splitlines()
            return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        return None

    def _is_excluded(self, path: Path) -> bool:
        rel = path.relative_to(self.root_path)
        rel_str = str(rel)
        for pattern in self.config.exclude_patterns:
            if any(part == pattern or rel_str.endswith(pattern) for part in rel.parts):
                return True
        if self._gitignore_spec and self._gitignore_spec.match_file(rel_str):
            return True
        return False

    def _iter_source_files(self):
        max_bytes = self.config.max_file_size_kb * 1024
        for path in self.root_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in _ANALYZERS:
                continue
            if self._is_excluded(path):
                continue
            if path.stat().st_size > max_bytes:
                log.debug("skipping_large_file", path=str(path))
                continue
            yield path

    def build(self, incremental: bool = True) -> BuildStats:
        stats = BuildStats()
        start = time.monotonic()

        for file_path in self._iter_source_files():
            try:
                content = file_path.read_text(errors="replace")
                file_hash = hash_file(file_path)
                fp_str = str(file_path)
                suffix = file_path.suffix

                if incremental:
                    stored_hash = self.store.get_file_hash(fp_str)
                    if stored_hash == file_hash:
                        stats.files_skipped += 1
                        continue

                # Remove stale data before re-indexing
                self.store.delete_file_nodes(fp_str)

                analyzer = self._analyzers.get(suffix)
                if analyzer is None:
                    continue

                result = analyzer.analyze(file_path, content)

                lang = _lang_for_suffix(suffix)
                self.store.upsert_file(fp_str, file_hash, lang)
                self.store.upsert_nodes(result.nodes)
                self.store.upsert_edges(result.edges)
                self.store.commit()

                stats.files_indexed += 1
                stats.nodes_created += len(result.nodes)
                stats.edges_created += len(result.edges)

                log.debug(
                    "file_indexed",
                    file=fp_str,
                    nodes=len(result.nodes),
                    edges=len(result.edges),
                )
            except Exception as e:
                msg = f"{file_path}: {e}"
                stats.errors.append(msg)
                log.error("index_error", file=str(file_path), error=str(e))

        # Cross-file resolution: link unresolved import/call edges to actual nodes
        resolved = self._resolve_cross_file_edges()
        stats.edges_created += resolved
        log.debug("cross_file_resolved", edges=resolved)

        stats.time_elapsed = time.monotonic() - start
        log.info(
            "build_complete",
            files=stats.files_indexed,
            skipped=stats.files_skipped,
            nodes=stats.nodes_created,
            edges=stats.edges_created,
            elapsed=f"{stats.time_elapsed:.2f}s",
            errors=len(stats.errors),
        )
        return stats

    def _resolve_cross_file_edges(self) -> int:
        """Post-index pass: resolve unresolved import/call edges to real target nodes.

        When the analyzer stores IMPORTS edges, it uses a synthetic target ID derived
        from the module name. After all files are indexed, we can look up the real
        module node and add a resolved edge pointing to it.
        """
        from codegraph.graph.schema import Edge, EdgeType
        from codegraph.graph.schema import NodeType

        resolved = 0
        new_edges: list[Edge] = []

        # Walk every module node and try to resolve its IMPORTS edges
        for module in self.store.get_all_nodes(NodeType.MODULE):
            for edge in self.store.get_edges_from(module.id, EdgeType.IMPORTS):
                if not edge.metadata.get("unresolved"):
                    continue
                target_module = edge.metadata.get("target_module", "")
                if not target_module:
                    continue

                # Try to find the target module by qualified name or short name
                target_node = self.store.get_node_by_qualified_name(target_module)
                if target_node is None:
                    # Try just the last segment (e.g. "models" from "sample_python_repo.models")
                    short = target_module.split(".")[-1]
                    target_node = self.store.get_node_by_name(short, NodeType.MODULE)

                if target_node and target_node.id != edge.target_id:
                    new_edges.append(Edge(
                        source_id=module.id,
                        target_id=target_node.id,
                        edge_type=EdgeType.IMPORTS,
                        metadata={"alias": edge.metadata.get("alias", ""), "resolved": True},
                    ))
                    resolved += 1

        # Also resolve CALLS edges — match unresolved callee names to actual function nodes
        for func in self.store.get_all_nodes(NodeType.FUNCTION) + self.store.get_all_nodes(NodeType.METHOD):
            for edge in self.store.get_edges_from(func.id, EdgeType.CALLS):
                callee_name = edge.metadata.get("unresolved_name", "")
                if not callee_name:
                    continue
                # Try qualified name first, then short name
                target = self.store.get_node_by_qualified_name(callee_name)
                if target is None:
                    short = callee_name.split(".")[-1]
                    target = (
                        self.store.get_node_by_name(short, NodeType.FUNCTION)
                        or self.store.get_node_by_name(short, NodeType.METHOD)
                    )
                if target and target.id != edge.target_id:
                    new_edges.append(Edge(
                        source_id=func.id,
                        target_id=target.id,
                        edge_type=EdgeType.CALLS,
                        metadata={"line": edge.metadata.get("line", 0), "resolved": True},
                    ))
                    resolved += 1

        if new_edges:
            self.store.upsert_edges(new_edges)
            self.store.commit()

        return resolved


def _lang_for_suffix(suffix: str) -> str:
    mapping = {".py": "python", ".js": "javascript", ".jsx": "javascript",
               ".ts": "typescript", ".tsx": "typescript"}
    return mapping.get(suffix, "unknown")
