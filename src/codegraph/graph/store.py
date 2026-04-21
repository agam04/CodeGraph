import json
import sqlite3
from pathlib import Path
from typing import Optional

from codegraph.graph.schema import Edge, EdgeType, Node, NodeType
from codegraph.utils.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT,
    file_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    language TEXT,
    docstring TEXT,
    signature TEXT,
    source_hash TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    metadata TEXT,
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    language TEXT
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_qname ON nodes(qualified_name);
CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id, edge_type);
"""


class GraphStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── File tracking ──────────────────────────────────────────────────────────

    def get_file_hash(self, path: str) -> Optional[str]:
        row = self._conn.execute("SELECT content_hash FROM files WHERE path = ?", (path,)).fetchone()
        return row["content_hash"] if row else None

    def upsert_file(self, path: str, content_hash: str, language: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO files (path, content_hash, language) VALUES (?, ?, ?)",
            (path, content_hash, language),
        )

    def delete_file_nodes(self, path: str) -> None:
        """Remove all nodes (and their edges) for a given file."""
        node_ids = [
            r["id"]
            for r in self._conn.execute("SELECT id FROM nodes WHERE file_path = ?", (path,)).fetchall()
        ]
        if node_ids:
            placeholders = ",".join("?" * len(node_ids))
            self._conn.execute(f"DELETE FROM edges WHERE source_id IN ({placeholders})", node_ids)
            self._conn.execute(f"DELETE FROM edges WHERE target_id IN ({placeholders})", node_ids)
            self._conn.execute("DELETE FROM nodes WHERE file_path = ?", (path,))

    # ── Nodes ──────────────────────────────────────────────────────────────────

    def upsert_node(self, node: Node) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO nodes
               (id, node_type, name, qualified_name, file_path, start_line, end_line,
                language, docstring, signature, source_hash, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                node.id, node.node_type.value, node.name, node.qualified_name,
                node.file_path, node.start_line, node.end_line, node.language,
                node.docstring, node.signature, node.source_hash,
                json.dumps(node.metadata),
            ),
        )

    def upsert_nodes(self, nodes: list[Node]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO nodes
               (id, node_type, name, qualified_name, file_path, start_line, end_line,
                language, docstring, signature, source_hash, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    n.id, n.node_type.value, n.name, n.qualified_name,
                    n.file_path, n.start_line, n.end_line, n.language,
                    n.docstring, n.signature, n.source_hash,
                    json.dumps(n.metadata),
                )
                for n in nodes
            ],
        )

    def get_node(self, node_id: str) -> Optional[Node]:
        row = self._conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return self._row_to_node(row) if row else None

    def get_node_by_name(self, name: str, node_type: Optional[NodeType] = None) -> Optional[Node]:
        if node_type:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE name = ? AND node_type = ? LIMIT 1",
                (name, node_type.value),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM nodes WHERE name = ? LIMIT 1", (name,)
            ).fetchone()
        return self._row_to_node(row) if row else None

    def get_node_by_qualified_name(self, qname: str) -> Optional[Node]:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE qualified_name = ? LIMIT 1", (qname,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def search_nodes(self, pattern: str, node_type: Optional[NodeType] = None, limit: int = 10) -> list[Node]:
        like = f"%{pattern}%"
        if node_type:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE name LIKE ? AND node_type = ? LIMIT ?",
                (like, node_type.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE name LIKE ? LIMIT ?", (like, limit)
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_nodes_for_file(self, file_path: str) -> list[Node]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE file_path = ?", (file_path,)
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_all_nodes(self, node_type: Optional[NodeType] = None) -> list[Node]:
        if node_type:
            rows = self._conn.execute(
                "SELECT * FROM nodes WHERE node_type = ?", (node_type.value,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        return [self._row_to_node(r) for r in rows]

    # ── Edges ──────────────────────────────────────────────────────────────────

    def upsert_edges(self, edges: list[Edge]) -> None:
        self._conn.executemany(
            "INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, metadata) VALUES (?,?,?,?)",
            [(e.source_id, e.target_id, e.edge_type.value, json.dumps(e.metadata)) for e in edges],
        )

    def get_edges_from(self, source_id: str, edge_type: Optional[EdgeType] = None) -> list[Edge]:
        if edge_type:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ? AND edge_type = ?",
                (source_id, edge_type.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE source_id = ?", (source_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_edges_to(self, target_id: str, edge_type: Optional[EdgeType] = None) -> list[Edge]:
        if edge_type:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE target_id = ? AND edge_type = ?",
                (target_id, edge_type.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM edges WHERE target_id = ?", (target_id,)
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_all_files(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM files").fetchall()
        return [dict(r) for r in rows]

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total_nodes = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        total_edges = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        total_files = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        lang_rows = self._conn.execute(
            "SELECT language, COUNT(*) as cnt FROM nodes WHERE language IS NOT NULL AND node_type='module' GROUP BY language"
        ).fetchall()
        last_indexed = self._conn.execute(
            "SELECT MAX(last_indexed) FROM files"
        ).fetchone()[0]

        type_counts = {}
        for nt in NodeType:
            cnt = self._conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE node_type = ?", (nt.value,)
            ).fetchone()[0]
            if cnt:
                type_counts[nt.value] = cnt

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_files": total_files,
            "type_counts": type_counts,
            "languages": {r["language"]: r["cnt"] for r in lang_rows},
            "last_indexed": last_indexed,
        }

    def commit(self) -> None:
        self._conn.commit()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> Node:
        return Node(
            id=row["id"],
            node_type=NodeType(row["node_type"]),
            name=row["name"],
            qualified_name=row["qualified_name"] or row["name"],
            file_path=row["file_path"],
            start_line=row["start_line"] or 0,
            end_line=row["end_line"] or 0,
            language=row["language"] or "",
            docstring=row["docstring"],
            signature=row["signature"],
            source_hash=row["source_hash"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> Edge:
        return Edge(
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=EdgeType(row["edge_type"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
