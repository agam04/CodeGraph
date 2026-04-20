from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NodeType(str, Enum):
    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    DOC_CHUNK = "doc_chunk"


class EdgeType(str, Enum):
    IMPORTS = "imports"
    DEFINES = "defines"
    CALLS = "calls"
    INHERITS = "inherits"
    REFERENCES = "references"
    DOCUMENTS = "documents"


@dataclass
class Node:
    id: str
    node_type: NodeType
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    source_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildStats:
    files_indexed: int = 0
    files_skipped: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    time_elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)
