import hashlib
from pathlib import Path


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def node_id(file_path: str, qualified_name: str) -> str:
    raw = f"{file_path}::{qualified_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]
