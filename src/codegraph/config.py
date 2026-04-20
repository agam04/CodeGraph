from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class CodegraphConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODEGRAPH_")

    data_dir: Path = Path("./data")
    repo_path: Optional[Path] = None

    respect_gitignore: bool = True
    exclude_patterns: list[str] = ["node_modules", "__pycache__", ".venv", "dist", ".git", "*.pyc"]
    max_file_size_kb: int = 500

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50

    graphql_port: int = 8000
    mcp_transport: Literal["stdio", "http"] = "stdio"
    log_level: str = "INFO"

    max_subgraph_tokens: int = 4000


_config: Optional[CodegraphConfig] = None


def get_config() -> CodegraphConfig:
    global _config
    if _config is None:
        _config = CodegraphConfig()
    return _config


def set_config(config: CodegraphConfig) -> None:
    global _config
    _config = config
