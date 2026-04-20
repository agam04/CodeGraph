from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

from codegraph.graph.schema import Edge, Node


class AnalysisResult(NamedTuple):
    nodes: list[Node]
    edges: list[Edge]


class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, file_path: Path, content: str) -> AnalysisResult:
        """Parse a source file and return extracted nodes and edges."""
