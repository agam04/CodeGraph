from typing import Optional

import strawberry

from codegraph.graph import NodeType
from codegraph.graph import queries as gq
from codegraph.graph.store import GraphStore
from codegraph.rag.retriever import RAGRetriever
from codegraph.utils.logging import get_logger

log = get_logger(__name__)


@strawberry.type(description="A function or method in the codebase.")
class Function:
    id: strawberry.ID = strawberry.field(description="Stable unique identifier.")
    name: str = strawberry.field(description="Short name, e.g. 'authenticate'.")
    qualified_name: str = strawberry.field(description="Fully qualified name, e.g. 'myapp.auth.authenticate'.")
    file_path: str = strawberry.field(description="Absolute path to the source file.")
    start_line: int = strawberry.field(description="Line where the function begins.")
    end_line: int = strawberry.field(description="Line where the function ends.")
    signature: Optional[str] = strawberry.field(default=None, description="Full signature, e.g. '(user: User, password: str) -> bool'.")
    docstring: Optional[str] = strawberry.field(default=None, description="Extracted docstring.")
    is_async: bool = strawberry.field(default=False, description="True if declared with async.")
    decorators: list[str] = strawberry.field(default_factory=list, description="Decorator names applied to the function.")
    language: str = strawberry.field(default="python", description="Source language.")


@strawberry.type(description="A class definition in the codebase.")
class Class:
    id: strawberry.ID = strawberry.field(description="Stable unique identifier.")
    name: str = strawberry.field(description="Short class name.")
    qualified_name: str = strawberry.field(description="Fully qualified class name.")
    file_path: str = strawberry.field(description="Absolute path to the source file.")
    start_line: int
    end_line: int
    docstring: Optional[str] = strawberry.field(default=None, description="Class docstring.")
    language: str = strawberry.field(default="python")


@strawberry.type(description="A source file.")
class File:
    path: str = strawberry.field(description="Absolute path.")
    language: str = strawberry.field(description="Detected language.")
    lines_of_code: int = strawberry.field(default=0)


@strawberry.type(description="A chunk of documentation (README, /docs, docstrings).")
class DocChunk:
    id: strawberry.ID
    content: str = strawberry.field(description="Raw chunk text.")
    source: str = strawberry.field(description="Source file path or symbol name.")
    relevance_score: Optional[float] = strawberry.field(default=None, description="Similarity score (0-1, higher is better).")


@strawberry.type(description="Statistics about the indexed codebase.")
class CodebaseStats:
    total_files: int
    total_functions: int
    total_classes: int
    total_nodes: int
    total_edges: int
    languages: list[str] = strawberry.field(description="Languages detected in the codebase.")
    last_indexed: Optional[str] = strawberry.field(default=None)


@strawberry.type(description="An edge in the code graph.")
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: str


@strawberry.type(description="A generic graph node (for subgraph responses).")
class GraphNode:
    id: str
    node_type: str
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    signature: Optional[str] = None
    docstring: Optional[str] = None


@strawberry.type(description="Token-budgeted subgraph around a central symbol — the killer feature for agents.")
class Subgraph:
    center_node: Optional[GraphNode] = strawberry.field(description="The requested symbol.")
    related_nodes: list[GraphNode] = strawberry.field(description="All nodes within `depth` hops, ranked by fan-in score.")
    edges: list[GraphEdge] = strawberry.field(description="Edges connecting the subgraph.")
    summary: str = strawberry.field(description="Auto-generated text summary an agent can read directly.")
    estimated_tokens: int = strawberry.field(description="Approximate token cost of this subgraph.")
    mermaid_diagram: str = strawberry.field(description="Mermaid flowchart of this subgraph — paste into markdown to visualize.")


@strawberry.type(description="Impact analysis: what breaks if you change this symbol?")
class ImpactAnalysis:
    target: GraphNode = strawberry.field(description="The symbol being analyzed.")
    immediate_callers: list[Function] = strawberry.field(description="Functions that directly call this symbol.")
    transitive_callers: list[Function] = strawberry.field(description="All functions reachable through the call chain.")
    affected_files: list[str] = strawberry.field(description="Unique files containing affected symbols.")
    risk_level: str = strawberry.field(description="'low' | 'medium' | 'high' — based on caller count and reach.")
    summary: str = strawberry.field(description="Human-readable impact summary safe to surface to an agent.")


def _make_function(node) -> Optional[Function]:
    if node is None:
        return None
    meta = node.metadata or {}
    return Function(
        id=node.id,
        name=node.name,
        qualified_name=node.qualified_name,
        file_path=node.file_path,
        start_line=node.start_line,
        end_line=node.end_line,
        signature=node.signature,
        docstring=node.docstring,
        is_async=meta.get("is_async", False),
        decorators=meta.get("decorators", []),
        language=node.language or "python",
    )


def _make_class(node) -> Optional[Class]:
    if node is None:
        return None
    return Class(
        id=node.id,
        name=node.name,
        qualified_name=node.qualified_name,
        file_path=node.file_path,
        start_line=node.start_line,
        end_line=node.end_line,
        docstring=node.docstring,
        language=node.language or "python",
    )


def _make_graph_node(node) -> GraphNode:
    return GraphNode(
        id=node.id,
        node_type=node.node_type.value,
        name=node.name,
        qualified_name=node.qualified_name,
        file_path=node.file_path,
        start_line=node.start_line,
        end_line=node.end_line,
        signature=node.signature,
        docstring=node.docstring,
    )


def _make_doc_chunk(chunk: dict) -> DocChunk:
    return DocChunk(
        id=chunk["id"],
        content=chunk["content"],
        source=chunk["source"],
        relevance_score=chunk.get("relevance_score"),
    )


def build_schema(store: GraphStore, retriever: RAGRetriever) -> strawberry.Schema:
    @strawberry.type
    class Query:
        @strawberry.field(description="Find a function by short name or fully qualified name.")
        def function(
            self,
            name: Optional[str] = None,
            qualified_name: Optional[str] = None,
        ) -> Optional[Function]:
            if qualified_name:
                node = store.get_node_by_qualified_name(qualified_name)
            elif name:
                node = store.get_node_by_name(name, NodeType.FUNCTION) or store.get_node_by_name(name, NodeType.METHOD)
            else:
                return None
            return _make_function(node)

        @strawberry.field(description="Find a class by short name or fully qualified name.")
        def class_(
            self,
            name: Optional[str] = None,
            qualified_name: Optional[str] = None,
        ) -> Optional[Class]:
            if qualified_name:
                node = store.get_node_by_qualified_name(qualified_name)
            elif name:
                node = store.get_node_by_name(name, NodeType.CLASS)
            else:
                return None
            return _make_class(node)

        @strawberry.field(description="Get all functions and classes defined in a file.")
        def file(self, path: str) -> Optional[File]:
            file_nodes = store.get_nodes_for_file(path)
            if not file_nodes:
                return None
            module = next((n for n in file_nodes if n.node_type == NodeType.MODULE), None)
            lang = module.language if module else "unknown"
            return File(path=path, language=lang, lines_of_code=module.end_line if module else 0)

        @strawberry.field(description="Search for functions whose name matches a substring pattern.")
        def search_functions(self, pattern: str, limit: int = 10) -> list[Function]:
            nodes = store.search_nodes(pattern, NodeType.FUNCTION, limit=limit)
            nodes += store.search_nodes(pattern, NodeType.METHOD, limit=limit)
            return [f for n in nodes if (f := _make_function(n)) is not None]

        @strawberry.field(description="Search for classes whose name matches a substring pattern.")
        def search_classes(self, pattern: str, limit: int = 10) -> list[Class]:
            nodes = store.search_nodes(pattern, NodeType.CLASS, limit=limit)
            return [c for n in nodes if (c := _make_class(n)) is not None]

        @strawberry.field(description="Who calls this function? Critical before refactoring.")
        def callers(self, qualified_name: str, depth: int = 1) -> list[Function]:
            node = store.get_node_by_qualified_name(qualified_name)
            if not node:
                return []
            callers = gq.get_callers(store, node.id, depth=depth)
            return [f for n in callers if (f := _make_function(n)) is not None]

        @strawberry.field(description="What does this function call? Understand its dependencies.")
        def callees(self, qualified_name: str, depth: int = 1) -> list[Function]:
            node = store.get_node_by_qualified_name(qualified_name)
            if not node:
                return []
            callees = gq.get_callees(store, node.id, depth=depth)
            return [f for n in callees if (f := _make_function(n)) is not None]

        @strawberry.field(description="What files does this file import, up to `depth` hops?")
        def dependencies(self, file_path: str, depth: int = 1) -> list[File]:
            deps = gq.get_dependencies(store, file_path, depth=depth)
            return [File(path=n.file_path, language=n.language or "unknown", lines_of_code=n.end_line) for n in deps]

        @strawberry.field(description="What files import this file, up to `depth` hops?")
        def dependents(self, file_path: str, depth: int = 1) -> list[File]:
            deps = gq.get_dependents(store, file_path, depth=depth)
            return [File(path=n.file_path, language=n.language or "unknown", lines_of_code=n.end_line) for n in deps]

        @strawberry.field(
            description="Get everything needed to understand or modify a symbol: callers, callees, "
                        "related docs, nearby code. Token-budgeted — safe to pass directly to an agent."
        )
        def context_for(self, qualified_name: str, depth: int = 2) -> Subgraph:
            node = store.get_node_by_qualified_name(qualified_name)
            if node is None:
                node = store.get_node_by_name(qualified_name)
            if node is None:
                return Subgraph(
                    center_node=None, related_nodes=[], edges=[],
                    summary=f"Symbol '{qualified_name}' not found.",
                    estimated_tokens=0, mermaid_diagram="",
                )
            from codegraph.config import get_config
            subgraph = gq.get_subgraph(store, node.id, depth=depth, max_tokens=get_config().max_subgraph_tokens)
            return Subgraph(
                center_node=_make_graph_node(subgraph["center_node"]) if subgraph["center_node"] else None,
                related_nodes=[_make_graph_node(n) for n in subgraph["related_nodes"]],
                edges=[GraphEdge(**e) for e in subgraph["edges"]],
                summary=subgraph["summary"],
                estimated_tokens=subgraph["estimated_tokens"],
                mermaid_diagram=gq.subgraph_to_mermaid(subgraph),
            )

        @strawberry.field(
            description="Impact analysis: which functions and files would break if this symbol changed? "
                        "Returns a risk level and full transitive call chain."
        )
        def impact_of(self, qualified_name: str, depth: int = 5) -> Optional[ImpactAnalysis]:
            node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
            if not node:
                return None
            result = gq.impact_analysis(store, node.id, depth=depth)
            if not result:
                return None
            return ImpactAnalysis(
                target=_make_graph_node(result["center"]),
                immediate_callers=[f for n in result["immediate_callers"] if (f := _make_function(n)) is not None],
                transitive_callers=[f for n in result["transitive_callers"] if (f := _make_function(n)) is not None],
                affected_files=result["affected_files"],
                risk_level=result["risk_level"],
                summary=result["summary"],
            )

        @strawberry.field(
            description="Find functions with no callers — potential dead code. "
                        "Excludes entry points, dunder methods, and test functions."
        )
        def dead_code(self) -> list[Function]:
            nodes = gq.find_dead_code(store)
            return [f for n in nodes if (f := _make_function(n)) is not None]

        @strawberry.field(description="Semantic search across documentation and docstrings. Uses hybrid BM25 + vector search with Reciprocal Rank Fusion.")
        def search_docs(self, query: str, limit: int = 5) -> list[DocChunk]:
            chunks = retriever.search_docs(query, k=limit)
            return [_make_doc_chunk(c) for c in chunks]

        @strawberry.field(description="Return all documentation linked to a symbol.")
        def docs_for_node(self, qualified_name: str) -> list[DocChunk]:
            node = store.get_node_by_qualified_name(qualified_name)
            if not node:
                return []
            chunks = retriever.docs_for_node(node.id)
            return [_make_doc_chunk(c) for c in chunks]

        @strawberry.field(
            description="Return the exact source code of a symbol — AST-verified ground truth. "
                        "Use this instead of recalling an implementation from memory to prevent hallucination."
        )
        def get_source(self, qualified_name: str) -> Optional[str]:
            node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
            if not node:
                return None
            try:
                lines = __import__("pathlib").Path(node.file_path).read_text(errors="replace").splitlines()
                start = max(0, node.start_line - 1)
                end = min(len(lines), node.end_line)
                return "\n".join(lines[start:end])
            except OSError:
                return None

        @strawberry.field(
            description="High-level overview of the indexed codebase.")
        def stats(self) -> CodebaseStats:
            s = store.stats()
            return CodebaseStats(
                total_files=s["total_files"],
                total_functions=s["type_counts"].get("function", 0) + s["type_counts"].get("method", 0),
                total_classes=s["type_counts"].get("class", 0),
                total_nodes=s["total_nodes"],
                total_edges=s["total_edges"],
                languages=list(s["languages"].keys()),
                last_indexed=s["last_indexed"],
            )

    return strawberry.Schema(query=Query)
