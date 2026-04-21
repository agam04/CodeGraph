from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from codegraph.config import CodegraphConfig, get_config
from codegraph.graph import GraphBuilder, GraphStore
from codegraph.graph import queries as gq
from codegraph.graph.schema import NodeType
from codegraph.rag.retriever import RAGRetriever
from codegraph.utils.logging import get_logger

log = get_logger(__name__)

mcp = FastMCP(
    "codegraph",
    instructions=(
        "codegraph gives you AST-verified ground truth about this codebase. "
        "ALWAYS prefer these tools over reading files directly — they prevent hallucination "
        "and reduce token usage by 50-80%. "
        "Workflow: (1) codebase_stats for orientation, "
        "(2) find_function / find_class to locate symbols, "
        "(3) get_context for the full picture before making changes, "
        "(4) impact_analysis before any refactor, "
        "(5) verify_signature to confirm a signature before calling or citing it. "
        "get_source returns exact source lines — use it instead of guessing implementations."
    ),
)

_store: GraphStore | None = None
_retriever: RAGRetriever | None = None
_config: CodegraphConfig | None = None


def init_mcp(store: GraphStore, retriever: RAGRetriever, config: CodegraphConfig) -> None:
    global _store, _retriever, _config
    _store = store
    _retriever = retriever
    _config = config


def _get_store() -> GraphStore:
    if _store is None:
        raise RuntimeError("MCP server not initialized. Call init_mcp() first.")
    return _store


def _get_retriever() -> RAGRetriever:
    if _retriever is None:
        raise RuntimeError("MCP server not initialized. Call init_mcp() first.")
    return _retriever


def _node_to_dict(node) -> dict[str, Any]:
    if node is None:
        return {}
    return {
        "id": node.id,
        "name": node.name,
        "qualified_name": node.qualified_name,
        "type": node.node_type.value,
        "file": node.file_path,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "signature": node.signature,
        "docstring": node.docstring,
        "is_async": node.metadata.get("is_async", False),
        "decorators": node.metadata.get("decorators", []),
        "language": node.language,
    }


@mcp.tool()
def find_function(name: str) -> dict:
    """Find a function by name. Returns location, signature, docstring, and lists of callers/callees."""
    store = _get_store()
    node = store.get_node_by_name(name, NodeType.FUNCTION) or store.get_node_by_name(name, NodeType.METHOD)
    if not node:
        # Try qualified name lookup
        node = store.get_node_by_qualified_name(name)
    if not node:
        return {"error": f"Function '{name}' not found. Try search_code to find similar names."}

    callers = gq.get_callers(store, node.id, depth=1)
    callees = gq.get_callees(store, node.id, depth=1)

    result = _node_to_dict(node)
    result["callers"] = [{"name": n.name, "file": n.file_path, "line": n.start_line} for n in callers[:10]]
    result["callees"] = [{"name": n.name, "file": n.file_path, "line": n.start_line} for n in callees[:10]]
    return result


@mcp.tool()
def find_class(name: str) -> dict:
    """Find a class by name. Returns location, methods, base classes, and subclasses."""
    store = _get_store()
    node = store.get_node_by_name(name, NodeType.CLASS)
    if not node:
        node = store.get_node_by_qualified_name(name)
    if not node:
        return {"error": f"Class '{name}' not found."}

    from codegraph.graph.schema import EdgeType
    define_edges = store.get_edges_from(node.id, EdgeType.DEFINES)
    method_nodes = [store.get_node(e.target_id) for e in define_edges]
    methods = [_node_to_dict(m) for m in method_nodes if m and m.node_type == NodeType.METHOD]

    inherit_edges = store.get_edges_from(node.id, EdgeType.INHERITS)
    base_ids = [e.target_id for e in inherit_edges]
    bases = [_node_to_dict(store.get_node(bid)) for bid in base_ids if store.get_node(bid)]

    subclass_edges = store.get_edges_to(node.id, EdgeType.INHERITS)
    subclass_ids = [e.source_id for e in subclass_edges]
    subclasses = [_node_to_dict(store.get_node(sid)) for sid in subclass_ids if store.get_node(sid)]

    result = _node_to_dict(node)
    result["methods"] = methods
    result["base_classes"] = bases
    result["subclasses"] = subclasses
    return result


@mcp.tool()
def get_context(qualified_name: str, depth: int = 2) -> dict:
    """Get everything needed to understand or modify this symbol: callers, callees, related docs.
    Token-budgeted to stay within context limits. This is the most powerful tool — use it first."""
    store = _get_store()
    retriever = _get_retriever()
    config = _config or get_config()

    node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
    if not node:
        return {"error": f"Symbol '{qualified_name}' not found. Use search_code to find it."}

    subgraph = gq.get_subgraph(store, node.id, depth=depth, max_tokens=config.max_subgraph_tokens)
    docs = retriever.docs_for_node(node.id)

    return {
        "center": _node_to_dict(node),
        "related_nodes": [_node_to_dict(n) for n in subgraph["related_nodes"]],
        "edges": subgraph["edges"],
        "summary": subgraph["summary"],
        "estimated_tokens": subgraph["estimated_tokens"],
        "related_docs": docs[:3],
    }


@mcp.tool()
def find_callers(qualified_name: str) -> list[dict]:
    """Who calls this function? Critical before refactoring — shows all call sites."""
    store = _get_store()
    node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
    if not node:
        return [{"error": f"Symbol '{qualified_name}' not found."}]
    callers = gq.get_callers(store, node.id, depth=2)
    return [{"name": n.name, "qualified_name": n.qualified_name, "file": n.file_path, "line": n.start_line} for n in callers]


@mcp.tool()
def find_callees(qualified_name: str) -> list[dict]:
    """What does this function call? Understand its dependencies before modifying."""
    store = _get_store()
    node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
    if not node:
        return [{"error": f"Symbol '{qualified_name}' not found."}]
    callees = gq.get_callees(store, node.id, depth=2)
    return [{"name": n.name, "qualified_name": n.qualified_name, "file": n.file_path, "line": n.start_line} for n in callees]


@mcp.tool()
def search_code(pattern: str, node_type: str = "function") -> list[dict]:
    """Search for functions, classes, or methods by name pattern (substring match)."""
    store = _get_store()
    nt_map = {"function": NodeType.FUNCTION, "class": NodeType.CLASS, "method": NodeType.METHOD, "variable": NodeType.VARIABLE}
    nt = nt_map.get(node_type.lower())
    nodes = store.search_nodes(pattern, nt, limit=15)
    return [_node_to_dict(n) for n in nodes]


@mcp.tool()
def search_docs(query: str, k: int = 5) -> list[dict]:
    """Semantic search across codebase documentation and docstrings."""
    retriever = _get_retriever()
    return retriever.search_docs(query, k=k)


@mcp.tool()
def file_dependencies(file_path: str) -> dict:
    """What does this file import, and what imports it? Understand coupling before touching a file."""
    store = _get_store()
    deps = gq.get_dependencies(store, file_path, depth=2)
    dependents = gq.get_dependents(store, file_path, depth=2)
    return {
        "file": file_path,
        "imports": [{"name": n.name, "file": n.file_path} for n in deps],
        "imported_by": [{"name": n.name, "file": n.file_path} for n in dependents],
    }


@mcp.tool()
def codebase_stats() -> dict:
    """High-level overview: file counts, languages, key modules. Start here for orientation."""
    store = _get_store()
    return store.stats()


@mcp.tool()
def impact_analysis(qualified_name: str, depth: int = 5) -> dict:
    """What would break if this function changed?

    Performs reverse BFS through the call graph to find all functions that
    transitively depend on this one. Returns a risk level (low/medium/high)
    and a summary safe to include in a PR description or review comment.

    Use this before any refactor or signature change.
    """
    store = _get_store()
    node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
    if not node:
        return {"error": f"Symbol '{qualified_name}' not found."}

    result = gq.impact_analysis(store, node.id, depth=depth)
    return {
        "target": _node_to_dict(result["center"]),
        "risk_level": result["risk_level"],
        "summary": result["summary"],
        "immediate_callers": [
            {"name": n.name, "qualified_name": n.qualified_name, "file": n.file_path, "line": n.start_line}
            for n in result["immediate_callers"]
        ],
        "transitive_callers": [
            {"name": n.name, "qualified_name": n.qualified_name, "file": n.file_path}
            for n in result["transitive_callers"]
        ],
        "affected_files": result["affected_files"],
    }


@mcp.tool()
def find_dead_code() -> list[dict]:
    """Find functions with no callers — potential dead code to clean up.

    Excludes entry points (main, run, handler), dunder methods, test functions,
    and decorated functions that may serve as callbacks or route handlers.
    """
    store = _get_store()
    dead = gq.find_dead_code(store)
    return [
        {"name": n.name, "qualified_name": n.qualified_name, "file": n.file_path, "line": n.start_line}
        for n in dead
    ]


@mcp.tool()
def get_diagram(qualified_name: str, depth: int = 2) -> dict:
    """Get a Mermaid flowchart diagram of the call graph around a symbol.

    Returns a mermaid_diagram string you can paste into markdown.
    Also returns the text summary. Useful for explaining architecture to reviewers.
    """
    store = _get_store()
    config = _config or get_config()
    node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
    if not node:
        return {"error": f"Symbol '{qualified_name}' not found."}

    subgraph = gq.get_subgraph(store, node.id, depth=depth, max_tokens=config.max_subgraph_tokens)
    diagram = gq.subgraph_to_mermaid(subgraph)
    return {
        "qualified_name": node.qualified_name,
        "summary": subgraph["summary"],
        "mermaid_diagram": diagram,
        "node_count": len(subgraph["related_nodes"]) + 1,
        "edge_count": len(subgraph["edges"]),
    }


@mcp.tool()
def get_source(qualified_name: str) -> dict:
    """Get the exact source code of a function or class — AST-verified ground truth.

    Use this instead of guessing or recalling an implementation from memory.
    Prevents hallucination: the returned code is read directly from the file
    at the exact line range the AST recorded during indexing.

    Returns the source lines, file path, and line range so you can cite the location.
    """
    store = _get_store()
    node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
    if not node:
        return {"error": f"Symbol '{qualified_name}' not found. Use search_code to locate it."}
    try:
        lines = Path(node.file_path).read_text(errors="replace").splitlines()
        start = max(0, node.start_line - 1)
        end = min(len(lines), node.end_line)
        source = "\n".join(lines[start:end])
        return {
            "qualified_name": node.qualified_name,
            "file": node.file_path,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "language": node.language,
            "source": source,
            "provenance": "ast_parsed",  # distinguishes ground truth from agent memory
        }
    except OSError as e:
        return {"error": f"Could not read source: {e}"}


@mcp.tool()
def verify_signature(qualified_name: str, claimed_signature: str) -> dict:
    """Verify whether a claimed function signature matches the actual AST-parsed signature.

    Use this before citing or calling a function to avoid hallucinating its parameters.
    Pass what you *think* the signature is; get back whether it's correct and what it actually is.

    Example: verify_signature("auth.authenticate", "(username, password)")
    """
    store = _get_store()
    node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
    if not node:
        return {"error": f"Symbol '{qualified_name}' not found."}

    actual = node.signature or "(unknown)"
    # Normalize whitespace for comparison
    def normalize(s: str) -> str:
        return " ".join(s.split()).lower().replace(" ", "")

    match = normalize(claimed_signature) == normalize(actual)
    return {
        "qualified_name": node.qualified_name,
        "claimed": claimed_signature,
        "actual": actual,
        "match": match,
        "file": node.file_path,
        "line": node.start_line,
        "verdict": "CORRECT" if match else "WRONG — use the actual signature above",
    }


@mcp.tool()
def token_savings_estimate(qualified_name: str) -> dict:
    """Compare the token cost of querying codegraph vs reading files naively.

    Shows concretely how much context window you save by using codegraph.
    Reads the relevant files and counts tokens; compares with the targeted graph response.
    """
    store = _get_store()
    node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
    if not node:
        return {"error": f"Symbol '{qualified_name}' not found."}

    # Cost of naive approach: read all files in the same module
    file_nodes = store.get_nodes_for_file(node.file_path)
    try:
        full_file = Path(node.file_path).read_text(errors="replace")
    except OSError:
        full_file = ""

    # Also count imported files the agent would likely read
    from codegraph.graph import queries as gq
    from codegraph.graph.schema import EdgeType
    import_edges = store.get_edges_from(
        next((n.id for n in file_nodes if n.node_type.value == "module"), ""),
        EdgeType.IMPORTS,
    )
    imported_content = ""
    for edge in import_edges[:3]:  # cap at 3 to be fair
        dep = store.get_node(edge.target_id)
        if dep:
            try:
                imported_content += Path(dep.file_path).read_text(errors="replace")
            except OSError:
                pass

    naive_tokens = _count_tokens(full_file + imported_content)

    # Cost of codegraph approach: just the targeted function data
    config = _config or get_config()
    subgraph = gq.get_subgraph(store, node.id, depth=2, max_tokens=config.max_subgraph_tokens)
    graph_tokens = subgraph["estimated_tokens"]

    savings_pct = max(0, round((1 - graph_tokens / max(naive_tokens, 1)) * 100))

    return {
        "symbol": node.qualified_name,
        "naive_approach_tokens": naive_tokens,
        "codegraph_tokens": graph_tokens,
        "tokens_saved": max(0, naive_tokens - graph_tokens),
        "savings_percent": savings_pct,
        "explanation": (
            f"Reading '{node.file_path}' + imports naively costs ~{naive_tokens} tokens. "
            f"Querying codegraph for the same context costs ~{graph_tokens} tokens. "
            f"That's {savings_pct}% fewer tokens — more context window for actual reasoning."
        ),
    }


def _count_tokens(text: str) -> int:
    """Approximate token count using the standard 4-chars-per-token heuristic."""
    return max(1, len(text) // 4)


@mcp.tool()
def reindex() -> dict:
    """Re-scan the codebase for changes. Incremental by default — only re-indexes changed files."""
    config = _config or get_config()
    store = _get_store()
    if config.repo_path is None:
        return {"error": "No repo_path configured."}
    builder = GraphBuilder(config.repo_path, store, config)
    stats = builder.build(incremental=True)
    return {
        "files_indexed": stats.files_indexed,
        "files_skipped": stats.files_skipped,
        "nodes_created": stats.nodes_created,
        "edges_created": stats.edges_created,
        "time_elapsed": stats.time_elapsed,
        "errors": stats.errors,
    }
