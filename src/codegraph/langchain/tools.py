"""LangChain tool wrappers for all 16 codegraph graph tools.

Each tool is created via a factory closure so the GraphStore and RAGRetriever
are bound at construction time, not looked up from a global. This makes the
tools testable (pass a mock store) and safe to use in multi-tenant contexts.

Usage:
    tools = make_codegraph_tools(store, rag_retriever)
    # Pass to a LangGraph agent or any LangChain AgentExecutor
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from codegraph.graph import queries as gq
from codegraph.graph.schema import EdgeType, NodeType
from codegraph.graph.store import GraphStore
from codegraph.rag.retriever import RAGRetriever


def _node_dict(node: Any) -> dict:
    if node is None:
        return {}
    return {
        "name": node.name,
        "qualified_name": node.qualified_name,
        "type": node.node_type.value,
        "file": node.file_path,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "signature": node.signature,
        "docstring": node.docstring,
        "language": node.language,
    }


def make_codegraph_tools(store: GraphStore, rag: RAGRetriever | None = None) -> list:
    """Return all 16 LangChain tools bound to *store* and *rag*.

    Categories (used by the agent router):
        structural — callers, callees, impact, dead code, dependencies, context
        lookup     — find_function, find_class, get_source, verify_signature,
                     get_diagram, token_savings_estimate
        semantic   — search_docs, search_code, codebase_stats, reindex
    """

    # ── structural ────────────────────────────────────────────────────────────

    @tool
    def find_callers(qualified_name: str) -> list[dict]:
        """Who calls this function or method? Returns all call sites up to depth 2.
        Use before any refactor to understand the blast radius."""
        node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
        if not node:
            return [{"error": f"Symbol '{qualified_name}' not found — try search_code first."}]
        callers = gq.get_callers(store, node.id, depth=2)
        return [
            {"name": n.name, "qualified_name": n.qualified_name, "file": n.file_path, "line": n.start_line}
            for n in callers
        ]

    @tool
    def find_callees(qualified_name: str) -> list[dict]:
        """What functions does this one call? Returns direct and transitive callees up to depth 2.
        Use to understand dependencies before modifying a function."""
        node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
        if not node:
            return [{"error": f"Symbol '{qualified_name}' not found — try search_code first."}]
        callees = gq.get_callees(store, node.id, depth=2)
        return [
            {"name": n.name, "qualified_name": n.qualified_name, "file": n.file_path, "line": n.start_line}
            for n in callees
        ]

    @tool
    def impact_analysis(qualified_name: str) -> dict:
        """What breaks if this function changes? Reverse BFS through call graph.
        Returns risk level (low/medium/high), immediate callers, transitive callers,
        and affected files. Always run this before refactoring a shared function."""
        node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
        if not node:
            return {"error": f"Symbol '{qualified_name}' not found."}
        result = gq.impact_analysis(store, node.id, depth=5)
        return {
            "target": _node_dict(result["center"]),
            "risk_level": result["risk_level"],
            "summary": result["summary"],
            "immediate_callers": [
                {"name": n.name, "qualified_name": n.qualified_name, "file": n.file_path, "line": n.start_line}
                for n in result["immediate_callers"]
            ],
            "transitive_callers": [
                {"name": n.name, "qualified_name": n.qualified_name}
                for n in result["transitive_callers"]
            ],
            "affected_files": result["affected_files"],
        }

    @tool
    def find_dead_code() -> list[dict]:
        """Find functions with zero callers — candidates for deletion.
        Excludes entry points, dunders, test functions, and decorated handlers."""
        dead = gq.find_dead_code(store)
        return [
            {"name": n.name, "qualified_name": n.qualified_name, "file": n.file_path, "line": n.start_line}
            for n in dead
        ]

    @tool
    def file_dependencies(file_path: str) -> dict:
        """What does this file import, and what files import it?
        Shows coupling — use before moving or deleting a file."""
        deps = gq.get_dependencies(store, file_path, depth=2)
        dependents = gq.get_dependents(store, file_path, depth=2)
        return {
            "file": file_path,
            "imports": [{"name": n.name, "file": n.file_path} for n in deps],
            "imported_by": [{"name": n.name, "file": n.file_path} for n in dependents],
        }

    @tool
    def get_context(qualified_name: str, depth: int = 2) -> dict:
        """Token-budgeted subgraph around a symbol: callers, callees, related docs.
        Fan-in ranked — most-called nodes surface first within the token budget.
        The most powerful tool for understanding before changing code."""
        node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
        if not node:
            return {"error": f"Symbol '{qualified_name}' not found."}
        subgraph = gq.get_subgraph(store, node.id, depth=depth, max_tokens=4000)
        docs = rag.docs_for_node(node.id) if rag else []
        return {
            "center": _node_dict(node),
            "related_nodes": [_node_dict(n) for n in subgraph["related_nodes"]],
            "edges": subgraph["edges"],
            "summary": subgraph["summary"],
            "estimated_tokens": subgraph["estimated_tokens"],
            "related_docs": docs[:3],
        }

    # ── lookup ────────────────────────────────────────────────────────────────

    @tool
    def find_function(name: str) -> dict:
        """Find a function or method by name. Returns its location, signature,
        docstring, callers, and callees. Use this to confirm a function exists
        and get its exact qualified name before calling other tools."""
        node = (
            store.get_node_by_name(name, NodeType.FUNCTION)
            or store.get_node_by_name(name, NodeType.METHOD)
            or store.get_node_by_qualified_name(name)
        )
        if not node:
            return {"error": f"Function '{name}' not found. Try search_code to find similar names."}
        callers = gq.get_callers(store, node.id, depth=1)
        callees = gq.get_callees(store, node.id, depth=1)
        result = _node_dict(node)
        result["callers"] = [{"name": n.name, "file": n.file_path, "line": n.start_line} for n in callers[:10]]
        result["callees"] = [{"name": n.name, "file": n.file_path, "line": n.start_line} for n in callees[:10]]
        return result

    @tool
    def find_class(name: str) -> dict:
        """Find a class by name. Returns its location, methods, base classes,
        and known subclasses. Use to understand an inheritance hierarchy."""
        node = store.get_node_by_name(name, NodeType.CLASS) or store.get_node_by_qualified_name(name)
        if not node:
            return {"error": f"Class '{name}' not found."}
        define_edges = store.get_edges_from(node.id, EdgeType.DEFINES)
        methods = [
            _node_dict(store.get_node(e.target_id))
            for e in define_edges
            if store.get_node(e.target_id) and store.get_node(e.target_id).node_type == NodeType.METHOD
        ]
        inherit_edges = store.get_edges_from(node.id, EdgeType.INHERITS)
        bases = [_node_dict(store.get_node(e.target_id)) for e in inherit_edges if store.get_node(e.target_id)]
        subclass_edges = store.get_edges_to(node.id, EdgeType.INHERITS)
        subclasses = [_node_dict(store.get_node(e.source_id)) for e in subclass_edges if store.get_node(e.source_id)]
        result = _node_dict(node)
        result["methods"] = methods
        result["base_classes"] = bases
        result["subclasses"] = subclasses
        return result

    @tool
    def get_source(qualified_name: str) -> dict:
        """Get the exact source code of a function or class — AST-verified ground truth.
        NEVER guess an implementation from memory — use this tool instead.
        Returns source lines, file path, and line range with provenance tag."""
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
                "provenance": "ast_parsed",
            }
        except OSError as e:
            return {"error": f"Could not read source: {e}"}

    @tool
    def verify_signature(qualified_name: str, claimed_signature: str) -> dict:
        """Check whether a claimed function signature matches the actual AST-parsed one.
        Use before citing or calling a function to prevent hallucinating parameters.
        Example: verify_signature('auth.authenticate', '(username, password)')"""
        node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
        if not node:
            return {"error": f"Symbol '{qualified_name}' not found."}
        actual = node.signature or "(unknown)"

        def normalize(s: str) -> str:
            return " ".join(s.split()).lower().replace(" ", "")

        match = normalize(claimed_signature) == normalize(actual)
        return {
            "qualified_name": node.qualified_name,
            "claimed": claimed_signature,
            "actual": actual,
            "match": match,
            "verdict": "CORRECT" if match else "WRONG — use the actual signature above",
        }

    @tool
    def get_diagram(qualified_name: str, depth: int = 2) -> dict:
        """Generate a Mermaid flowchart of the call graph around a symbol.
        Paste the returned mermaid_diagram into markdown to render it.
        Useful for explaining architecture in PRs or documentation."""
        node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
        if not node:
            return {"error": f"Symbol '{qualified_name}' not found."}
        subgraph = gq.get_subgraph(store, node.id, depth=depth, max_tokens=4000)
        diagram = gq.subgraph_to_mermaid(subgraph)
        return {
            "qualified_name": node.qualified_name,
            "summary": subgraph["summary"],
            "mermaid_diagram": diagram,
            "node_count": len(subgraph["related_nodes"]) + 1,
            "edge_count": len(subgraph["edges"]),
        }

    @tool
    def token_savings_estimate(qualified_name: str) -> dict:
        """Compare token cost: querying codegraph vs reading files naively.
        Shows how much context window is saved per symbol lookup.
        Useful for justifying codegraph adoption in a team setting."""
        node = store.get_node_by_qualified_name(qualified_name) or store.get_node_by_name(qualified_name)
        if not node:
            return {"error": f"Symbol '{qualified_name}' not found."}
        file_nodes = store.get_nodes_for_file(node.file_path)
        try:
            full_file = Path(node.file_path).read_text(errors="replace")
        except OSError:
            full_file = ""
        module_id = next((n.id for n in file_nodes if n.node_type.value == "module"), "")
        import_edges = store.get_edges_from(module_id, EdgeType.IMPORTS) if module_id else []
        imported = ""
        for edge in import_edges[:3]:
            dep = store.get_node(edge.target_id)
            if dep:
                try:
                    imported += Path(dep.file_path).read_text(errors="replace")
                except OSError:
                    pass
        naive_tokens = max(1, len(full_file + imported) // 4)
        subgraph = gq.get_subgraph(store, node.id, depth=2, max_tokens=4000)
        graph_tokens = subgraph["estimated_tokens"]
        savings_pct = max(0, round((1 - graph_tokens / naive_tokens) * 100))
        return {
            "symbol": node.qualified_name,
            "naive_approach_tokens": naive_tokens,
            "codegraph_tokens": graph_tokens,
            "tokens_saved": max(0, naive_tokens - graph_tokens),
            "savings_percent": savings_pct,
            "explanation": (
                f"Reading '{node.file_path}' + imports naively costs ~{naive_tokens} tokens. "
                f"Querying codegraph costs ~{graph_tokens} tokens. "
                f"That's {savings_pct}% fewer tokens."
            ),
        }

    # ── semantic ──────────────────────────────────────────────────────────────

    @tool
    def search_code(pattern: str, node_type: str = "function") -> list[dict]:
        """Substring search across all function/class/method names in the graph.
        Use to find a symbol when you only know part of its name.
        node_type: 'function', 'class', 'method', or 'variable'"""
        nt_map = {
            "function": NodeType.FUNCTION,
            "class": NodeType.CLASS,
            "method": NodeType.METHOD,
            "variable": NodeType.VARIABLE,
        }
        nt = nt_map.get(node_type.lower())
        nodes = store.search_nodes(pattern, nt, limit=15)
        return [_node_dict(n) for n in nodes]

    @tool
    def search_docs(query: str, k: int = 5) -> list[dict]:
        """Hybrid semantic search: BM25 keyword + CodeBERT code embeddings + MiniLM doc embeddings,
        merged with Reciprocal Rank Fusion. Best for open-ended questions like
        'how does authentication work?' or 'find the rate limiting logic'."""
        if rag is None:
            return [{"error": "RAG retriever not available."}]
        return rag.search_docs(query, k=k)

    @tool
    def codebase_stats() -> dict:
        """High-level overview of the indexed codebase: file counts, languages,
        function/class totals, last indexed timestamp. Start here for orientation."""
        return store.stats()

    @tool
    def reindex() -> dict:
        """Trigger an incremental reindex — only re-parses files that changed since last index.
        Run after making edits so the graph reflects the current state of the codebase."""
        from codegraph.config import get_config
        from codegraph.graph.builder import GraphBuilder
        config = get_config()
        if config.repo_path is None:
            return {"error": "No repo_path configured. Set CODEGRAPH_REPO_PATH."}
        builder = GraphBuilder(config.repo_path, store, config)
        stats = builder.build(incremental=True)
        return {
            "files_indexed": stats.files_indexed,
            "files_skipped": stats.files_skipped,
            "nodes_created": stats.nodes_created,
            "edges_created": stats.edges_created,
            "errors": stats.errors,
        }

    return [
        # structural
        find_callers,
        find_callees,
        impact_analysis,
        find_dead_code,
        file_dependencies,
        get_context,
        # lookup
        find_function,
        find_class,
        get_source,
        verify_signature,
        get_diagram,
        token_savings_estimate,
        # semantic
        search_code,
        search_docs,
        codebase_stats,
        reindex,
    ]


# Tool category mapping — used by the agent router to narrow the active toolset
TOOL_CATEGORIES: dict[str, list[str]] = {
    "structural": [
        "find_callers", "find_callees", "impact_analysis",
        "find_dead_code", "file_dependencies", "get_context",
    ],
    "lookup": [
        "find_function", "find_class", "get_source",
        "verify_signature", "get_diagram", "token_savings_estimate",
    ],
    "semantic": [
        "search_docs", "search_code", "codebase_stats", "find_function",
    ],
}
