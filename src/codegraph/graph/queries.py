from collections import deque

from codegraph.graph.schema import EdgeType, Node, NodeType
from codegraph.graph.store import GraphStore


# ── Basic traversals ──────────────────────────────────────────────────────────

def get_callers(store: GraphStore, node_id: str, depth: int = 1) -> list[Node]:
    visited: set[str] = {node_id}
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    result: list[Node] = []
    while queue:
        current_id, d = queue.popleft()
        if d >= depth:
            continue
        for edge in store.get_edges_to(current_id, EdgeType.CALLS):
            if edge.source_id not in visited:
                visited.add(edge.source_id)
                caller = store.get_node(edge.source_id)
                if caller:
                    result.append(caller)
                    queue.append((edge.source_id, d + 1))
    return result


def get_callees(store: GraphStore, node_id: str, depth: int = 1) -> list[Node]:
    visited: set[str] = {node_id}
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    result: list[Node] = []
    while queue:
        current_id, d = queue.popleft()
        if d >= depth:
            continue
        for edge in store.get_edges_from(current_id, EdgeType.CALLS):
            if edge.target_id not in visited:
                visited.add(edge.target_id)
                callee = store.get_node(edge.target_id)
                if callee:
                    result.append(callee)
                    queue.append((edge.target_id, d + 1))
    return result


def get_dependencies(store: GraphStore, file_path: str, depth: int = 1) -> list[Node]:
    file_nodes = store.get_nodes_for_file(file_path)
    module_node = next((n for n in file_nodes if n.node_type == NodeType.MODULE), None)
    if not module_node:
        return []
    visited: set[str] = {module_node.id}
    queue: deque[tuple[str, int]] = deque([(module_node.id, 0)])
    result: list[Node] = []
    while queue:
        current_id, d = queue.popleft()
        if d >= depth:
            continue
        for edge in store.get_edges_from(current_id, EdgeType.IMPORTS):
            if edge.target_id not in visited:
                visited.add(edge.target_id)
                dep = store.get_node(edge.target_id)
                if dep:
                    result.append(dep)
                    queue.append((edge.target_id, d + 1))
    return result


def get_dependents(store: GraphStore, file_path: str, depth: int = 1) -> list[Node]:
    file_nodes = store.get_nodes_for_file(file_path)
    module_node = next((n for n in file_nodes if n.node_type == NodeType.MODULE), None)
    if not module_node:
        return []
    visited: set[str] = {module_node.id}
    queue: deque[tuple[str, int]] = deque([(module_node.id, 0)])
    result: list[Node] = []
    while queue:
        current_id, d = queue.popleft()
        if d >= depth:
            continue
        for edge in store.get_edges_to(current_id, EdgeType.IMPORTS):
            if edge.source_id not in visited:
                visited.add(edge.source_id)
                dep = store.get_node(edge.source_id)
                if dep:
                    result.append(dep)
                    queue.append((edge.source_id, d + 1))
    return result


# ── Fan-in scoring ────────────────────────────────────────────────────────────

def fan_in_score(store: GraphStore, node_id: str) -> int:
    """Count of CALLS edges pointing to this node — a proxy for how critical it is."""
    return len(store.get_edges_to(node_id, EdgeType.CALLS))


# ── Token-budgeted subgraph with fan-in ranking ───────────────────────────────

def get_subgraph(
    store: GraphStore,
    center_node_id: str,
    depth: int = 2,
    max_tokens: int = 4000,
) -> dict:
    center = store.get_node(center_node_id)
    if not center:
        return {"center_node": None, "related_nodes": [], "edges": [], "summary": "", "estimated_tokens": 0}

    visited: set[str] = {center_node_id}
    # BFS: collect candidates with their fan-in score so we can rank within budget
    candidates: list[tuple[Node, int]] = []  # (node, fan_in)

    queue: deque[tuple[str, int]] = deque([(center_node_id, 0)])
    subgraph_edges: list[dict] = []

    while queue:
        current_id, d = queue.popleft()
        if d >= depth:
            continue
        all_edges = store.get_edges_from(current_id) + store.get_edges_to(current_id)
        for edge in all_edges:
            other_id = edge.target_id if edge.source_id == current_id else edge.source_id
            subgraph_edges.append({
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_type": edge.edge_type.value,
            })
            if other_id not in visited:
                visited.add(other_id)
                other = store.get_node(other_id)
                if other:
                    score = fan_in_score(store, other_id)
                    candidates.append((other, score))
                    queue.append((other_id, d + 1))

    # Rank by fan-in score (most-called first) and fill token budget
    candidates.sort(key=lambda x: x[1], reverse=True)
    related: list[Node] = []
    token_estimate = _node_tokens(center)
    for node, _ in candidates:
        node_tokens = _node_tokens(node)
        if token_estimate + node_tokens > max_tokens:
            continue
        token_estimate += node_tokens
        related.append(node)

    # Deduplicate edges
    seen_edges: set[tuple] = set()
    unique_edges = []
    for e in subgraph_edges:
        key = (e["source_id"], e["target_id"], e["edge_type"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    return {
        "center_node": center,
        "related_nodes": related,
        "edges": unique_edges,
        "summary": _generate_summary(center, related),
        "estimated_tokens": token_estimate,
    }


# ── Impact analysis ───────────────────────────────────────────────────────────

def impact_analysis(store: GraphStore, node_id: str, depth: int = 5) -> dict:
    """Reverse BFS: what functions and files transitively depend on this node?

    Returns a structured result with risk scoring that agents can act on.
    """
    center = store.get_node(node_id)
    if not center:
        return {}

    # Collect immediate and transitive callers separately
    immediate: list[Node] = []
    transitive: list[Node] = []
    visited: set[str] = {node_id}

    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    while queue:
        current_id, d = queue.popleft()
        if d >= depth:
            continue
        for edge in store.get_edges_to(current_id, EdgeType.CALLS):
            if edge.source_id not in visited:
                visited.add(edge.source_id)
                caller = store.get_node(edge.source_id)
                if caller:
                    if d == 0:
                        immediate.append(caller)
                    else:
                        transitive.append(caller)
                    queue.append((edge.source_id, d + 1))

    # Also find files that import the file containing this node
    affected_files: set[str] = set()
    for node in [center] + immediate + transitive:
        affected_files.add(node.file_path)

    # Risk scoring: high fan-in + deep transitive reach = high risk
    direct_count = len(immediate)
    transitive_count = len(transitive)
    total = direct_count + transitive_count
    if total == 0:
        risk = "low"
    elif total <= 3:
        risk = "low"
    elif total <= 10:
        risk = "medium"
    else:
        risk = "high"

    summary_parts = [f"`{center.qualified_name}` is called by {direct_count} function(s) directly"]
    if transitive_count:
        summary_parts.append(f"and {transitive_count} more transitively")
    summary_parts.append(f"across {len(affected_files)} file(s).")
    if risk == "high":
        summary_parts.append("HIGH RISK: changes here will have wide impact.")
    elif risk == "medium":
        summary_parts.append("MEDIUM RISK: review callers before modifying.")

    return {
        "center": center,
        "immediate_callers": immediate,
        "transitive_callers": transitive,
        "affected_files": sorted(affected_files),
        "risk_level": risk,
        "summary": " ".join(summary_parts),
    }


# ── Dead code detection ───────────────────────────────────────────────────────

_ENTRY_POINT_NAMES = {
    "main", "__init__", "__main__", "setup", "teardown",
    "run", "start", "app", "handler", "lambda_handler",
    "pytest_configure", "celery_app",
}

_DECORATOR_ENTRY_POINTS = {"app.route", "router.get", "router.post", "task", "pytest.fixture"}


def find_dead_code(store: GraphStore) -> list[Node]:
    """Return functions/methods with no known callers and no entry-point indicators."""
    functions = store.get_all_nodes(NodeType.FUNCTION) + store.get_all_nodes(NodeType.METHOD)
    dead: list[Node] = []
    for node in functions:
        # Skip obvious entry points
        if node.name.lower() in _ENTRY_POINT_NAMES:
            continue
        # Skip test functions
        if node.name.startswith("test_") or node.name.startswith("Test"):
            continue
        # Skip decorated functions that could be entry points
        decorators = node.metadata.get("decorators", [])
        if any(d in _DECORATOR_ENTRY_POINTS for d in decorators):
            continue
        # Skip dunder methods
        if node.name.startswith("__") and node.name.endswith("__"):
            continue
        callers = store.get_edges_to(node.id, EdgeType.CALLS)
        if not callers:
            dead.append(node)

    # Sort by file then line for readable output
    dead.sort(key=lambda n: (n.file_path, n.start_line))
    return dead


# ── Mermaid diagram export ────────────────────────────────────────────────────

def subgraph_to_mermaid(subgraph: dict) -> str:
    """Convert a subgraph dict to a Mermaid flowchart string.

    Agents can embed this in markdown responses for a visual call graph.
    """
    center = subgraph.get("center_node")
    related = subgraph.get("related_nodes", [])
    edges = subgraph.get("edges", [])

    all_nodes: dict[str, Node] = {}
    if center:
        all_nodes[center.id] = center
    for n in related:
        all_nodes[n.id] = n

    # Build safe Mermaid node IDs (alphanumeric only)
    def safe_id(node_id: str) -> str:
        return "n" + node_id[:12].replace("-", "")

    def node_label(node: Node) -> str:
        return f"{node.name}\\n[{node.node_type.value}]"

    lines = ["```mermaid", "flowchart TD"]

    for node in all_nodes.values():
        nid = safe_id(node.id)
        label = node_label(node)
        if center and node.id == center.id:
            lines.append(f"    {nid}[\"{label}\"]:::center")
        else:
            lines.append(f"    {nid}[\"{label}\"]")

    # Edge type to arrow style
    _ARROWS = {
        "calls": "-->",
        "imports": "-.->",
        "defines": "--o",
        "inherits": "-->|inherits|",
        "references": "..->",
        "documents": "-.->|docs|",
    }

    seen_mermaid_edges: set[tuple] = set()
    for edge in edges:
        src = edge.get("source_id", "")
        tgt = edge.get("target_id", "")
        etype = edge.get("edge_type", "")
        if src not in all_nodes or tgt not in all_nodes:
            continue
        key = (src, tgt, etype)
        if key in seen_mermaid_edges:
            continue
        seen_mermaid_edges.add(key)
        arrow = _ARROWS.get(etype, "-->")
        lines.append(f"    {safe_id(src)} {arrow} {safe_id(tgt)}")

    if center:
        lines.append("    classDef center fill:#f90,stroke:#333,color:#000")

    lines.append("```")
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _node_tokens(node: Node) -> int:
    text = f"{node.qualified_name} {node.signature or ''} {node.docstring or ''}"
    return max(10, len(text) // 4)


def _generate_summary(center: Node, related: list[Node]) -> str:
    callers = [n for n in related if n.node_type in (NodeType.FUNCTION, NodeType.METHOD)]
    files = {n.file_path for n in related}
    parts = [f"Subgraph centered on `{center.qualified_name}` ({center.node_type.value})."]
    if center.docstring:
        parts.append(center.docstring.split(".")[0] + ".")
    if callers:
        names = ", ".join(n.name for n in callers[:5])
        parts.append(f"Related functions: {names}.")
    if files:
        parts.append(f"Spans {len(files)} file(s).")
    return " ".join(parts)
