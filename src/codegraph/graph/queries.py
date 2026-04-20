from collections import deque
from typing import Optional

from codegraph.graph.schema import EdgeType, Node, NodeType
from codegraph.graph.store import GraphStore


def get_callers(store: GraphStore, node_id: str, depth: int = 1) -> list[Node]:
    """Return all nodes that call the given node, up to `depth` hops."""
    visited: set[str] = {node_id}
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    result: list[Node] = []

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        for edge in store.get_edges_to(current_id, EdgeType.CALLS):
            if edge.source_id not in visited:
                visited.add(edge.source_id)
                caller = store.get_node(edge.source_id)
                if caller:
                    result.append(caller)
                    queue.append((edge.source_id, current_depth + 1))

    return result


def get_callees(store: GraphStore, node_id: str, depth: int = 1) -> list[Node]:
    """Return all nodes called by the given node, up to `depth` hops."""
    visited: set[str] = {node_id}
    queue: deque[tuple[str, int]] = deque([(node_id, 0)])
    result: list[Node] = []

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        for edge in store.get_edges_from(current_id, EdgeType.CALLS):
            if edge.target_id not in visited:
                visited.add(edge.target_id)
                callee = store.get_node(edge.target_id)
                if callee:
                    result.append(callee)
                    queue.append((edge.target_id, current_depth + 1))

    return result


def get_dependencies(store: GraphStore, file_path: str, depth: int = 1) -> list[Node]:
    """Return all files that this file imports, up to `depth` hops."""
    file_nodes = store.get_nodes_for_file(file_path)
    module_node = next((n for n in file_nodes if n.node_type == NodeType.MODULE), None)
    if not module_node:
        return []

    visited: set[str] = {module_node.id}
    queue: deque[tuple[str, int]] = deque([(module_node.id, 0)])
    result: list[Node] = []

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        for edge in store.get_edges_from(current_id, EdgeType.IMPORTS):
            if edge.target_id not in visited:
                visited.add(edge.target_id)
                dep = store.get_node(edge.target_id)
                if dep:
                    result.append(dep)
                    queue.append((edge.target_id, current_depth + 1))

    return result


def get_dependents(store: GraphStore, file_path: str, depth: int = 1) -> list[Node]:
    """Return all files that import this file, up to `depth` hops."""
    file_nodes = store.get_nodes_for_file(file_path)
    module_node = next((n for n in file_nodes if n.node_type == NodeType.MODULE), None)
    if not module_node:
        return []

    visited: set[str] = {module_node.id}
    queue: deque[tuple[str, int]] = deque([(module_node.id, 0)])
    result: list[Node] = []

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        for edge in store.get_edges_to(current_id, EdgeType.IMPORTS):
            if edge.source_id not in visited:
                visited.add(edge.source_id)
                dep = store.get_node(edge.source_id)
                if dep:
                    result.append(dep)
                    queue.append((edge.source_id, current_depth + 1))

    return result


def get_subgraph(
    store: GraphStore,
    center_node_id: str,
    depth: int = 2,
    max_tokens: int = 4000,
) -> dict:
    """BFS subgraph around a node. Returns nodes and edges within token budget."""
    center = store.get_node(center_node_id)
    if not center:
        return {"center_node": None, "related_nodes": [], "edges": [], "summary": "", "estimated_tokens": 0}

    visited: set[str] = {center_node_id}
    queue: deque[tuple[str, int]] = deque([(center_node_id, 0)])
    related: list[Node] = []
    subgraph_edges: list[dict] = []
    token_estimate = _node_tokens(center)

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
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
                other = store.get_node(other_id)
                if other:
                    node_tokens = _node_tokens(other)
                    if token_estimate + node_tokens > max_tokens:
                        continue
                    token_estimate += node_tokens
                    visited.add(other_id)
                    related.append(other)
                    queue.append((other_id, current_depth + 1))

    # Deduplicate edges
    seen_edges: set[tuple] = set()
    unique_edges = []
    for e in subgraph_edges:
        key = (e["source_id"], e["target_id"], e["edge_type"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    summary = _generate_summary(center, related)
    return {
        "center_node": center,
        "related_nodes": related,
        "edges": unique_edges,
        "summary": summary,
        "estimated_tokens": token_estimate,
    }


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
