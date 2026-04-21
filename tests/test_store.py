
from codegraph.graph.schema import Edge, EdgeType, Node, NodeType


def _make_node(name: str, ntype: NodeType = NodeType.FUNCTION, file_path: str = "test.py") -> Node:
    from codegraph.utils.hashing import node_id
    nid = node_id(file_path, name)
    return Node(
        id=nid,
        node_type=ntype,
        name=name,
        qualified_name=name,
        file_path=file_path,
        start_line=1,
        end_line=10,
        language="python",
        docstring=f"Docstring for {name}",
        signature="(x: int) -> str",
        source_hash="abc123",
    )


def test_upsert_and_get_node(tmp_db):
    node = _make_node("foo")
    tmp_db.upsert_node(node)
    tmp_db.commit()
    retrieved = tmp_db.get_node(node.id)
    assert retrieved is not None
    assert retrieved.name == "foo"
    assert retrieved.node_type == NodeType.FUNCTION
    assert retrieved.docstring == "Docstring for foo"


def test_get_node_by_name(tmp_db):
    node = _make_node("bar")
    tmp_db.upsert_node(node)
    tmp_db.commit()
    retrieved = tmp_db.get_node_by_name("bar", NodeType.FUNCTION)
    assert retrieved is not None
    assert retrieved.name == "bar"


def test_get_node_by_qualified_name(tmp_db):
    node = _make_node("mymodule.myfunc")
    tmp_db.upsert_node(node)
    tmp_db.commit()
    retrieved = tmp_db.get_node_by_qualified_name("mymodule.myfunc")
    assert retrieved is not None


def test_search_nodes(tmp_db):
    for name in ["authenticate", "auth_token", "logout"]:
        tmp_db.upsert_node(_make_node(name))
    tmp_db.commit()
    results = tmp_db.search_nodes("auth", NodeType.FUNCTION)
    names = {n.name for n in results}
    assert "authenticate" in names
    assert "auth_token" in names
    assert "logout" not in names


def test_upsert_edges(tmp_db):
    n1 = _make_node("caller")
    n2 = _make_node("callee")
    tmp_db.upsert_nodes([n1, n2])
    edge = Edge(n1.id, n2.id, EdgeType.CALLS, {"line": 42})
    tmp_db.upsert_edges([edge])
    tmp_db.commit()

    edges_from = tmp_db.get_edges_from(n1.id, EdgeType.CALLS)
    assert len(edges_from) == 1
    assert edges_from[0].target_id == n2.id
    assert edges_from[0].metadata["line"] == 42


def test_get_edges_to(tmp_db):
    n1 = _make_node("a")
    n2 = _make_node("b")
    tmp_db.upsert_nodes([n1, n2])
    tmp_db.upsert_edges([Edge(n1.id, n2.id, EdgeType.CALLS)])
    tmp_db.commit()
    edges = tmp_db.get_edges_to(n2.id, EdgeType.CALLS)
    assert len(edges) == 1
    assert edges[0].source_id == n1.id


def test_file_hash_tracking(tmp_db):
    assert tmp_db.get_file_hash("foo.py") is None
    tmp_db.upsert_file("foo.py", "abc123", "python")
    tmp_db.commit()
    assert tmp_db.get_file_hash("foo.py") == "abc123"


def test_incremental_skip(tmp_db):
    tmp_db.upsert_file("same.py", "hash1", "python")
    tmp_db.commit()
    # Simulating builder check: same hash → skip
    stored = tmp_db.get_file_hash("same.py")
    assert stored == "hash1"
    # Different hash → should reindex
    tmp_db.upsert_file("same.py", "hash2", "python")
    tmp_db.commit()
    assert tmp_db.get_file_hash("same.py") == "hash2"


def test_delete_file_nodes(tmp_db):
    n = _make_node("old_func", file_path="old.py")
    tmp_db.upsert_node(n)
    tmp_db.commit()
    assert tmp_db.get_node(n.id) is not None
    tmp_db.delete_file_nodes("old.py")
    tmp_db.commit()
    assert tmp_db.get_node(n.id) is None


def test_stats(tmp_db):
    for i in range(3):
        tmp_db.upsert_node(_make_node(f"func{i}"))
    tmp_db.upsert_node(_make_node("MyClass", NodeType.CLASS))
    tmp_db.commit()
    s = tmp_db.stats()
    assert s["total_nodes"] >= 4
    assert s["type_counts"].get("function", 0) >= 3


def test_upsert_is_idempotent(tmp_db):
    node = _make_node("idem")
    tmp_db.upsert_node(node)
    tmp_db.upsert_node(node)
    tmp_db.commit()
    results = tmp_db.search_nodes("idem", NodeType.FUNCTION)
    assert len(results) == 1
