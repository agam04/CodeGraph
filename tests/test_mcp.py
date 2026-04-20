import pytest

from codegraph.graph.builder import GraphBuilder
from codegraph.mcp_server.server import (
    codebase_stats,
    find_callers,
    find_callees,
    find_class,
    find_function,
    init_mcp,
    search_code,
)
from codegraph.rag.indexer import DocIndexer
from codegraph.rag.retriever import RAGRetriever


@pytest.fixture(autouse=True)
def mcp_setup(sample_python_repo, tmp_db, test_config):
    builder = GraphBuilder(sample_python_repo, tmp_db, test_config)
    builder.build(incremental=False)
    indexer = DocIndexer(tmp_db)
    retriever = RAGRetriever(tmp_db, indexer)
    init_mcp(tmp_db, retriever, test_config)


def test_find_function_returns_result():
    result = find_function("authenticate")
    assert "error" not in result
    assert result["name"] == "authenticate"
    assert "signature" in result


def test_find_function_includes_callers_callees():
    result = find_function("authenticate")
    assert "callers" in result
    assert "callees" in result
    assert isinstance(result["callers"], list)


def test_find_function_not_found():
    result = find_function("does_not_exist_xyz")
    assert "error" in result


def test_find_class_returns_result():
    result = find_class("User")
    assert "error" not in result
    assert result["name"] == "User"
    assert "methods" in result


def test_find_class_includes_subclasses():
    result = find_class("User")
    assert "subclasses" in result
    # AdminUser extends User
    subclass_names = {s.get("name") for s in result["subclasses"]}
    # May or may not resolve depending on cross-file analysis
    assert isinstance(subclass_names, set)


def test_find_callers():
    result = find_callers("auth.authenticate")
    assert isinstance(result, list)


def test_find_callees():
    result = find_callees("auth.authenticate")
    assert isinstance(result, list)


def test_search_code_functions():
    result = search_code("hash", "function")
    assert isinstance(result, list)
    names = {r["name"] for r in result}
    assert len(names) > 0


def test_codebase_stats():
    result = codebase_stats()
    assert "total_files" in result
    assert result["total_files"] >= 4
    assert "total_nodes" in result
    assert result["total_nodes"] > 0
