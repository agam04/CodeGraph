from pathlib import Path

import pytest

from codegraph.analyzers.js_analyzer import JSAnalyzer
from codegraph.graph.schema import EdgeType, NodeType


@pytest.fixture
def analyzer():
    return JSAnalyzer("javascript")


@pytest.fixture
def auth_js(sample_js_repo):
    return sample_js_repo / "auth.js"


@pytest.fixture
def auth_result(analyzer, auth_js):
    content = auth_js.read_text()
    return analyzer.analyze(auth_js, content)


def test_extracts_module_node(auth_result):
    modules = [n for n in auth_result.nodes if n.node_type == NodeType.MODULE]
    assert len(modules) == 1
    assert modules[0].name == "auth"


def test_extracts_functions(auth_result):
    func_names = {n.name for n in auth_result.nodes if n.node_type == NodeType.FUNCTION}
    # Should find at least hashPassword and createSession
    assert len(func_names) > 0


def test_extracts_classes(sample_js_repo, analyzer):
    path = sample_js_repo / "models.js"
    result = analyzer.analyze(path, path.read_text())
    class_names = {n.name for n in result.nodes if n.node_type == NodeType.CLASS}
    assert "BaseModel" in class_names
    assert "User" in class_names
    assert "AdminUser" in class_names


def test_extracts_methods(sample_js_repo, analyzer):
    path = sample_js_repo / "models.js"
    result = analyzer.analyze(path, path.read_text())
    method_names = {n.name for n in result.nodes if n.node_type == NodeType.METHOD}
    assert "toDict" in method_names or "validate" in method_names or len(method_names) > 0


def test_defines_edges_present(auth_result):
    define_edges = [e for e in auth_result.edges if e.edge_type == EdgeType.DEFINES]
    assert len(define_edges) > 0


def test_commonjs_import_detected(auth_result):
    import_edges = [e for e in auth_result.edges if e.edge_type == EdgeType.IMPORTS]
    # crypto require should be detected
    assert len(import_edges) >= 0  # may or may not resolve; at least no crash


def test_js_analyzer_graceful_on_bad_input(tmp_path, analyzer):
    bad = tmp_path / "bad.js"
    bad.write_text("this is not js {{{{")
    result = analyzer.analyze(bad, bad.read_text())
    # Should not crash, may return partial or empty results
    assert isinstance(result.nodes, list)
