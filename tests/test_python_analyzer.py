from pathlib import Path

import pytest

from codegraph.analyzers.python_analyzer import PythonAnalyzer
from codegraph.graph.schema import EdgeType, NodeType


@pytest.fixture
def analyzer():
    return PythonAnalyzer()


@pytest.fixture
def auth_path(sample_python_repo):
    return sample_python_repo / "auth.py"


@pytest.fixture
def auth_result(analyzer, auth_path):
    content = auth_path.read_text()
    return analyzer.analyze(auth_path, content)


def test_extracts_module_node(auth_result):
    module_nodes = [n for n in auth_result.nodes if n.node_type == NodeType.MODULE]
    assert len(module_nodes) == 1
    assert module_nodes[0].name == "auth"


def test_extracts_functions(auth_result):
    func_names = {n.name for n in auth_result.nodes if n.node_type == NodeType.FUNCTION}
    assert "authenticate" in func_names
    assert "hash_password" in func_names
    assert "_lookup_user" in func_names
    assert "_verify_password" in func_names
    assert "_create_session" in func_names
    assert "invalidate_session" in func_names
    assert "async_authenticate" in func_names


def test_extracts_async_function(auth_result):
    async_funcs = [n for n in auth_result.nodes if n.name == "async_authenticate"]
    assert len(async_funcs) == 1
    assert async_funcs[0].metadata.get("is_async") is True


def test_extracts_docstrings(auth_result):
    authenticate = next(n for n in auth_result.nodes if n.name == "authenticate")
    assert authenticate.docstring is not None
    assert "Authenticate" in authenticate.docstring


def test_extracts_signature(auth_result):
    authenticate = next(n for n in auth_result.nodes if n.name == "authenticate")
    assert authenticate.signature is not None
    assert "username" in authenticate.signature
    assert "password" in authenticate.signature


def test_extracts_imports_as_edges(auth_result):
    import_edges = [e for e in auth_result.edges if e.edge_type == EdgeType.IMPORTS]
    target_modules = {e.metadata.get("target_module", "") for e in import_edges}
    assert any("hashlib" in t for t in target_modules)


def test_extracts_defines_edges(auth_result):
    define_edges = [e for e in auth_result.edges if e.edge_type == EdgeType.DEFINES]
    assert len(define_edges) > 0


def test_extracts_calls_edges(auth_result):
    call_edges = [e for e in auth_result.edges if e.edge_type == EdgeType.CALLS]
    assert len(call_edges) > 0


def test_extracts_variables(sample_python_repo, analyzer):
    content = (sample_python_repo / "auth.py").read_text()
    result = analyzer.analyze(sample_python_repo / "auth.py", content)
    var_names = {n.name for n in result.nodes if n.node_type == NodeType.VARIABLE}
    assert "SECRET_KEY" in var_names
    assert "TOKEN_TTL" in var_names


def test_class_extraction(sample_python_repo, analyzer):
    path = sample_python_repo / "models.py"
    result = analyzer.analyze(path, path.read_text())
    class_names = {n.name for n in result.nodes if n.node_type == NodeType.CLASS}
    assert "User" in class_names
    assert "BaseModel" in class_names
    assert "AdminUser" in class_names


def test_class_methods_extracted(sample_python_repo, analyzer):
    path = sample_python_repo / "models.py"
    result = analyzer.analyze(path, path.read_text())
    method_names = {n.name for n in result.nodes if n.node_type == NodeType.METHOD}
    assert "to_dict" in method_names
    assert "validate" in method_names


def test_inherits_edge(sample_python_repo, analyzer):
    path = sample_python_repo / "models.py"
    result = analyzer.analyze(path, path.read_text())
    inherit_edges = [e for e in result.edges if e.edge_type == EdgeType.INHERITS]
    assert len(inherit_edges) > 0


def test_syntax_error_returns_empty():
    analyzer = PythonAnalyzer()
    result = analyzer.analyze(Path("bad.py"), "def bad(:\n    pass")
    assert result.nodes == []
    assert result.edges == []


def test_node_ids_are_stable(auth_path, analyzer):
    content = auth_path.read_text()
    r1 = analyzer.analyze(auth_path, content)
    r2 = analyzer.analyze(auth_path, content)
    ids1 = {n.id for n in r1.nodes}
    ids2 = {n.id for n in r2.nodes}
    assert ids1 == ids2
