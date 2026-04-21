

from codegraph.graph.builder import GraphBuilder
from codegraph.graph.schema import NodeType


def test_build_sample_python_repo(sample_python_repo, tmp_db, test_config):
    builder = GraphBuilder(sample_python_repo, tmp_db, test_config)
    stats = builder.build(incremental=False)

    assert stats.files_indexed >= 4  # models, auth, api, utils
    assert stats.nodes_created > 0
    assert stats.edges_created > 0
    assert len(stats.errors) == 0


def test_extracts_authenticate_function(sample_python_repo, tmp_db, test_config):
    builder = GraphBuilder(sample_python_repo, tmp_db, test_config)
    builder.build(incremental=False)

    node = tmp_db.get_node_by_name("authenticate", NodeType.FUNCTION)
    assert node is not None
    assert "auth.py" in node.file_path


def test_extracts_class(sample_python_repo, tmp_db, test_config):
    builder = GraphBuilder(sample_python_repo, tmp_db, test_config)
    builder.build(incremental=False)

    user_class = tmp_db.get_node_by_name("User", NodeType.CLASS)
    assert user_class is not None


def test_incremental_skips_unchanged(sample_python_repo, tmp_db, test_config):
    builder = GraphBuilder(sample_python_repo, tmp_db, test_config)
    stats1 = builder.build(incremental=False)

    # Second build: all files unchanged
    stats2 = builder.build(incremental=True)
    assert stats2.files_skipped == stats1.files_indexed
    assert stats2.files_indexed == 0


def test_stats_after_build(sample_python_repo, tmp_db, test_config):
    builder = GraphBuilder(sample_python_repo, tmp_db, test_config)
    builder.build(incremental=False)
    s = tmp_db.stats()
    assert s["total_files"] >= 4
    assert s["type_counts"].get("function", 0) > 0
    assert "python" in s["languages"]


def test_excludes_pycache(sample_python_repo, tmp_db, test_config, tmp_path):
    # Create a fake __pycache__ file in a temp copy
    import shutil
    repo_copy = tmp_path / "repo"
    shutil.copytree(sample_python_repo, repo_copy)
    pycache = repo_copy / "__pycache__"
    pycache.mkdir()
    (pycache / "cached.py").write_text("x = 1")

    builder = GraphBuilder(repo_copy, tmp_db, test_config)
    builder.build(incremental=False)

    # Cached file should not be indexed
    files = tmp_db.get_all_files()
    paths = [f["path"] for f in files]
    assert not any("__pycache__" in p for p in paths)
