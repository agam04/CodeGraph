from pathlib import Path

import pytest

from codegraph.config import CodegraphConfig, set_config
from codegraph.graph.store import GraphStore


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    store = GraphStore(db_path)
    yield store
    store.close()


@pytest.fixture
def sample_python_repo():
    return Path(__file__).parent / "fixtures" / "sample_python_repo"


@pytest.fixture
def sample_js_repo():
    return Path(__file__).parent / "fixtures" / "sample_js_repo"


@pytest.fixture
def test_config(tmp_path):
    config = CodegraphConfig(data_dir=tmp_path / "data", repo_path=None)
    set_config(config)
    return config
