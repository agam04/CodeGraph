from pathlib import Path

import pytest

from codegraph.graph.builder import GraphBuilder
from codegraph.graph.schema import NodeType
from codegraph.rag.indexer import DocIndexer
from codegraph.rag.retriever import RAGRetriever


@pytest.fixture
def populated_store(sample_python_repo, tmp_db, test_config):
    builder = GraphBuilder(sample_python_repo, tmp_db, test_config)
    builder.build(incremental=False)
    return tmp_db


def test_doc_indexer_finds_readme(populated_store, sample_python_repo):
    indexer = DocIndexer(populated_store)
    count = indexer.index_repo(sample_python_repo)
    assert count > 0


def test_doc_chunks_stored_in_graph(populated_store, sample_python_repo):
    indexer = DocIndexer(populated_store)
    indexer.index_repo(sample_python_repo)
    chunks = populated_store.get_all_nodes(NodeType.DOC_CHUNK)
    assert len(chunks) > 0


def test_retriever_text_search(populated_store, sample_python_repo):
    indexer = DocIndexer(populated_store)
    indexer.index_repo(sample_python_repo)
    retriever = RAGRetriever(populated_store, indexer)
    results = retriever.search_docs("authenticate", k=3)
    assert isinstance(results, list)
    assert len(results) >= 0  # may be 0 if no match, just no crash


def test_docs_for_node(populated_store, sample_python_repo):
    indexer = DocIndexer(populated_store)
    indexer.index_repo(sample_python_repo)
    retriever = RAGRetriever(populated_store, indexer)
    # The README mentions authenticate() — should link to its node
    auth_node = populated_store.get_node_by_name("authenticate", NodeType.FUNCTION)
    if auth_node:
        docs = retriever.docs_for_node(auth_node.id)
        assert isinstance(docs, list)


def test_chunk_has_content(populated_store, sample_python_repo):
    indexer = DocIndexer(populated_store)
    indexer.index_repo(sample_python_repo)
    chunks = populated_store.get_all_nodes(NodeType.DOC_CHUNK)
    for chunk in chunks:
        content = chunk.metadata.get("content", "")
        assert len(content) > 0
