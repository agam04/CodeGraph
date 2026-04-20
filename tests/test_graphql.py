import pytest

from codegraph.graph.builder import GraphBuilder
from codegraph.graphql.schema import build_schema
from codegraph.rag.indexer import DocIndexer
from codegraph.rag.retriever import RAGRetriever


@pytest.fixture
def populated_store(sample_python_repo, tmp_db, test_config):
    builder = GraphBuilder(sample_python_repo, tmp_db, test_config)
    builder.build(incremental=False)
    return tmp_db


@pytest.fixture
def gql_schema(populated_store):
    indexer = DocIndexer(populated_store)
    retriever = RAGRetriever(populated_store, indexer)
    return build_schema(populated_store, retriever)


def _exec(schema, query: str):
    return schema.execute_sync(query)


def test_query_function_by_name(gql_schema):
    result = _exec(gql_schema, """
        query {
            function(name: "authenticate") {
                name
                qualifiedName
                filePath
                signature
                docstring
            }
        }
    """)
    assert result.errors is None
    fn = result.data["function"]
    assert fn is not None
    assert fn["name"] == "authenticate"
    assert fn["signature"] is not None
    assert "username" in fn["signature"]


def test_query_class_by_name(gql_schema):
    result = _exec(gql_schema, """
        query {
            class_(name: "User") {
                name
                qualifiedName
                docstring
            }
        }
    """)
    assert result.errors is None
    cls = result.data["class_"]
    assert cls is not None
    assert cls["name"] == "User"


def test_search_functions(gql_schema):
    result = _exec(gql_schema, """
        query {
            searchFunctions(pattern: "auth", limit: 5) {
                name
                filePath
            }
        }
    """)
    assert result.errors is None
    funcs = result.data["searchFunctions"]
    assert len(funcs) > 0


def test_callers_query(gql_schema):
    result = _exec(gql_schema, """
        query {
            callers(qualifiedName: "auth.authenticate", depth: 1) {
                name
                filePath
            }
        }
    """)
    assert result.errors is None
    callers = result.data["callers"]
    assert isinstance(callers, list)


def test_context_for_query(gql_schema):
    result = _exec(gql_schema, """
        query {
            contextFor(qualifiedName: "auth.authenticate", depth: 2) {
                summary
                estimatedTokens
                relatedNodes {
                    name
                    nodeType
                }
                edges {
                    edgeType
                }
            }
        }
    """)
    assert result.errors is None
    ctx = result.data["contextFor"]
    assert ctx["summary"] != ""
    assert ctx["estimatedTokens"] > 0


def test_stats_query(gql_schema):
    result = _exec(gql_schema, """
        query {
            stats {
                totalFiles
                totalFunctions
                totalClasses
                totalNodes
                totalEdges
                languages
            }
        }
    """)
    assert result.errors is None
    s = result.data["stats"]
    assert s["totalFiles"] >= 4
    assert s["totalFunctions"] > 0
    assert "python" in s["languages"]


def test_function_not_found_returns_null(gql_schema):
    result = _exec(gql_schema, """
        query {
            function(name: "nonexistent_xyz_abc") {
                name
            }
        }
    """)
    assert result.errors is None
    assert result.data["function"] is None


def test_search_classes(gql_schema):
    result = _exec(gql_schema, """
        query {
            searchClasses(pattern: "User", limit: 5) {
                name
            }
        }
    """)
    assert result.errors is None
    names = {c["name"] for c in result.data["searchClasses"]}
    assert "User" in names
