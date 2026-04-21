"""Tests for anti-hallucination tools: get_source, verify_signature, token_savings_estimate."""
import pytest

from codegraph.graph.builder import GraphBuilder
from codegraph.mcp_server.server import (
    get_source,
    verify_signature,
    token_savings_estimate,
    init_mcp,
)
from codegraph.rag.indexer import DocIndexer
from codegraph.rag.retriever import RAGRetriever
from codegraph.graphql.schema import build_schema


@pytest.fixture(autouse=True)
def mcp_setup(sample_python_repo, tmp_db, test_config):
    builder = GraphBuilder(sample_python_repo, tmp_db, test_config)
    builder.build(incremental=False)
    indexer = DocIndexer(tmp_db)
    retriever = RAGRetriever(tmp_db, indexer)
    init_mcp(tmp_db, retriever, test_config)


# ── get_source ────────────────────────────────────────────────────────────────

def test_get_source_returns_code():
    result = get_source("authenticate")
    assert "error" not in result
    assert "source" in result
    assert "def authenticate" in result["source"]


def test_get_source_has_provenance():
    result = get_source("authenticate")
    assert result.get("provenance") == "ast_parsed"


def test_get_source_has_line_numbers():
    result = get_source("authenticate")
    assert "start_line" in result
    assert "end_line" in result
    assert result["start_line"] > 0
    assert result["end_line"] >= result["start_line"]


def test_get_source_not_found():
    result = get_source("completely_nonexistent_xyz_func")
    assert "error" in result


def test_get_source_returns_exact_signature_line():
    result = get_source("authenticate")
    # The first line of the source should contain the def
    first_line = result["source"].splitlines()[0]
    assert "authenticate" in first_line


def test_get_source_graphql(sample_python_repo, tmp_db, test_config):
    indexer = DocIndexer(tmp_db)
    retriever = RAGRetriever(tmp_db, indexer)
    schema = build_schema(tmp_db, retriever)
    result = schema.execute_sync("""
        query {
            getSource(qualifiedName: "auth.authenticate")
        }
    """)
    assert result.errors is None
    src = result.data["getSource"]
    assert src is not None
    assert "authenticate" in src


# ── verify_signature ──────────────────────────────────────────────────────────

def test_verify_signature_correct():
    # Get the actual signature first
    get_source("authenticate")
    # A clearly wrong signature should fail
    result = verify_signature("authenticate", "(x)")
    assert result["match"] is False
    assert "actual" in result
    assert "verdict" in result


def test_verify_signature_returns_actual():
    result = verify_signature("authenticate", "(username, password)")
    assert "actual" in result
    assert result["actual"] is not None
    # The actual signature should mention username and password
    assert "username" in result["actual"]
    assert "password" in result["actual"]


def test_verify_signature_exact_match():
    # First get the real signature
    result = get_source("authenticate")
    assert "error" not in result
    # Now get the parsed signature from find_function
    from codegraph.mcp_server.server import find_function
    fn = find_function("authenticate")
    actual = fn.get("signature", "")
    # Verifying with the actual signature should match
    verify = verify_signature("authenticate", actual)
    assert verify["match"] is True
    assert verify["verdict"] == "CORRECT"


def test_verify_signature_not_found():
    result = verify_signature("nonexistent_xyz", "(x)")
    assert "error" in result


def test_verify_signature_verdict_on_wrong():
    result = verify_signature("authenticate", "(totally_wrong_args)")
    assert "WRONG" in result["verdict"]


# ── token_savings_estimate ────────────────────────────────────────────────────

def test_token_savings_has_required_fields():
    result = token_savings_estimate("authenticate")
    assert "error" not in result
    assert "naive_approach_tokens" in result
    assert "codegraph_tokens" in result
    assert "tokens_saved" in result
    assert "savings_percent" in result
    assert "explanation" in result


def test_token_savings_naive_greater_than_graph():
    result = token_savings_estimate("authenticate")
    # Reading a file should cost more tokens than a targeted query
    assert result["naive_approach_tokens"] >= result["codegraph_tokens"]


def test_token_savings_percent_valid_range():
    result = token_savings_estimate("authenticate")
    pct = result["savings_percent"]
    assert 0 <= pct <= 100


def test_token_savings_explanation_is_human_readable():
    result = token_savings_estimate("authenticate")
    explanation = result["explanation"]
    assert "token" in explanation.lower()
    assert "%" in explanation


def test_token_savings_not_found():
    result = token_savings_estimate("nonexistent_xyz_func")
    assert "error" in result
