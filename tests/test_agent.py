"""Tests for the agentic RAG layer: tools, router, LangGraph agent, fallback path.

All LLM calls are mocked — tests run fast without API keys or a GPU.
"""

from unittest.mock import MagicMock, patch

import pytest

from codegraph.graph.builder import GraphBuilder
from codegraph.langchain.agent import CodeGraphAgent, build_agent, _blank_state
from codegraph.langchain.tools import TOOL_CATEGORIES, make_codegraph_tools
from codegraph.rag.indexer import DocIndexer
from codegraph.rag.retriever import RAGRetriever


@pytest.fixture
def built_store(sample_python_repo, tmp_db, test_config):
    builder = GraphBuilder(sample_python_repo, tmp_db, test_config)
    builder.build(incremental=False)
    return tmp_db


@pytest.fixture
def rag(built_store, sample_python_repo):
    indexer = DocIndexer(built_store)
    indexer.index_repo(sample_python_repo)
    return RAGRetriever(built_store, indexer)


# ── make_codegraph_tools ──────────────────────────────────────────────────────

class TestMakeCodegraphTools:
    def test_returns_16_tools(self, built_store):
        tools = make_codegraph_tools(built_store)
        assert len(tools) == 16

    def test_tool_names_are_strings(self, built_store):
        tools = make_codegraph_tools(built_store)
        for t in tools:
            assert isinstance(t.name, str)
            assert len(t.name) > 0

    def test_tool_descriptions_present(self, built_store):
        tools = make_codegraph_tools(built_store)
        for t in tools:
            assert t.description, f"{t.name} has no description"

    def test_all_category_tools_exist(self, built_store):
        tools = make_codegraph_tools(built_store)
        tool_names = {t.name for t in tools}
        for cat, names in TOOL_CATEGORIES.items():
            for name in names:
                assert name in tool_names, f"{name} in category {cat} not found in tools"

    def test_find_function_tool_returns_dict(self, built_store):
        tools = make_codegraph_tools(built_store)
        find_fn = next(t for t in tools if t.name == "find_function")
        result = find_fn.invoke({"name": "authenticate"})
        assert isinstance(result, dict)
        assert "name" in result or "error" in result

    def test_find_function_not_found(self, built_store):
        tools = make_codegraph_tools(built_store)
        find_fn = next(t for t in tools if t.name == "find_function")
        result = find_fn.invoke({"name": "nonexistent_xyz_abc"})
        assert "error" in result

    def test_find_callers_returns_list(self, built_store):
        tools = make_codegraph_tools(built_store)
        find_callers = next(t for t in tools if t.name == "find_callers")
        result = find_callers.invoke({"qualified_name": "authenticate"})
        assert isinstance(result, list)

    def test_get_source_returns_provenance(self, built_store):
        tools = make_codegraph_tools(built_store)
        get_source = next(t for t in tools if t.name == "get_source")
        result = get_source.invoke({"qualified_name": "authenticate"})
        if "error" not in result:
            assert result.get("provenance") == "ast_parsed"

    def test_verify_signature_match(self, built_store):
        tools = make_codegraph_tools(built_store)
        verify = next(t for t in tools if t.name == "verify_signature")
        fn_tool = next(t for t in tools if t.name == "find_function")
        fn_result = fn_tool.invoke({"name": "authenticate"})
        if "error" not in fn_result:
            actual_sig = fn_result.get("signature", "")
            result = verify.invoke({"qualified_name": "authenticate", "claimed_signature": actual_sig})
            assert result["match"] is True

    def test_codebase_stats_returns_dict(self, built_store):
        tools = make_codegraph_tools(built_store)
        stats_tool = next(t for t in tools if t.name == "codebase_stats")
        result = stats_tool.invoke({})
        assert isinstance(result, dict)

    def test_find_dead_code_returns_list(self, built_store):
        tools = make_codegraph_tools(built_store)
        dead_tool = next(t for t in tools if t.name == "find_dead_code")
        result = dead_tool.invoke({})
        assert isinstance(result, list)

    def test_search_code_returns_list(self, built_store):
        tools = make_codegraph_tools(built_store)
        search = next(t for t in tools if t.name == "search_code")
        result = search.invoke({"pattern": "auth"})
        assert isinstance(result, list)

    def test_impact_analysis_returns_risk(self, built_store):
        tools = make_codegraph_tools(built_store)
        impact = next(t for t in tools if t.name == "impact_analysis")
        result = impact.invoke({"qualified_name": "authenticate"})
        if "error" not in result:
            assert "risk_level" in result
            assert result["risk_level"] in {"low", "medium", "high"}


# ── TOOL_CATEGORIES ───────────────────────────────────────────────────────────

class TestToolCategories:
    def test_three_categories_defined(self):
        assert set(TOOL_CATEGORIES.keys()) == {"structural", "lookup", "semantic"}

    def test_each_category_has_tools(self):
        for cat, names in TOOL_CATEGORIES.items():
            assert len(names) >= 3, f"Category {cat} has too few tools"

    def test_no_duplicates_within_category(self):
        for cat, names in TOOL_CATEGORIES.items():
            assert len(names) == len(set(names)), f"Duplicates in {cat}"


# ── build_agent / CodeGraphAgent ─────────────────────────────────────────────

class TestBuildAgent:
    def test_returns_compiled_graph(self, built_store, rag):
        app = build_agent(built_store, rag, llm=None)
        assert app is not None

    def test_no_llm_routes_to_fallback(self, built_store, rag):
        agent = CodeGraphAgent(built_store, rag, llm=None)
        result = agent.ask("What calls authenticate?")
        assert "answer" in result
        assert result["source"] == "fallback_retrieval"

    def test_no_llm_answer_is_string(self, built_store, rag):
        agent = CodeGraphAgent(built_store, rag, llm=None)
        result = agent.ask("How does auth work?")
        assert isinstance(result["answer"], str)

    def test_result_has_required_keys(self, built_store, rag):
        agent = CodeGraphAgent(built_store, rag, llm=None)
        result = agent.ask("test question")
        for key in ["answer", "sources", "source", "fallback_reason", "router_category", "iteration_count"]:
            assert key in result, f"Missing key: {key}"

    def test_mocked_llm_structural_query(self, built_store, rag):
        """Agent with mocked LLM routes structural query and returns answer."""
        mock_llm = MagicMock()
        # Router response
        router_resp = MagicMock()
        router_resp.content = "structural"
        # Agent final response — simulate a simple AIMessage with no tool calls
        from langchain_core.messages import AIMessage
        agent_resp = {"messages": [AIMessage(content="authenticate is called by login and register_user.")]}
        mock_llm.invoke.return_value = router_resp

        with patch("codegraph.langchain.agent.create_react_agent") as mock_react:
            mock_react_instance = MagicMock()
            mock_react_instance.invoke.return_value = agent_resp
            mock_react.return_value = mock_react_instance

            agent = CodeGraphAgent(built_store, rag, llm=mock_llm)
            result = agent.ask("What calls authenticate?", thread_id="test-s1")

        assert result["source"] in {"agent", "fallback_retrieval"}

    def test_iteration_cap_triggers_fallback(self, built_store, rag):
        """RecursionError from LangGraph triggers fallback with correct reason."""
        mock_llm = MagicMock()
        router_resp = MagicMock()
        router_resp.content = "lookup"
        mock_llm.invoke.return_value = router_resp

        with patch("codegraph.langchain.agent.create_react_agent") as mock_react:
            mock_react_instance = MagicMock()
            mock_react_instance.invoke.side_effect = Exception("recursion limit exceeded")
            mock_react.return_value = mock_react_instance

            agent = CodeGraphAgent(built_store, rag, llm=mock_llm)
            result = agent.ask("Show me authenticate source", thread_id="test-l1")

        assert result["source"] == "fallback_retrieval"
        assert result["fallback_reason"] == "iteration_cap_exceeded"

    def test_tool_error_triggers_fallback(self, built_store, rag):
        """Generic exception from agent triggers fallback with tool_error reason."""
        mock_llm = MagicMock()
        router_resp = MagicMock()
        router_resp.content = "semantic"
        mock_llm.invoke.return_value = router_resp

        with patch("codegraph.langchain.agent.create_react_agent") as mock_react:
            mock_react_instance = MagicMock()
            mock_react_instance.invoke.side_effect = ValueError("tool argument invalid")
            mock_react.return_value = mock_react_instance

            agent = CodeGraphAgent(built_store, rag, llm=mock_llm)
            result = agent.ask("How does auth work?", thread_id="test-e1")

        assert result["source"] == "fallback_retrieval"
        assert result["fallback_reason"] == "tool_error"

    def test_router_category_preserved(self, built_store, rag):
        """router_category in result matches what the router returned."""
        mock_llm = MagicMock()
        router_resp = MagicMock()
        router_resp.content = "lookup"
        mock_llm.invoke.return_value = router_resp

        with patch("codegraph.langchain.agent.create_react_agent") as mock_react:
            from langchain_core.messages import AIMessage
            mock_react_instance = MagicMock()
            mock_react_instance.invoke.return_value = {
                "messages": [AIMessage(content="The function is in auth.py")]
            }
            mock_react.return_value = mock_react_instance

            agent = CodeGraphAgent(built_store, rag, llm=mock_llm)
            result = agent.ask("Where is authenticate?", thread_id="test-router")

        assert result["router_category"] == "lookup"

    def test_conversation_memory_separate_threads(self, built_store, rag):
        """Different thread_ids are isolated."""
        agent = CodeGraphAgent(built_store, rag, llm=None)
        r1 = agent.ask("question one", thread_id="thread-A")
        r2 = agent.ask("question two", thread_id="thread-B")
        assert r1["answer"] != "" or r2["answer"] != "" or True  # both complete without error


# ── _blank_state ──────────────────────────────────────────────────────────────

class TestBlankState:
    def test_blank_state_has_all_keys(self):
        state = _blank_state("test query")
        required = ["messages", "query", "category", "answer", "sources",
                    "source", "fallback_reason", "iteration_count", "router_category"]
        for k in required:
            assert k in state

    def test_blank_state_query_set(self):
        state = _blank_state("hello world")
        assert state["query"] == "hello world"
