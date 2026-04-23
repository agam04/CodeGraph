"""Agentic RAG over the codegraph knowledge graph.

Architecture:
    router → react_agent (max 5 iterations) → [fallback on failure]

The router classifies each query into structural / lookup / semantic and
narrows the active toolset to 6 tools instead of all 16. This prevents
context dilution while keeping the agent focused on the right tool type.

If the agent hits the iteration cap, throws, or produces no final answer,
the graph routes to a static hybrid-retrieval fallback and logs the reason.
Conversation history is maintained via LangGraph MemorySaver checkpoints.

Usage:
    from codegraph.langchain.agent import build_agent

    agent = build_agent(store, rag_retriever, llm=ChatAnthropic(...))
    result = agent.invoke({"query": "What calls authenticate()?"}, config={"configurable": {"thread_id": "1"}})
    print(result["answer"])
    print(result["source"])          # "agent" or "fallback_retrieval"
    print(result["fallback_reason"]) # None | "iteration_cap_exceeded" | "tool_error" | "agent_no_final_answer"
"""

from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from codegraph.graph.store import GraphStore
from codegraph.langchain.tools import TOOL_CATEGORIES, make_codegraph_tools
from codegraph.rag.retriever import RAGRetriever
from codegraph.utils.logging import get_logger

log = get_logger(__name__)

_MAX_ITERATIONS = 5

_ROUTER_PROMPT = """\
You are a query router for a code knowledge graph tool.
Classify the user's question into exactly one category:

  structural — questions about call relationships, who calls what, impact of changes,
               dead code, file dependencies, call graphs
  lookup     — questions about a specific function or class: its source code,
               signature, location, diagram, or token cost
  semantic   — open-ended questions, concept searches, "how does X work",
               finding functions by behaviour, general codebase stats

Reply with ONLY one word: structural, lookup, or semantic.
"""

_AGENT_SYSTEM = """\
You are an expert code analyst with access to a structured code knowledge graph.
The graph contains AST-verified facts about every function, class, and import in the codebase.

Rules:
1. NEVER guess a function signature, file path, or call relationship from memory.
   Always call a tool to get the verified answer.
2. Use the most specific tool available. If you know the function name, call
   find_function or get_source — not search_docs.
3. After at most {max_iterations} tool calls, give a final answer citing tool results.
4. If a tool returns an error, try search_code to find the correct name, then retry.
""".format(max_iterations=_MAX_ITERATIONS)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    category: str                  # structural | lookup | semantic
    answer: str
    sources: list[str]
    source: str                    # "agent" | "fallback_retrieval"
    fallback_reason: str | None    # None | iteration_cap_exceeded | tool_error | agent_no_final_answer
    iteration_count: int
    router_category: str           # preserved for eval tracking


# ── Node implementations ──────────────────────────────────────────────────────

def _make_router_node(llm: Any):
    def router_node(state: AgentState) -> dict:
        query = state["query"]
        try:
            response = llm.invoke([
                SystemMessage(content=_ROUTER_PROMPT),
                HumanMessage(content=query),
            ])
            raw = response.content.strip().lower()
            category = raw if raw in {"structural", "lookup", "semantic"} else "semantic"
        except Exception as e:
            log.warning("router_failed", error=str(e))
            category = "semantic"

        log.info("router_classified", query=query[:60], category=category)
        return {
            "category": category,
            "router_category": category,
            "messages": [HumanMessage(content=query)],
        }
    return router_node


def _make_agent_node(llm: Any, all_tools: list):
    tool_by_name = {t.name: t for t in all_tools}

    def agent_node(state: AgentState) -> dict:
        category = state.get("category", "semantic")
        active_names = TOOL_CATEGORIES.get(category, TOOL_CATEGORIES["semantic"])
        active_tools = [tool_by_name[n] for n in active_names if n in tool_by_name]

        inner_agent = create_react_agent(
            llm,
            tools=active_tools,
            prompt=ChatPromptTemplate.from_messages([
                ("system", _AGENT_SYSTEM),
                ("placeholder", "{messages}"),
            ]),
        )

        try:
            result = inner_agent.invoke(
                {"messages": state["messages"]},
                config={"recursion_limit": _MAX_ITERATIONS * 2 + 2},
            )
            messages = result.get("messages", [])
            final = next(
                (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
                None,
            )

            if final is None:
                return {
                    "messages": messages,
                    "answer": "",
                    "source": "fallback_retrieval",
                    "fallback_reason": "agent_no_final_answer",
                    "iteration_count": len([m for m in messages if hasattr(m, "tool_calls") and m.tool_calls]),
                }

            sources = list({
                m.content if isinstance(m.content, str) else json.dumps(m.content)
                for m in messages
                if hasattr(m, "name") and m.name  # ToolMessage
            })[:5]

            return {
                "messages": messages,
                "answer": final,
                "sources": sources,
                "source": "agent",
                "fallback_reason": None,
                "iteration_count": len([m for m in messages if hasattr(m, "tool_calls") and m.tool_calls]),
            }

        except Exception as e:
            err = str(e)
            log.warning("agent_failed", error=err)
            reason = "iteration_cap_exceeded" if "recursion" in err.lower() else "tool_error"
            return {
                "answer": "",
                "source": "fallback_retrieval",
                "fallback_reason": reason,
                "iteration_count": _MAX_ITERATIONS,
            }

    return agent_node


def _make_fallback_node(rag: RAGRetriever | None):
    def fallback_node(state: AgentState) -> dict:
        query = state["query"]
        reason = state.get("fallback_reason", "unknown")
        log.info("fallback_triggered", reason=reason, query=query[:60])

        if rag is None:
            return {
                "answer": f"Agent could not answer (reason: {reason}). No fallback retriever available.",
                "sources": [],
                "source": "fallback_retrieval",
            }

        docs = rag.search_docs(query, k=5)
        if not docs:
            return {
                "answer": f"Agent could not answer (reason: {reason}) and no relevant docs found.",
                "sources": [],
                "source": "fallback_retrieval",
            }

        context = "\n\n".join(
            f"[{d.get('result_kind', 'doc')}] {d.get('name', '')} — {d.get('snippet', d.get('content', ''))[:300]}"
            for d in docs
        )
        sources = list({d.get("source", "") for d in docs if d.get("source")})
        return {
            "answer": f"(Fallback retrieval — agent {reason})\n\nRelevant context:\n{context}",
            "sources": sources,
            "source": "fallback_retrieval",
        }

    return fallback_node


def _route_after_agent(state: AgentState) -> str:
    if state.get("source") == "fallback_retrieval":
        return "fallback"
    return END


# ── Public builder ────────────────────────────────────────────────────────────

def build_agent(
    store: GraphStore,
    rag: RAGRetriever | None = None,
    llm: Any = None,
) -> Any:
    """Build and return a compiled LangGraph agent.

    Args:
        store:  Indexed GraphStore to query.
        rag:    RAGRetriever for semantic fallback. Optional but recommended.
        llm:    Any LangChain chat model (ChatAnthropic, ChatOpenAI, ChatGoogleGenerativeAI…).
                If None, every query routes immediately to fallback.

    Returns:
        A compiled LangGraph app. Call with:
            result = agent.invoke(
                {"query": "...", "category": "", "answer": "", "sources": [],
                 "source": "", "fallback_reason": None, "iteration_count": 0,
                 "router_category": ""},
                config={"configurable": {"thread_id": "session-1"}},
            )
    """
    all_tools = make_codegraph_tools(store, rag)
    memory = MemorySaver()

    workflow = StateGraph(AgentState)

    if llm is None:
        # No LLM — always fall back
        workflow.add_node("fallback", _make_fallback_node(rag))
        workflow.set_entry_point("fallback")
        workflow.add_edge("fallback", END)
    else:
        workflow.add_node("router", _make_router_node(llm))
        workflow.add_node("agent", _make_agent_node(llm, all_tools))
        workflow.add_node("fallback", _make_fallback_node(rag))

        workflow.set_entry_point("router")
        workflow.add_edge("router", "agent")
        workflow.add_conditional_edges("agent", _route_after_agent, {"fallback": "fallback", END: END})
        workflow.add_edge("fallback", END)

    return workflow.compile(checkpointer=memory)


def _blank_state(query: str) -> AgentState:
    """Return a minimal valid initial state for invoke()."""
    return AgentState(
        messages=[],
        query=query,
        category="",
        answer="",
        sources=[],
        source="",
        fallback_reason=None,
        iteration_count=0,
        router_category="",
    )


class CodeGraphAgent:
    """High-level wrapper around the LangGraph agent with a simple .ask() interface.

    Maintains per-session conversation history via thread_id.

    Usage:
        agent = CodeGraphAgent(store, rag, llm=ChatAnthropic(...))
        result = agent.ask("What calls authenticate()?")
        result = agent.ask("And what is the risk of changing it?", thread_id="session-1")
    """

    def __init__(self, store: GraphStore, rag: RAGRetriever | None = None, llm: Any = None):
        self._app = build_agent(store, rag, llm)

    def ask(self, question: str, thread_id: str = "default") -> dict:
        """Ask a question. Returns answer, sources, source, fallback_reason, router_category."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self._app.invoke(_blank_state(question), config=config)
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "source": result.get("source", ""),
            "fallback_reason": result.get("fallback_reason"),
            "router_category": result.get("router_category", ""),
            "iteration_count": result.get("iteration_count", 0),
        }
