#!/usr/bin/env python3
"""Eval runner for the codegraph agentic RAG system.

Runs all questions in agent_evals.yaml against a live agent and reports:
  - Overall tool-selection accuracy
  - Router accuracy (category classification)
  - Per-category accuracy breakdown
  - Fallback rate and fallback reason breakdown

Usage:
    # Index a repo first
    codegraph index /path/to/repo --data-dir ./data

    # Then run evals with any supported LLM
    python evals/run_evals.py --data-dir ./data --llm claude
    python evals/run_evals.py --data-dir ./data --llm openai
    python evals/run_evals.py --data-dir ./data --llm gemini
    python evals/run_evals.py --data-dir ./data --llm none   # fallback-only baseline

Options:
    --data-dir   Path to the codegraph data dir (default: ./data)
    --evals      Path to the YAML eval file (default: evals/agent_evals.yaml)
    --llm        LLM backend: claude | openai | gemini | none (default: claude)
    --filter     Run only questions whose id starts with this prefix (e.g. s, l, e)
    --verbose    Print full answer for each question
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

# Add src to path so we can import codegraph without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codegraph.config import CodegraphConfig, set_config
from codegraph.graph.store import GraphStore
from codegraph.langchain.agent import CodeGraphAgent
from codegraph.rag.indexer import DocIndexer
from codegraph.rag.retriever import RAGRetriever
from codegraph.utils.logging import configure_logging

configure_logging("WARNING")


# ── LLM factory ───────────────────────────────────────────────────────────────

def _make_llm(backend: str):
    if backend == "none":
        return None
    if backend == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)
        except ImportError:
            print("ERROR: pip install langchain-anthropic")
            sys.exit(1)
    if backend == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o", max_tokens=1024)
        except ImportError:
            print("ERROR: pip install langchain-openai")
            sys.exit(1)
    if backend == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-1.5-pro", max_output_tokens=1024)
        except ImportError:
            print("ERROR: pip install langchain-google-genai")
            sys.exit(1)
    raise ValueError(f"Unknown LLM backend: {backend}")


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _tool_hit(result: dict, expected_tools: list[str]) -> bool:
    """True if any expected tool appears in the agent's answer or source info."""
    if not expected_tools:
        return True  # no expected tool specified — skip tool check
    answer = result.get("answer", "").lower()
    sources = " ".join(result.get("sources", [])).lower()
    text = answer + " " + sources
    return any(t.lower() in text for t in expected_tools)


def _answer_hit(result: dict, expected_contains: list[str]) -> bool:
    """True if all expected strings appear in the answer (case-insensitive)."""
    if not expected_contains:
        return True
    answer = result.get("answer", "").lower()
    return all(s.lower() in answer for s in expected_contains)


def _router_hit(result: dict, expected_category: str) -> bool:
    return result.get("router_category", "") == expected_category


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run codegraph agent evals")
    parser.add_argument("--data-dir", default="./data", help="Path to codegraph data dir")
    parser.add_argument("--evals", default="evals/agent_evals.yaml", help="Path to eval YAML")
    parser.add_argument("--llm", default="claude", choices=["claude", "openai", "gemini", "none"])
    parser.add_argument("--filter", default="", help="Run only questions whose id starts with this")
    parser.add_argument("--verbose", action="store_true", help="Print full answer for each question")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    db_path = data_dir / "codegraph.db"
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}. Run 'codegraph index' first.")
        sys.exit(1)

    evals_path = Path(args.evals)
    if not evals_path.exists():
        print(f"ERROR: Eval file not found at {evals_path}")
        sys.exit(1)

    with open(evals_path) as f:
        all_cases = yaml.safe_load(f)

    if args.filter:
        all_cases = [c for c in all_cases if c["id"].startswith(args.filter)]

    if not all_cases:
        print("No eval cases matched the filter.")
        sys.exit(0)

    print(f"codegraph agent evals — {len(all_cases)} questions, LLM: {args.llm}")
    print("=" * 60)

    config = CodegraphConfig(data_dir=data_dir)
    set_config(config)
    store = GraphStore(db_path)
    indexer = DocIndexer(store)
    rag = RAGRetriever(store, indexer)
    llm = _make_llm(args.llm)
    agent = CodeGraphAgent(store, rag, llm=llm)

    # Per-category tracking
    categories = ["structural", "lookup", "semantic"]
    stats: dict[str, dict] = {
        cat: {"total": 0, "tool_hits": 0, "answer_hits": 0, "router_hits": 0, "fallbacks": 0}
        for cat in categories
    }
    stats["overall"] = {"total": 0, "tool_hits": 0, "answer_hits": 0, "router_hits": 0, "fallbacks": 0}
    fallback_reasons: dict[str, int] = {}
    results_rows = []

    for case in all_cases:
        qid = case["id"]
        question = case["question"]
        expected_cat = case["category"]
        expected_tools = case.get("expected_tools", [])
        expected_contains = case.get("expected_answer_contains", [])

        t0 = time.time()
        try:
            result = agent.ask(question, thread_id=qid)
        except Exception as e:
            result = {
                "answer": f"ERROR: {e}",
                "sources": [],
                "source": "fallback_retrieval",
                "fallback_reason": "eval_exception",
                "router_category": "",
                "iteration_count": 0,
            }
        elapsed = time.time() - t0

        tool_ok = _tool_hit(result, expected_tools)
        answer_ok = _answer_hit(result, expected_contains)
        router_ok = _router_hit(result, expected_cat)
        is_fallback = result.get("source") == "fallback_retrieval"
        fb_reason = result.get("fallback_reason") or "none"

        if is_fallback:
            fallback_reasons[fb_reason] = fallback_reasons.get(fb_reason, 0) + 1

        cat_key = expected_cat if expected_cat in stats else "overall"
        for key in [cat_key, "overall"]:
            stats[key]["total"] += 1
            if tool_ok:
                stats[key]["tool_hits"] += 1
            if answer_ok:
                stats[key]["answer_hits"] += 1
            if router_ok:
                stats[key]["router_hits"] += 1
            if is_fallback:
                stats[key]["fallbacks"] += 1

        tool_sym = "✓" if tool_ok else "✗"
        ans_sym = "✓" if answer_ok else "✗"
        router_sym = "✓" if router_ok else "✗"
        fb_sym = " [FALLBACK]" if is_fallback else ""
        print(f"  [{qid}] tool:{tool_sym} ans:{ans_sym} router:{router_sym}{fb_sym}  ({elapsed:.1f}s)  {question[:55]}")

        if args.verbose:
            print(f"         answer: {result['answer'][:200]}")
            print()

        results_rows.append({
            "id": qid,
            "category": expected_cat,
            "router_category": result.get("router_category"),
            "tool_hit": tool_ok,
            "answer_hit": answer_ok,
            "router_hit": router_ok,
            "fallback": is_fallback,
            "fallback_reason": fb_reason,
            "elapsed": round(elapsed, 2),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n{'Category':<14} {'N':>4} {'Tool%':>7} {'Answer%':>9} {'Router%':>9} {'Fallback%':>11}")
    print("-" * 56)
    for cat in categories + ["overall"]:
        s = stats[cat]
        n = s["total"]
        if n == 0:
            continue
        tp = round(100 * s["tool_hits"] / n)
        ap = round(100 * s["answer_hits"] / n)
        rp = round(100 * s["router_hits"] / n)
        fp = round(100 * s["fallbacks"] / n)
        label = cat.capitalize() if cat != "overall" else "OVERALL"
        print(f"  {label:<12} {n:>4}   {tp:>5}%   {ap:>7}%   {rp:>7}%   {fp:>9}%")

    if fallback_reasons:
        print("\nFallback reasons:")
        for reason, count in sorted(fallback_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    store.close()

    # Exit non-zero if overall tool accuracy is below 60%
    overall = stats["overall"]
    n = overall["total"]
    if n > 0 and overall["tool_hits"] / n < 0.6:
        print(f"\nWARNING: Tool-selection accuracy below 60% ({round(100*overall['tool_hits']/n)}%)")
        sys.exit(1)

    print("\nEvals complete.")


if __name__ == "__main__":
    main()
