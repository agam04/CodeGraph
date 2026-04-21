"""Demo: CodebaseQA with LangChain + codegraph.

Run after indexing a repo:
    codegraph index /path/to/repo --data-dir ./data
    python examples/langchain_qa_demo.py

Requires one of:
    pip install langchain-anthropic       # for Claude
    pip install langchain-huggingface     # for local HuggingFace models
"""

from pathlib import Path

from codegraph.config import CodegraphConfig, set_config
from codegraph.graph.store import GraphStore
from codegraph.langchain import build_codebase_qa
from codegraph.rag.indexer import DocIndexer
from codegraph.rag.retriever import RAGRetriever
from codegraph.utils.logging import configure_logging

configure_logging("WARNING")

config = CodegraphConfig(data_dir=Path("./data"))
set_config(config)
store = GraphStore(config.data_dir / "codegraph.db")
indexer = DocIndexer(store)
rag = RAGRetriever(store, indexer)

# ── Option A: Claude (best quality) ──────────────────────────────────────────
try:
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=512)
    print("Using Claude claude-sonnet-4-6")
except ImportError:
    llm = None
    print("langchain-anthropic not installed. Run: pip install langchain-anthropic")

# ── Option B: local HuggingFace model ────────────────────────────────────────
# from langchain_huggingface import HuggingFacePipeline
# llm = HuggingFacePipeline.from_model_id(
#     "microsoft/phi-2",
#     task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 256},
# )

qa = build_codebase_qa(store, rag, llm=llm, k=5)

questions = [
    "How does authentication work in this codebase?",
    "What functions call authenticate() and where are they?",
    "What would break if I changed the User model?",
    "Are there any functions with no documentation?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    if llm is None:
        # Show retrieval even without LLM
        docs = qa.retriever.invoke(q)
        print(f"Retrieved {len(docs)} relevant docs:")
        for d in docs[:2]:
            print(f"  [{d.metadata.get('result_kind')}] {d.metadata.get('source')}")
            print(f"  {d.page_content[:120]}...")
    else:
        result = qa.ask(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'][:3])}")

store.close()
