"""CodebaseQA — a LangChain chain that answers questions about your codebase.

Grounds every answer in the codegraph knowledge base so the LLM can't
hallucinate function signatures, import paths, or architecture details.

Usage:
    # With Claude (recommended)
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-sonnet-4-6")

    # Or with any local HuggingFace model
    from langchain_huggingface import HuggingFacePipeline
    llm = HuggingFacePipeline.from_model_id("microsoft/phi-2", task="text-generation")

    qa = build_codebase_qa(store, rag_retriever, llm=llm)
    result = qa.ask("How does authentication work?")
    print(result["answer"])
    print(result["sources"])
"""

from typing import Any

from codegraph.graph.store import GraphStore
from codegraph.langchain.retriever import CodeGraphRetriever
from codegraph.rag.retriever import RAGRetriever
from codegraph.utils.logging import get_logger

log = get_logger(__name__)

_SYSTEM_PROMPT = """You are a code intelligence assistant with access to a structured graph of the codebase.
The context below contains relevant code symbols and documentation retrieved from the codebase graph.
Answer the question using ONLY the provided context — do not rely on training memory for specific function
signatures, file paths, or implementation details. If the context doesn't contain enough information, say so.
Always cite the source file and line number when referencing specific code.

Context:
{context}
"""

_QUESTION_TEMPLATE = """Question: {question}

Answer (cite sources with file:line):"""


class CodebaseQA:
    """Question-answering chain grounded in the codegraph knowledge base.

    Answers questions about the codebase using retrieval-augmented generation.
    The retriever pulls relevant code symbols and documentation; the LLM
    synthesises an answer citing exact sources.
    """

    def __init__(
        self,
        retriever: CodeGraphRetriever,
        llm: Any,
        k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def ask(self, question: str) -> dict:
        """Ask a question about the codebase.

        Returns:
            dict with keys: answer (str), sources (list[str]), context_used (str)
        """
        docs = self.retriever.invoke(question)
        if not docs:
            return {
                "answer": "No relevant code or documentation found for this question.",
                "sources": [],
                "context_used": "",
            }

        context = self._format_context(docs)
        prompt = _SYSTEM_PROMPT.format(context=context) + _QUESTION_TEMPLATE.format(question=question)

        try:
            response = self._call_llm(prompt)
        except Exception as e:
            log.error("llm_call_failed", error=str(e))
            return {"answer": f"LLM call failed: {e}", "sources": [], "context_used": context}

        sources = list({
            d.metadata.get("source", "") for d in docs if d.metadata.get("source")
        })

        return {
            "answer": response,
            "sources": sources,
            "context_used": context,
            "docs_retrieved": len(docs),
        }

    def _call_llm(self, prompt: str) -> str:
        # Support both chat models (invoke) and completion models (predict/__call__)
        if hasattr(self.llm, "invoke"):
            result = self.llm.invoke(prompt)
            return result.content if hasattr(result, "content") else str(result)
        if hasattr(self.llm, "predict"):
            return self.llm.predict(prompt)
        return str(self.llm(prompt))

    @staticmethod
    def _format_context(docs: list) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            kind = doc.metadata.get("result_kind", "")
            parts.append(f"[{i}] ({kind}) {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


def build_codebase_qa(
    store: GraphStore,
    rag_retriever: RAGRetriever,
    llm: Any = None,
    k: int = 5,
) -> CodebaseQA:
    """Build a CodebaseQA instance ready to answer questions.

    Args:
        store: The graph store.
        rag_retriever: The RAG retriever (hybrid BM25 + FAISS).
        llm: Any LangChain-compatible LLM. If None, a warning is logged.
        k: Number of documents to retrieve per query.

    Returns:
        A CodebaseQA instance.

    Example:
        from langchain_anthropic import ChatAnthropic
        qa = build_codebase_qa(store, retriever, llm=ChatAnthropic(model="claude-sonnet-4-6"))
        result = qa.ask("What functions handle user login?")
    """
    if llm is None:
        log.warning("no_llm_provided", hint="Pass an LLM via the llm= argument to get answers")

    retriever = CodeGraphRetriever(store=store, rag=rag_retriever, k=k)
    return CodebaseQA(retriever=retriever, llm=llm, k=k)
