# codegraph — Agent & Developer Instructions

This file is read automatically by AI coding agents (Claude Code, OpenAI Codex, Gemini CLI)
and any developer working in this repo. Follow these instructions exactly.

---

## What this project does

codegraph indexes a codebase (Python, JS, TS) into a persistent graph backed by SQLite, then
exposes it through a GraphQL API and an MCP server with 16 tools. Core purpose: give AI agents
AST-verified ground truth so they stop hallucinating function signatures, call relationships,
and import paths — and stop wasting context tokens reading files they could query instead.

---

## Project layout

```
src/codegraph/
  config.py            — CodegraphConfig (Pydantic Settings, CODEGRAPH_ env prefix)
  main.py              — CLI entry point (click): index, serve, enrich, stats
  graph/
    schema.py          — NodeType, EdgeType, Node, Edge, BuildStats dataclasses
    store.py           — GraphStore: SQLite persistence, all read/write methods
    builder.py         — GraphBuilder.build(): files → analyzers → store → cross-file resolution
    queries.py         — Graph traversal: callers, callees, subgraph, impact, dead code, Mermaid
  analyzers/
    python_analyzer.py — stdlib ast; extracts functions, classes, calls, imports, signatures
    js_analyzer.py     — tree-sitter; JS + TS support
  graphql/
    schema.py          — Strawberry GraphQL schema, all query resolvers
  mcp_server/
    server.py          — FastMCP server, 16 tools
  rag/
    embedders.py       — CodeAwareEmbedder: CodeBERT (code) + MiniLM (text), dual FAISS
    indexer.py         — DocIndexer: builds doc-FAISS and code-FAISS indices
    retriever.py       — RAGRetriever: BM25 + doc-FAISS + code-FAISS merged via RRF
    docgen.py          — DocstringGenerator: CodeT5 auto-docstring generation
  langchain/
    retriever.py       — CodeGraphRetriever(BaseRetriever) — works with any LangChain LLM
    qa_chain.py        — CodebaseQA: grounded QA with anti-hallucination system prompt
tests/                 — pytest suite, 123 tests, all mocking heavy model downloads
examples/              — runnable demos
```

---

## How to run

```bash
# Install
pip install -e ".[dev]"          # dev + test dependencies
pip install -e ".[langchain]"    # optional LangChain extras

# Index a repo
codegraph index /path/to/repo --data-dir ./data

# Start GraphQL server (playground at http://localhost:8000/graphql)
codegraph serve --data-dir ./data

# Start MCP server (stdio — for Claude Desktop, Cursor, etc.)
codegraph serve --mcp-stdio --repo /path/to/repo

# Generate docstrings for undocumented functions (CodeT5)
codegraph enrich --data-dir ./data

# Stats
codegraph stats --data-dir ./data
```

---

## Running tests

```bash
pytest                            # all 123 tests with coverage
pytest tests/test_python_analyzer.py -v
pytest -k "test_impact"           # filter by name
```

All tests mock HuggingFace model downloads — run fast without a GPU.

---

## MCP tools (16 total) — use these instead of reading files

| Tool | When to use |
|------|-------------|
| `get_source(name)` | Exact implementation — never guess from memory |
| `verify_signature(name, claimed_sig)` | Confirm a signature before citing it |
| `token_savings_estimate(name)` | Tokens saved vs naive file reading |
| `find_function(name)` | Location, signature, callers, callees |
| `find_class(name)` | Methods, base classes, subclasses |
| `get_context(name, depth)` | Token-budgeted subgraph ranked by fan-in |
| `impact_analysis(name)` | What breaks if this changes — run before any refactor |
| `find_dead_code()` | Unreachable functions |
| `get_diagram(name)` | Mermaid call graph |
| `find_callers(name)` | Who calls this? |
| `find_callees(name)` | What does this call? |
| `search_code(pattern)` | Substring search across functions/classes |
| `search_docs(query)` | Hybrid BM25 + vector search |
| `file_dependencies(path)` | Import graph for a file |
| `codebase_stats()` | Languages, counts, last indexed |
| `reindex()` | Incremental reindex (changed files only) |

---

## Anti-hallucination rules for all agents

1. **Never cite a function signature from memory.** Call `verify_signature` first.
2. **Never read a file to understand a function.** Call `get_source` or `find_function` instead.
3. **Never modify a widely-called function without running `impact_analysis` first.**
4. **Before writing an import path, call `find_function` to confirm the qualified name.**
5. If `verify_signature` returns `"match": false`, use the `actual` field — not your recalled version.

---

## Integration by AI platform

### Claude (Claude Desktop / Claude Code)

MCP config (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "codegraph": {
      "command": "codegraph",
      "args": ["serve", "--mcp-stdio", "--repo", "/path/to/your/repo"],
      "env": {
        "CODEGRAPH_DATA_DIR": "/path/to/your/repo/.codegraph"
      }
    }
  }
}
```

Claude Code picks up this file automatically. All 16 MCP tools become available as native tools
in any conversation inside this repo.

---

### OpenAI (GPT-4o / Assistants API / Codex)

**Option A — REST via GraphQL** (works with any OpenAI integration):
```python
import openai, httpx

client = openai.OpenAI()
graph = httpx.Client(base_url="http://localhost:8000")

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_codegraph",
            "description": "Query the codegraph knowledge base. Returns AST-verified facts about functions, classes, call relationships, and import paths in the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "graphql_query": {
                        "type": "string",
                        "description": "A GraphQL query string against the codegraph schema."
                    }
                },
                "required": ["graphql_query"]
            }
        }
    }
]

def handle_tool_call(name, args):
    if name == "query_codegraph":
        resp = graph.post("/graphql", json={"query": args["graphql_query"]})
        return resp.json()

messages = [{"role": "user", "content": "What calls authenticate() and where?"}]
response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
# handle tool_calls in response.choices[0].message.tool_calls
```

**Option B — LangChain (recommended)**:
```python
from langchain_openai import ChatOpenAI
from codegraph.langchain import build_codebase_qa
from codegraph.graph.store import GraphStore
from codegraph.rag.indexer import DocIndexer
from codegraph.rag.retriever import RAGRetriever

store = GraphStore("./data/codegraph.db")
rag = RAGRetriever(store, DocIndexer(store))

qa = build_codebase_qa(store, rag, llm=ChatOpenAI(model="gpt-4o"), k=5)
result = qa.ask("What would break if I changed the User model?")
print(result["answer"])
print(result["sources"])
```

**Option C — Assistants API with file search disabled** (use codegraph instead):
```python
assistant = client.beta.assistants.create(
    name="CodeReviewer",
    instructions=open("INSTRUCTIONS.md").read(),
    model="gpt-4o",
    tools=tools   # the query_codegraph tool above
)
```

---

### Gemini (Gemini 1.5 Pro / Vertex AI)

**Option A — Function calling via Google GenAI SDK**:
```python
import google.generativeai as genai
import httpx

genai.configure(api_key="YOUR_API_KEY")
graph = httpx.Client(base_url="http://localhost:8000")

query_codegraph = genai.protos.FunctionDeclaration(
    name="query_codegraph",
    description="Query codegraph for AST-verified facts about functions, classes, and call relationships.",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "graphql_query": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="GraphQL query string against the codegraph schema."
            )
        },
        required=["graphql_query"]
    )
)

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    tools=[genai.protos.Tool(function_declarations=[query_codegraph])],
    system_instruction=open("INSTRUCTIONS.md").read()
)

chat = model.start_chat()
response = chat.send_message("Which functions in auth.py have no callers?")

# Handle function calls
for part in response.parts:
    if fn := part.function_call:
        result = graph.post("/graphql", json={"query": fn.args["graphql_query"]}).json()
        response = chat.send_message(
            genai.protos.Content(parts=[genai.protos.Part(
                function_response=genai.protos.FunctionResponse(name=fn.name, response=result)
            )])
        )
```

**Option B — LangChain (recommended)**:
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from codegraph.langchain import build_codebase_qa

qa = build_codebase_qa(store, rag, llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro"), k=5)
result = qa.ask("How does session management work?")
```

**Option C — Vertex AI (enterprise)**:
```python
from langchain_google_vertexai import ChatVertexAI
qa = build_codebase_qa(store, rag, llm=ChatVertexAI(model_name="gemini-1.5-pro"), k=5)
```

---

## LangChain quick reference (works with Claude, OpenAI, Gemini)

```python
from codegraph.langchain import CodeGraphRetriever, build_codebase_qa

# Retriever only — plug into any existing chain
retriever = CodeGraphRetriever(store=store, rag=rag, k=5)
docs = retriever.invoke("how does password hashing work?")
# docs[i].metadata → { source, result_kind, start_line, qualified_name }

# Full QA chain
qa = build_codebase_qa(store, rag, llm=your_llm, k=5)
result = qa.ask("What is the risk of modifying process_payment()?")
# result → { answer, sources, context_used, docs_retrieved }
```

---

## Key design decisions — do not change without reading this

- **`_CallExtractor.generic_visit(node)`** in `python_analyzer.py` — must be `generic_visit`, not
  `visit`. Using `visit` on a FunctionDef re-enters `visit_FunctionDef` which silently skips
  all call edges inside the function body.

- **Cross-file resolution pass** in `builder.py` — runs after all files are indexed. Unresolved
  CALLS/IMPORTS edges are linked to real target nodes by qualified name lookup. Removing this
  breaks all cross-file call edges.

- **RRF constant `_RRF_K = 60`** in `retriever.py` — standard value from the RRF paper.

- **Dimension guard in `_faiss_search_code`** — CodeBERT and MiniLM have different embedding
  dimensions. The guard (`q.shape[1] != index.d`) prevents silent wrong-dimension searches.

- **Provenance separation** — `node.docstring` is AST-extracted (ground truth).
  `node.metadata["generated_docstring"]` is CodeT5-generated. Never merge them.

---

## Configuration

All settings via `CODEGRAPH_` environment variables:

```bash
CODEGRAPH_DATA_DIR=./data
CODEGRAPH_REPO_PATH=/path/to/repo
CODEGRAPH_GRAPHQL_PORT=8000
CODEGRAPH_MAX_SUBGRAPH_TOKENS=4000
CODEGRAPH_EMBEDDING_MODEL=all-MiniLM-L6-v2
CODEGRAPH_RESPECT_GITIGNORE=true
CODEGRAPH_MAX_FILE_SIZE_KB=500
```

---

## Extending codegraph

**Add a new MCP tool:**
1. Add query logic in `src/codegraph/graph/queries.py`
2. Add GraphQL resolver in `src/codegraph/graphql/schema.py` if needed
3. Add `@mcp.tool()` function in `src/codegraph/mcp_server/server.py`
4. Add tests in `tests/`
5. Update the tools table in `README.md` and `INSTRUCTIONS.md`

**Add a new language:**
1. Create `src/codegraph/analyzers/<lang>_analyzer.py` (same interface as `python_analyzer.py`)
2. Register it in `builder.py` `_get_analyzer()` dispatch
3. Add to supported languages table in `README.md`
