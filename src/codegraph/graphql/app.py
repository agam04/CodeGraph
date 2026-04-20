from pathlib import Path

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

from codegraph.config import CodegraphConfig
from codegraph.graph.store import GraphStore
from codegraph.graphql.schema import build_schema
from codegraph.rag.indexer import DocIndexer
from codegraph.rag.retriever import RAGRetriever
from codegraph.utils.logging import get_logger

log = get_logger(__name__)


def create_app(config: CodegraphConfig, store: GraphStore, indexer: DocIndexer) -> FastAPI:
    retriever = RAGRetriever(store, indexer)
    schema = build_schema(store, retriever)

    graphql_router = GraphQLRouter(schema, graphiql=True)

    app = FastAPI(
        title="codegraph",
        description="Graph-based codebase intelligence for AI agents.",
        version="0.1.0",
    )
    app.include_router(graphql_router, prefix="/graphql")

    @app.get("/health")
    async def health():
        return {"status": "ok", "stats": store.stats()}

    return app
