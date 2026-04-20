import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from codegraph.config import CodegraphConfig, set_config
from codegraph.utils.logging import configure_logging

console = Console()


def _build_store_and_config(repo: Optional[Path], data_dir: Optional[Path]) -> tuple:
    config = CodegraphConfig(
        repo_path=repo,
        data_dir=data_dir or Path("./data"),
    )
    set_config(config)
    configure_logging(config.log_level)
    from codegraph.graph.store import GraphStore
    db_path = config.data_dir / "codegraph.db"
    store = GraphStore(db_path)
    return config, store


@click.group()
def cli():
    """codegraph — Give AI agents a graph view of your codebase."""


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--data-dir", type=click.Path(path_type=Path), default=None, help="Where to store the graph DB.")
@click.option("--no-incremental", is_flag=True, default=False, help="Force full reindex.")
@click.option("--index-docs", is_flag=True, default=True, help="Also index documentation (README, /docs).")
def index(repo_path: Path, data_dir: Optional[Path], no_incremental: bool, index_docs: bool):
    """Index a codebase into the graph."""
    config, store = _build_store_and_config(repo_path, data_dir)

    from codegraph.graph.builder import GraphBuilder
    with console.status(f"[bold green]Indexing {repo_path}..."):
        builder = GraphBuilder(repo_path, store, config)
        stats = builder.build(incremental=not no_incremental)

    console.print(f"[green]✓[/green] Indexed {stats.files_indexed} files "
                  f"({stats.files_skipped} unchanged), "
                  f"{stats.nodes_created} nodes, {stats.edges_created} edges "
                  f"in {stats.time_elapsed:.2f}s")

    if stats.errors:
        console.print(f"[yellow]⚠ {len(stats.errors)} error(s):[/yellow]")
        for err in stats.errors[:5]:
            console.print(f"  {err}")

    if index_docs:
        from codegraph.rag.indexer import DocIndexer
        with console.status("[bold blue]Indexing documentation..."):
            indexer = DocIndexer(store, config.chunk_size, config.chunk_overlap)
            chunks = indexer.index_repo(repo_path)
        console.print(f"[blue]✓[/blue] Indexed {chunks} doc chunks")

    store.close()


@cli.command()
@click.option("--graphql-port", default=8000, help="Port for GraphQL HTTP server.")
@click.option("--mcp-stdio", is_flag=True, default=False, help="Run MCP server on stdio.")
@click.option("--repo", type=click.Path(path_type=Path), default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
def serve(graphql_port: int, mcp_stdio: bool, repo: Optional[Path], data_dir: Optional[Path]):
    """Start the GraphQL and/or MCP server."""
    config, store = _build_store_and_config(repo, data_dir)

    from codegraph.rag.indexer import DocIndexer
    from codegraph.rag.retriever import RAGRetriever
    indexer = DocIndexer(store)
    retriever = RAGRetriever(store, indexer)

    if mcp_stdio:
        from codegraph.mcp_server.server import init_mcp, mcp
        init_mcp(store, retriever, config)
        console.print("[blue]Starting MCP server on stdio...[/blue]")
        mcp.run(transport="stdio")
    else:
        import uvicorn
        from codegraph.graphql.app import create_app
        app = create_app(config, store, indexer)
        console.print(f"[green]GraphQL server at http://localhost:{graphql_port}/graphql[/green]")
        uvicorn.run(app, host="0.0.0.0", port=graphql_port)


@cli.command()
@click.option("--repo", type=click.Path(path_type=Path), default=Path("."))
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
def reindex(repo: Path, data_dir: Optional[Path]):
    """Incrementally reindex a codebase (only changed files)."""
    config, store = _build_store_and_config(repo, data_dir)
    from codegraph.graph.builder import GraphBuilder
    with console.status("[bold green]Reindexing..."):
        builder = GraphBuilder(repo, store, config)
        stats = builder.build(incremental=True)
    console.print(f"[green]✓[/green] {stats.files_indexed} changed, {stats.files_skipped} skipped "
                  f"in {stats.time_elapsed:.2f}s")
    store.close()


@cli.command()
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
def stats(data_dir: Optional[Path]):
    """Show codebase statistics."""
    config, store = _build_store_and_config(None, data_dir)
    s = store.stats()
    table = Table(title="Codebase Stats")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Files", str(s["total_files"]))
    table.add_row("Nodes", str(s["total_nodes"]))
    table.add_row("Edges", str(s["total_edges"]))
    for lang, cnt in s["languages"].items():
        table.add_row(f"  {lang} modules", str(cnt))
    for nt, cnt in s["type_counts"].items():
        table.add_row(f"  {nt}s", str(cnt))
    table.add_row("Last indexed", str(s["last_indexed"] or "never"))
    console.print(table)
    store.close()


@cli.command()
@click.argument("tool_name")
@click.option("--name", default=None)
@click.option("--data-dir", type=click.Path(path_type=Path), default=None)
def query(tool_name: str, name: Optional[str], data_dir: Optional[Path]):
    """Quick query from the terminal (e.g. codegraph query find_function --name authenticate)."""
    import json
    config, store = _build_store_and_config(None, data_dir)
    from codegraph.rag.indexer import DocIndexer
    from codegraph.rag.retriever import RAGRetriever
    from codegraph.mcp_server.server import init_mcp
    indexer = DocIndexer(store)
    retriever = RAGRetriever(store, indexer)
    init_mcp(store, retriever, config)

    import codegraph.mcp_server.server as srv
    tool_fn = getattr(srv, tool_name, None)
    if tool_fn is None:
        console.print(f"[red]Unknown tool: {tool_name}[/red]")
        sys.exit(1)

    kwargs = {}
    if name:
        kwargs["name"] = name
    result = tool_fn(**kwargs)
    console.print_json(json.dumps(result, default=str))
    store.close()


if __name__ == "__main__":
    cli()
