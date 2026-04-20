"""Example: index a repo and run basic queries programmatically."""
from pathlib import Path

from codegraph.config import CodegraphConfig, set_config
from codegraph.graph.builder import GraphBuilder
from codegraph.graph.store import GraphStore
from codegraph.graph import queries as gq
from codegraph.utils.logging import configure_logging

configure_logging("INFO")

config = CodegraphConfig(
    data_dir=Path("./data"),
    repo_path=Path("."),
)
set_config(config)

store = GraphStore(config.data_dir / "codegraph.db")

print("Indexing...")
builder = GraphBuilder(Path("."), store, config)
stats = builder.build(incremental=True)
print(f"Done: {stats.files_indexed} files, {stats.nodes_created} nodes, {stats.edges_created} edges")

# Query a function
node = store.get_node_by_name("authenticate")
if node:
    print(f"\nFound: {node.qualified_name} ({node.file_path}:{node.start_line})")
    print(f"Signature: {node.signature}")
    print(f"Docstring: {node.docstring}")

    callers = gq.get_callers(store, node.id)
    print(f"\nCallers ({len(callers)}):")
    for c in callers:
        print(f"  {c.qualified_name} @ {c.file_path}:{c.start_line}")

    callees = gq.get_callees(store, node.id)
    print(f"\nCallees ({len(callees)}):")
    for c in callees:
        print(f"  {c.qualified_name}")

store.close()
