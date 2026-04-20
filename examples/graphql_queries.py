"""Example GraphQL queries to run against a live codegraph server."""
import httpx

BASE = "http://localhost:8000/graphql"


def gql(query: str) -> dict:
    resp = httpx.post(BASE, json={"query": query})
    resp.raise_for_status()
    return resp.json()


# 1. Find a function and its callers/callees
print(gql("""
{
  function(name: "authenticate") {
    name
    qualifiedName
    filePath
    startLine
    signature
    docstring
    isAsync
    callers { name filePath startLine }
    callees { name filePath startLine }
  }
}
"""))

# 2. Get the full context subgraph — this is what you pass to an agent
print(gql("""
{
  contextFor(qualifiedName: "auth.authenticate", depth: 2) {
    summary
    estimatedTokens
    centerNode { name nodeType filePath }
    relatedNodes { name nodeType filePath }
    edges { sourceId targetId edgeType }
  }
}
"""))

# 3. Semantic doc search
print(gql("""
{
  searchDocs(query: "how does authentication work", limit: 3) {
    content
    source
    relevanceScore
  }
}
"""))

# 4. Codebase overview
print(gql("""
{
  stats {
    totalFiles
    totalFunctions
    totalClasses
    totalNodes
    totalEdges
    languages
    lastIndexed
  }
}
"""))
