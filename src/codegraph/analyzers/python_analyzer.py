import ast
from pathlib import Path
from typing import Optional

from codegraph.analyzers.base import AnalysisResult, BaseAnalyzer
from codegraph.graph.schema import Edge, EdgeType, Node, NodeType
from codegraph.utils.hashing import hash_content, node_id
from codegraph.utils.logging import get_logger

log = get_logger(__name__)


def _get_docstring(node: ast.AST) -> Optional[str]:
    if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return ast.get_docstring(node)


def _build_signature(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = func.args
    parts: list[str] = []

    defaults_offset = len(args.args) - len(args.defaults)

    for i, arg in enumerate(args.args):
        ann = ast.unparse(arg.annotation) if arg.annotation else ""
        default_idx = i - defaults_offset
        if default_idx >= 0:
            default = f"={ast.unparse(args.defaults[default_idx])}"
        else:
            default = ""
        parts.append(f"{arg.arg}{(':' + ann) if ann else ''}{default}")

    if args.vararg:
        ann = f": {ast.unparse(args.vararg.annotation)}" if args.vararg.annotation else ""
        parts.append(f"*{args.vararg.arg}{ann}")

    if args.kwarg:
        ann = f": {ast.unparse(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        parts.append(f"**{args.kwarg.arg}{ann}")

    ret = f" -> {ast.unparse(func.returns)}" if func.returns else ""
    return f"({', '.join(parts)}){ret}"


class _Visitor(ast.NodeVisitor):
    def __init__(self, file_path: str, module_name: str, content_hash: str) -> None:
        self.file_path = file_path
        self.module_name = module_name
        self.content_hash = content_hash
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self._scope_stack: list[str] = [module_name]
        self._class_stack: list[str] = []
        # Maps local alias → qualified import target
        self._imports: dict[str, str] = {}

    def _current_scope(self) -> str:
        return self._scope_stack[-1]

    def _make_node(
        self,
        node_type: NodeType,
        name: str,
        qualified: str,
        start: int,
        end: int,
        docstring: Optional[str] = None,
        signature: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Node:
        nid = node_id(self.file_path, qualified)
        return Node(
            id=nid,
            node_type=node_type,
            name=name,
            qualified_name=qualified,
            file_path=self.file_path,
            start_line=start,
            end_line=end,
            language="python",
            docstring=docstring,
            signature=signature,
            source_hash=self.content_hash,
            metadata=metadata or {},
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self._imports[local] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            target = f"{base}.{alias.name}" if base else alias.name
            self._imports[local] = target

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified = f"{self._current_scope()}.{node.name}"
        class_node = self._make_node(
            NodeType.CLASS,
            node.name,
            qualified,
            node.lineno,
            node.end_lineno or node.lineno,
            docstring=_get_docstring(node),
            metadata={"decorators": [ast.unparse(d) for d in node.decorator_list]},
        )
        self.nodes.append(class_node)

        # DEFINES edge from module/parent
        parent_id = node_id(self.file_path, self._current_scope())
        self.edges.append(Edge(parent_id, class_node.id, EdgeType.DEFINES))

        # INHERITS edges
        for base in node.bases:
            base_name = ast.unparse(base)
            resolved = self._imports.get(base_name.split(".")[0], base_name)
            base_qualified = resolved if "." in resolved else f"{self.module_name}.{resolved}"
            base_id = node_id(self.file_path, base_qualified)
            self.edges.append(Edge(class_node.id, base_id, EdgeType.INHERITS, {"base_name": base_name}))

        self._scope_stack.append(qualified)
        self._class_stack.append(qualified)
        self.generic_visit(node)
        self._class_stack.pop()
        self._scope_stack.pop()

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_method = bool(self._class_stack)
        ntype = NodeType.METHOD if is_method else NodeType.FUNCTION
        qualified = f"{self._current_scope()}.{node.name}"
        is_async = isinstance(node, ast.AsyncFunctionDef)

        func_node = self._make_node(
            ntype,
            node.name,
            qualified,
            node.lineno,
            node.end_lineno or node.lineno,
            docstring=_get_docstring(node),
            signature=_build_signature(node),
            metadata={
                "is_async": is_async,
                "decorators": [ast.unparse(d) for d in node.decorator_list],
            },
        )
        self.nodes.append(func_node)

        parent_id = node_id(self.file_path, self._current_scope())
        self.edges.append(Edge(parent_id, func_node.id, EdgeType.DEFINES))

        # Extract calls from the function body
        self._scope_stack.append(qualified)
        call_extractor = _CallExtractor(self.file_path, qualified, func_node.id, self._imports, self.module_name)
        call_extractor.generic_visit(node)
        self.edges.extend(call_extractor.edges)
        # Visit nested functions/classes (don't use generic_visit to avoid double-visiting calls)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(child)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Only capture module-level variables
        if len(self._scope_stack) != 1:
            return
        for target in node.targets:
            if isinstance(target, ast.Name):
                qualified = f"{self.module_name}.{target.id}"
                var_node = self._make_node(
                    NodeType.VARIABLE,
                    target.id,
                    qualified,
                    node.lineno,
                    node.end_lineno or node.lineno,
                )
                self.nodes.append(var_node)
                parent_id = node_id(self.file_path, self.module_name)
                self.edges.append(Edge(parent_id, var_node.id, EdgeType.DEFINES))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if len(self._scope_stack) != 1:
            return
        if isinstance(node.target, ast.Name):
            qualified = f"{self.module_name}.{node.target.id}"
            var_node = self._make_node(
                NodeType.VARIABLE,
                node.target.id,
                qualified,
                node.lineno,
                node.end_lineno or node.lineno,
                metadata={"annotation": ast.unparse(node.annotation)},
            )
            self.nodes.append(var_node)
            parent_id = node_id(self.file_path, self.module_name)
            self.edges.append(Edge(parent_id, var_node.id, EdgeType.DEFINES))


class _CallExtractor(ast.NodeVisitor):
    """Extracts CALLS edges from a function body without descending into nested defs."""

    def __init__(
        self,
        file_path: str,
        caller_qualified: str,
        caller_id: str,
        imports: dict[str, str],
        module_name: str,
    ) -> None:
        self.file_path = file_path
        self.caller_qualified = caller_qualified
        self.caller_id = caller_id
        self.imports = imports
        self.module_name = module_name
        self.edges: list[Edge] = []
        self._seen_callees: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass  # don't descend into nested functions

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass

    def visit_Call(self, node: ast.Call) -> None:
        callee_name = self._resolve_call(node.func)
        if callee_name and callee_name not in self._seen_callees:
            self._seen_callees.add(callee_name)
            callee_id = node_id(self.file_path, callee_name)
            self.edges.append(
                Edge(self.caller_id, callee_id, EdgeType.CALLS, {"line": node.lineno, "unresolved_name": callee_name})
            )
        self.generic_visit(node)

    def _resolve_call(self, node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Name):
            local = node.id
            if local in self.imports:
                return self.imports[local]
            return f"{self.module_name}.{local}"
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                obj = node.value.id
                attr = node.attr
                if obj in self.imports:
                    return f"{self.imports[obj]}.{attr}"
                return f"{self.module_name}.{obj}.{attr}"
        return None


class PythonAnalyzer(BaseAnalyzer):
    def analyze(self, file_path: Path, content: str) -> AnalysisResult:
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            log.warning("python_parse_error", file=str(file_path), error=str(e))
            return AnalysisResult(nodes=[], edges=[])

        content_hash = hash_content(content)
        # Derive module name from file path (best-effort)
        module_name = file_path.stem
        if file_path.name == "__init__.py":
            module_name = file_path.parent.name

        fp = str(file_path)

        # File-level module node
        module_node = Node(
            id=node_id(fp, module_name),
            node_type=NodeType.MODULE,
            name=module_name,
            qualified_name=module_name,
            file_path=fp,
            start_line=1,
            end_line=len(content.splitlines()),
            language="python",
            docstring=_get_docstring(tree),
            source_hash=content_hash,
        )

        visitor = _Visitor(fp, module_name, content_hash)
        visitor.visit(tree)

        nodes = [module_node] + visitor.nodes
        edges = visitor.edges

        # Import edges: file → imported module (resolved later by builder)
        for local, target in visitor._imports.items():
            edges.append(
                Edge(
                    module_node.id,
                    node_id(fp, target),
                    EdgeType.IMPORTS,
                    {"alias": local, "target_module": target, "unresolved": True},
                )
            )

        return AnalysisResult(nodes=nodes, edges=edges)
