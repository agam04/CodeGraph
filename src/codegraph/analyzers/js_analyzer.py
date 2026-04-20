from pathlib import Path
from typing import Optional

from codegraph.analyzers.base import AnalysisResult, BaseAnalyzer
from codegraph.graph.schema import Edge, EdgeType, Node, NodeType
from codegraph.utils.hashing import hash_content, node_id
from codegraph.utils.logging import get_logger

log = get_logger(__name__)


def _extract_jsdoc(text: str, end_byte: int, source: bytes) -> Optional[str]:
    """Find JSDoc comment immediately before a node."""
    before = source[:end_byte].rstrip()
    if before.endswith(b"*/"):
        start = before.rfind(b"/*")
        if start != -1:
            comment = before[start:].decode("utf-8", errors="replace")
            lines = []
            for line in comment.splitlines():
                line = line.strip().lstrip("/*").lstrip("*").strip()
                if line:
                    lines.append(line)
            return " ".join(lines) or None
    return None


class JSAnalyzer(BaseAnalyzer):
    def __init__(self, lang: str = "javascript") -> None:
        self.lang = lang
        self._parser = None
        self._language = None

    def _get_parser(self):
        if self._parser is not None:
            return self._parser
        try:
            import tree_sitter_javascript as tsjs
            import tree_sitter_typescript as tsts
            from tree_sitter import Language, Parser

            if self.lang == "typescript":
                lang_obj = Language(tsts.language_typescript())
            elif self.lang == "tsx":
                lang_obj = Language(tsts.language_tsx())
            else:
                lang_obj = Language(tsjs.language())

            self._language = lang_obj
            self._parser = Parser(lang_obj)
        except Exception as e:
            log.error("js_parser_init_failed", lang=self.lang, error=str(e))
            self._parser = None
        return self._parser

    def analyze(self, file_path: Path, content: str) -> AnalysisResult:
        parser = self._get_parser()
        if parser is None:
            return AnalysisResult(nodes=[], edges=[])

        content_bytes = content.encode("utf-8")
        try:
            tree = parser.parse(content_bytes)
        except Exception as e:
            log.warning("js_parse_error", file=str(file_path), error=str(e))
            return AnalysisResult(nodes=[], edges=[])

        content_hash = hash_content(content)
        fp = str(file_path)
        module_name = file_path.stem
        lines = content.splitlines()

        module_node = Node(
            id=node_id(fp, module_name),
            node_type=NodeType.MODULE,
            name=module_name,
            qualified_name=module_name,
            file_path=fp,
            start_line=1,
            end_line=len(lines),
            language=self.lang,
            source_hash=content_hash,
        )

        nodes: list[Node] = [module_node]
        edges: list[Edge] = []
        imports: dict[str, str] = {}

        cursor = tree.walk()
        self._walk(cursor, content_bytes, fp, module_name, content_hash, nodes, edges, imports, lines)

        # Import edges
        for alias, target in imports.items():
            edges.append(
                Edge(
                    module_node.id,
                    node_id(fp, target),
                    EdgeType.IMPORTS,
                    {"alias": alias, "target_module": target, "unresolved": True},
                )
            )

        return AnalysisResult(nodes=nodes, edges=edges)

    def _node_text(self, node, source: bytes) -> str:
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _walk(self, cursor, source: bytes, fp: str, module_name: str, content_hash: str,
              nodes: list[Node], edges: list[Edge], imports: dict[str, str], lines: list[str]) -> None:
        root = cursor.node
        self._visit_node(root, source, fp, module_name, content_hash, nodes, edges, imports)

    def _visit_node(self, node, source: bytes, fp: str, module_name: str, content_hash: str,
                    nodes: list[Node], edges: list[Edge], imports: dict[str, str]) -> None:
        module_id = node_id(fp, module_name)

        if node.type in ("import_statement", "import_declaration"):
            self._handle_import(node, source, imports)

        elif node.type == "call_expression":
            # CommonJS require()
            func = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if func and self._node_text(func, source) == "require" and args:
                for child in args.children:
                    if child.type == "string":
                        val = self._node_text(child, source).strip("'\"`")
                        imports[val] = val

        elif node.type in ("function_declaration", "function_expression",
                           "arrow_function", "method_definition",
                           "generator_function_declaration"):
            self._handle_function(node, source, fp, module_name, content_hash,
                                  nodes, edges, module_id)

        elif node.type == "class_declaration":
            self._handle_class(node, source, fp, module_name, content_hash,
                               nodes, edges, module_id)

        for child in node.children:
            self._visit_node(child, source, fp, module_name, content_hash, nodes, edges, imports)

    def _handle_import(self, node, source: bytes, imports: dict[str, str]) -> None:
        source_child = node.child_by_field_name("source")
        if source_child:
            module_path = self._node_text(source_child, source).strip("'\"`")
            # Find what's imported
            for child in node.children:
                if child.type == "import_clause":
                    for sub in child.children:
                        if sub.type == "identifier":
                            imports[self._node_text(sub, source)] = module_path
                        elif sub.type in ("named_imports", "namespace_import"):
                            for item in sub.children:
                                if item.type == "import_specifier":
                                    name_node = item.child_by_field_name("name")
                                    if name_node:
                                        imports[self._node_text(name_node, source)] = module_path

    def _handle_function(self, node, source: bytes, fp: str, module_name: str,
                         content_hash: str, nodes: list[Node], edges: list[Edge], parent_id: str) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return  # anonymous function without assignment context
        name = self._node_text(name_node, source)
        qualified = f"{module_name}.{name}"
        is_async = any(c.type == "async" for c in node.children)
        params_node = node.child_by_field_name("parameters")
        signature = self._node_text(params_node, source) if params_node else "()"

        docstring = _extract_jsdoc(
            self._node_text(node, source),
            node.start_byte,
            source,
        )

        func_node = Node(
            id=node_id(fp, qualified),
            node_type=NodeType.FUNCTION,
            name=name,
            qualified_name=qualified,
            file_path=fp,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            language=self.lang,
            docstring=docstring,
            signature=signature,
            source_hash=content_hash,
            metadata={"is_async": is_async},
        )
        nodes.append(func_node)
        edges.append(Edge(parent_id, func_node.id, EdgeType.DEFINES))

    def _handle_class(self, node, source: bytes, fp: str, module_name: str,
                      content_hash: str, nodes: list[Node], edges: list[Edge], parent_id: str) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = self._node_text(name_node, source)
        qualified = f"{module_name}.{name}"
        class_node_obj = Node(
            id=node_id(fp, qualified),
            node_type=NodeType.CLASS,
            name=name,
            qualified_name=qualified,
            file_path=fp,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            language=self.lang,
            source_hash=content_hash,
        )
        nodes.append(class_node_obj)
        edges.append(Edge(parent_id, class_node_obj.id, EdgeType.DEFINES))

        # Heritage (extends)
        heritage = node.child_by_field_name("heritage")
        if heritage:
            for child in heritage.children:
                if child.type == "extends_clause":
                    for sub in child.children:
                        if sub.type == "identifier":
                            base_name = self._node_text(sub, source)
                            base_id = node_id(fp, f"{module_name}.{base_name}")
                            edges.append(Edge(class_node_obj.id, base_id, EdgeType.INHERITS))

        # Methods
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "method_definition":
                    mname_node = child.child_by_field_name("name")
                    if mname_node:
                        mname = self._node_text(mname_node, source)
                        mqualified = f"{qualified}.{mname}"
                        params_node = child.child_by_field_name("parameters")
                        sig = self._node_text(params_node, source) if params_node else "()"
                        is_async = any(c.type == "async" for c in child.children)
                        method_node = Node(
                            id=node_id(fp, mqualified),
                            node_type=NodeType.METHOD,
                            name=mname,
                            qualified_name=mqualified,
                            file_path=fp,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            language=self.lang,
                            source_hash=content_hash,
                            signature=sig,
                            metadata={"is_async": is_async},
                        )
                        nodes.append(method_node)
                        edges.append(Edge(class_node_obj.id, method_node.id, EdgeType.DEFINES))
