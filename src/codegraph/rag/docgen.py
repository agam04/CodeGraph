"""Auto-docstring generation using Salesforce CodeT5.

For every function that has no docstring, CodeT5 generates one from its
source code. The generated docstring is stored in node metadata under
`generated_docstring` — distinct from `docstring` (which is AST-extracted)
so callers always know the provenance.

Model: Salesforce/codet5-base-codexglue-sum-python (~900MB)
Task:  code summarisation → short docstring

Usage:
    generator = DocstringGenerator()
    docstring = generator.generate("def add(a, b):\\n    return a + b")
    # → "Add two numbers and return their sum."
"""

from pathlib import Path
from typing import Optional

from codegraph.graph.schema import Node, NodeType
from codegraph.graph.store import GraphStore
from codegraph.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL = "Salesforce/codet5-base-codexglue-sum-python"
_FALLBACK_MODEL = "Salesforce/codet5-base"


class DocstringGenerator:
    """Generates docstrings for undocumented functions using CodeT5."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._pipeline = None
        self._available = True

    def generate(self, source_code: str, language: str = "python") -> Optional[str]:
        """Generate a one-line docstring for the given source code snippet."""
        pipeline = self._get_pipeline()
        if pipeline is None:
            return None
        try:
            # CodeT5 expects the source with a summarize prefix
            prompt = f"Summarize Python: {source_code.strip()}"
            result = pipeline(
                prompt,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
            )
            text = result[0]["generated_text"].strip()
            # Capitalise and ensure it ends with a period
            if text and not text.endswith("."):
                text += "."
            return text or None
        except Exception as e:
            log.warning("docgen_generate_failed", error=str(e))
            return None

    def enrich_store(self, store: GraphStore, overwrite: bool = False) -> dict:
        """Generate docstrings for all undocumented functions in the store.

        Args:
            store: The graph store to enrich.
            overwrite: If True, regenerate even for functions that already
                       have a `generated_docstring` in metadata.

        Returns:
            Summary dict with counts.
        """
        targets = (
            store.get_all_nodes(NodeType.FUNCTION)
            + store.get_all_nodes(NodeType.METHOD)
        )

        skipped = generated = failed = 0
        for node in targets:
            # Already has a real docstring — skip
            if node.docstring:
                skipped += 1
                continue
            # Already enriched and not overwriting
            if not overwrite and node.metadata.get("generated_docstring"):
                skipped += 1
                continue

            source = self._read_source(node)
            if not source:
                failed += 1
                continue

            doc = self.generate(source, language=node.language or "python")
            if doc:
                node.metadata["generated_docstring"] = doc
                node.metadata["docstring_provenance"] = "codet5_generated"
                store.upsert_node(node)
                generated += 1
                log.debug("docstring_generated", name=node.qualified_name)
            else:
                failed += 1

        store.commit()
        log.info(
            "docstring_enrichment_complete",
            generated=generated,
            skipped=skipped,
            failed=failed,
        )
        return {"generated": generated, "skipped": skipped, "failed": failed}

    def _get_pipeline(self):
        if not self._available:
            return None
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
            log.info("loading_docgen_model", model=self.model_name)
            self._pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
            )
            log.info("docgen_model_loaded", model=self.model_name)
            return self._pipeline
        except Exception as e:
            log.warning("docgen_model_unavailable", model=self.model_name, error=str(e))
            # Try the base model as fallback
            if self.model_name != _FALLBACK_MODEL:
                self.model_name = _FALLBACK_MODEL
                return self._get_pipeline()
            self._available = False
            return None

    @staticmethod
    def _read_source(node: Node) -> Optional[str]:
        try:
            lines = Path(node.file_path).read_text(errors="replace").splitlines()
            start = max(0, node.start_line - 1)
            end = min(len(lines), node.end_line)
            return "\n".join(lines[start:end])
        except OSError:
            return None
