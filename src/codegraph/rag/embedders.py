"""Code-aware dual embedding strategy.

Uses a code-specific model (CodeT5p / CodeBERT) for functions and classes,
and a general text model (all-MiniLM-L6-v2) for documentation and README chunks.

Why this matters: general text models embed "for i in range(n)" and
"iterate n times" similarly because they share meaning. But for code search,
we want "authenticate(user, password)" to be close to other auth functions
regardless of their prose description. Code-specific models trained on
CodeSearchNet achieve this because they've seen millions of code/docstring pairs.
"""


import numpy as np

from codegraph.utils.logging import get_logger

log = get_logger(__name__)

# Model identifiers
CODE_EMBEDDING_MODEL = "microsoft/codebert-base"
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class CodeAwareEmbedder:
    """Dual-model embedder: code-specific model for symbols, text model for docs.

    Falls back to the text model for both if the code model can't be loaded.
    """

    def __init__(
        self,
        code_model_name: str = CODE_EMBEDDING_MODEL,
        text_model_name: str = TEXT_EMBEDDING_MODEL,
    ) -> None:
        self.code_model_name = code_model_name
        self.text_model_name = text_model_name
        self._text_model = None
        self._code_tokenizer = None
        self._code_model = None
        self._use_code_model = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_code(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed source code snippets using the code-specific model."""
        if self._use_code_model:
            try:
                return self._encode_with_codebert(texts)
            except Exception as e:
                log.warning("codebert_failed_falling_back", error=str(e))
                self._use_code_model = False
        return self._encode_with_text_model(texts, show_progress)

    def embed_text(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed natural language text using the general text model."""
        return self._encode_with_text_model(texts, show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query. Uses text model — queries are natural language."""
        return self._encode_with_text_model([query])[0]

    @property
    def code_dim(self) -> int:
        """Embedding dimension for code vectors."""
        if self._use_code_model and self._ensure_code_model():
            return 768  # CodeBERT hidden size
        return self.text_dim

    @property
    def text_dim(self) -> int:
        """Embedding dimension for text vectors."""
        model = self._get_text_model()
        if model is not None:
            return model.get_sentence_embedding_dimension()
        return 384  # all-MiniLM-L6-v2 default

    # ── Private ────────────────────────────────────────────────────────────────

    def _get_text_model(self):
        if self._text_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._text_model = SentenceTransformer(self.text_model_name)
                log.info("text_model_loaded", model=self.text_model_name)
            except Exception as e:
                log.error("text_model_load_failed", model=self.text_model_name, error=str(e))
        return self._text_model

    def _ensure_code_model(self) -> bool:
        if self._code_model is not None:
            return True
        try:
            from transformers import AutoModel, AutoTokenizer
            self._code_tokenizer = AutoTokenizer.from_pretrained(self.code_model_name)
            self._code_model = AutoModel.from_pretrained(self.code_model_name)
            self._code_model.eval()
            log.info("code_model_loaded", model=self.code_model_name)
            return True
        except Exception as e:
            log.warning("code_model_load_failed", model=self.code_model_name, error=str(e))
            self._use_code_model = False
            return False

    def _encode_with_codebert(self, texts: list[str]) -> np.ndarray:
        """Mean-pool the last hidden state from CodeBERT."""
        import torch

        if not self._ensure_code_model():
            return self._encode_with_text_model(texts)

        embeddings = []
        # Process in small batches to avoid OOM
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Truncate long code to 512 tokens
            inputs = self._code_tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            with torch.no_grad():
                outputs = self._code_model(**inputs)
            # Mean pool over non-padding tokens
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            mean_pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
            embeddings.append(mean_pooled.numpy())

        return np.vstack(embeddings)

    def _encode_with_text_model(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        model = self._get_text_model()
        if model is None:
            # Last-resort: zero vectors
            return np.zeros((len(texts), 384), dtype="float32")
        return np.array(
            model.encode(texts, show_progress_bar=show_progress),
            dtype="float32",
        )
