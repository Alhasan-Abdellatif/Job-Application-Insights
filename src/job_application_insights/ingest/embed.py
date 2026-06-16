"""Turn :class:`Chunk` text into dense vectors using a sentence-transformer model.

What "embedding" means here:

* A small encoder-only transformer takes a string and returns a fixed-size
  numpy vector. Similar text → similar vectors (small cosine distance);
  different text → far vectors. The vector is the "semantic fingerprint" of
  the text.
* We use **BAAI/bge-small-en-v1.5** by default — 384 dimensions, ~130MB, runs
  on CPU at ~1000 texts/sec. Top of the MTEB leaderboard for its size class
  as of 2024-2025. Larger BGE / E5 / mxbai variants are easy drop-in upgrades.
* Embeddings are **L2-normalised** by default, so cosine similarity is just a
  dot product. Most vector DBs assume normalised vectors when using cosine
  similarity — keeps the contract simple downstream.

Design choices:

* ``Embedder`` is a class because loading the model is expensive (~5s, plus a
  one-time HuggingFace download). You instantiate once and reuse the same
  object for the lifetime of a process.
* ``embed()`` takes a sequence of strings (batched internally) — text-only,
  no Chunk coupling. Useful for query embedding at retrieval time.
* ``embed_chunks()`` is the sugar that calls ``embed()`` on chunk texts.
  Returns ``(chunks, embeddings)`` as parallel arrays — the format every
  vector DB expects.
* Returns ``numpy.ndarray`` instead of nested ``list[list[float]]`` — fast,
  memory-efficient, and slottable into any downstream library (Chroma,
  Qdrant, FAISS, scikit-learn) without conversion.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from job_application_insights.ingest.chunk import Chunk

# ────────────────────────────── constants ──────────────────────────────

DEFAULT_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
"""Default embedding model. Small, fast, free, and currently best-in-class
for its size on the MTEB leaderboard."""

DEFAULT_BATCH_SIZE: int = 32
"""Batch size for ``model.encode``. 32 is the standard default — fits in CPU
memory and gives near-peak throughput. Increase on GPU."""


# ────────────────────────────── public API ──────────────────────────────


class Embedder:
    """Wraps a sentence-transformer model to embed text into vectors.

    Parameters
    ----------
    model_name
        Hugging Face repo id of the sentence-transformer model. Defaults to
        ``BAAI/bge-small-en-v1.5``.
    device
        Where to run the model. ``None`` lets sentence-transformers auto-pick
        ("cuda" if available, otherwise "mps" on Apple Silicon, else "cpu").
    batch_size
        Batch size for encoding. Higher batches → better throughput on GPU,
        diminishing returns on CPU.
    normalize
        L2-normalise output vectors. Leave this ``True`` for any cosine-
        similarity-based retrieval (which is everything we'll build).

    Notes
    -----
    Instantiation triggers a one-time model download from Hugging Face Hub.
    Subsequent runs load from the on-disk cache (``~/.cache/huggingface/``)
    in <1s.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        """Embedding vector dimension (384 for ``bge-small-en-v1.5``)."""
        # ``get_embedding_dimension`` is the modern API; older sentence-transformers
        # versions exposed it under the longer name. Prefer the modern one and
        # fall back to the deprecated alias for compatibility.
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        dim = get_dim()
        if dim is None:  # pragma: no cover — sentence-transformers always populates this
            raise RuntimeError(f"could not determine embedding dimension for {self.model_name}")
        return int(dim)

    def embed(
        self,
        texts: Sequence[str],
        *,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Embed a sequence of strings into a 2-D numpy array.

        Parameters
        ----------
        texts
            Strings to embed. Empty strings are allowed but probably useless —
            most embedding models give them a near-zero vector.
        show_progress_bar
            Whether to print a tqdm bar while encoding. Useful for big batches
            (>1000 texts); off by default to keep test output clean.

        Returns
        -------
        ``ndarray`` of shape ``(len(texts), self.dimension)``, dtype float32,
        L2-normalised if ``self.normalize`` is True.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        # sentence-transformers sometimes returns float64 on CPU — pin to float32
        # for consistency with what vector DBs expect.
        return np.asarray(embeddings, dtype=np.float32)

    def embed_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Embed each chunk's ``.text``. Returns ``ndarray (n_chunks, dimension)``.

        The returned array is parallel to ``chunks``: ``embeddings[i]``
        corresponds to ``chunks[i]``. Callers must keep them aligned when
        passing to a vector store.
        """
        return self.embed([c.text for c in chunks], show_progress_bar=show_progress_bar)


def assert_aligned(chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
    """Defensive check: raise if ``chunks`` and ``embeddings`` aren't parallel.

    Use at the boundary of any function that consumes both — better to fail
    loudly here than to silently store the wrong embedding for the wrong
    chunk in the vector DB.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) length mismatch"
        )
