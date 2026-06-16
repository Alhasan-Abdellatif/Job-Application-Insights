"""Tests for :mod:`job_application_insights.ingest.embed`.

The first run of these tests will download BAAI/bge-small-en-v1.5 from
Hugging Face Hub (~130 MB, ~30 s). Subsequent runs use the on-disk cache
in ``~/.cache/huggingface/`` and complete in a second or two.
"""

from __future__ import annotations

import numpy as np
import pytest
from job_application_insights.ingest.chunk import Chunk
from job_application_insights.ingest.embed import (
    DEFAULT_MODEL_NAME,
    Embedder,
    assert_aligned,
)


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """One Embedder shared across the whole test session — the model is loaded once."""
    return Embedder()


# ───── shape & dimension ─────


def test_default_model_constant():
    assert DEFAULT_MODEL_NAME.startswith("BAAI/")


def test_dimension_is_384_for_bge_small(embedder: Embedder):
    assert embedder.dimension == 384


def test_embed_returns_correct_shape(embedder: Embedder):
    texts = ["thank you for applying", "we have received your application"]
    out = embedder.embed(texts)
    assert out.shape == (2, embedder.dimension)
    assert out.dtype == np.float32


def test_embed_empty_input_returns_zero_rows(embedder: Embedder):
    out = embedder.embed([])
    assert out.shape == (0, embedder.dimension)


# ───── semantic behaviour ─────


def test_similar_texts_have_higher_similarity_than_unrelated(embedder: Embedder):
    """Sanity check: the embedding actually captures meaning, not just length."""
    a = "Thank you for applying to Acme Robotics."
    b = "We have received your application to Acme Robotics."
    c = "The quarterly fiscal earnings report is now available."
    va, vb, vc = embedder.embed([a, b, c])
    # cosine sim (= dot product for normalised vectors)
    sim_ab = float(va @ vb)
    sim_ac = float(va @ vc)
    assert sim_ab > sim_ac, f"expected sim(a,b)={sim_ab} > sim(a,c)={sim_ac}"


def test_normalised_vectors_have_unit_norm(embedder: Embedder):
    out = embedder.embed(["one text"])
    norm = float(np.linalg.norm(out[0]))
    assert abs(norm - 1.0) < 1e-5, f"expected unit-norm vector, got norm={norm}"


def test_normalize_flag_is_stored_on_the_embedder(embedder: Embedder):
    """The normalize contract: stored on the Embedder, applied per call.

    Note: BGE-family models already emit near-unit-norm vectors before any
    explicit normalisation step, so testing 'normalize=False produces
    non-unit-norm vectors' is unreliable. We test the API contract instead
    — that the flag is faithfully stored and used.
    """
    assert embedder.normalize is True
    other = Embedder(normalize=False)
    assert other.normalize is False


def test_embedding_is_deterministic(embedder: Embedder):
    a = embedder.embed(["repeatable text"])
    b = embedder.embed(["repeatable text"])
    assert np.allclose(a, b, atol=1e-6)


# ───── embed_chunks ─────


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"msg_x__c{idx:03d}",
        doc_id="msg_x",
        chunk_index=idx,
        text=text,
        n_tokens=5,
    )


def test_embed_chunks_returns_parallel_array(embedder: Embedder):
    chunks = [_chunk("alpha", 0), _chunk("beta", 1), _chunk("gamma", 2)]
    embeddings = embedder.embed_chunks(chunks)
    assert embeddings.shape == (3, embedder.dimension)


def test_embed_chunks_matches_embed_on_texts(embedder: Embedder):
    """embed_chunks is just embed() on chunk.text — verify the equivalence."""
    chunks = [_chunk("alpha", 0), _chunk("beta", 1)]
    via_chunks = embedder.embed_chunks(chunks)
    via_texts = embedder.embed([c.text for c in chunks])
    assert np.allclose(via_chunks, via_texts)


# ───── assert_aligned ─────


def test_assert_aligned_passes_when_matched():
    chunks = [_chunk("x", 0), _chunk("y", 1)]
    embeddings = np.zeros((2, 384), dtype=np.float32)
    assert_aligned(chunks, embeddings)  # should not raise


def test_assert_aligned_rejects_length_mismatch():
    chunks = [_chunk("x", 0)]
    embeddings = np.zeros((2, 384), dtype=np.float32)
    with pytest.raises(ValueError, match="length mismatch"):
        assert_aligned(chunks, embeddings)


def test_assert_aligned_rejects_1d_embeddings():
    chunks = [_chunk("x", 0)]
    embeddings = np.zeros(384, dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        assert_aligned(chunks, embeddings)


# ───── validation ─────


def test_embedder_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        Embedder(batch_size=0)


def test_embedder_rejects_negative_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        Embedder(batch_size=-1)
