"""
analysis/word_embeddings.py — word2vec embeddings for text corpora.

Implements the embedding-training pipeline from Jansen, *Machine
Learning for Algorithmic Trading* (2nd ed.) Chapter 16: train a
``word2vec`` model on a corpus (earnings calls / 10-K risk-factor
sections / news headlines) and project new documents to a fixed-size
vector by averaging the word vectors of the tokens that appear in the
vocabulary.

``gensim`` is the canonical implementation in the book.  We import it
through a try/except gate so the rest of the platform keeps working on
machines that don't have it installed (mirrors the torch gating in
:mod:`strategies.dl_signal`).

Public surface
--------------
``EmbeddingResult``      — dataclass bundling the trained Word2Vec model
``train_embeddings``     — fit on a list of raw text documents
``document_embedding``   — average-pool word vectors for one document
``nearest_terms``        — top-k closest vocabulary terms to a given term

Reference
---------
    Jansen, *Machine Learning for Algorithmic Trading* (2nd ed.) Ch 16.4.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

try:
    from gensim.models import Word2Vec  # type: ignore[import]
    _GENSIM_AVAILABLE = True
except ImportError:
    Word2Vec = None  # type: ignore[assignment,misc]
    _GENSIM_AVAILABLE = False


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")


def _tokenise(text: str) -> list[str]:
    """Lowercase + alphanumeric tokenisation; drops 1-character tokens."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class EmbeddingResult:
    """Trained word2vec artefacts."""

    model: object              # gensim.models.Word2Vec
    vector_size: int
    vocab_size: int


def _require_gensim() -> None:
    if not _GENSIM_AVAILABLE:
        raise RuntimeError(
            "gensim is not installed. Run: pip install 'gensim>=4.3.0'"
        )


def train_embeddings(
    documents: Sequence[str],
    vector_size: int = 64,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 10,
    seed: int = 42,
    workers: int = 1,
) -> EmbeddingResult:
    """Fit a word2vec (skip-gram) model on the supplied corpus.

    Parameters
    ----------
    documents :
        Iterable of raw text documents. Each is tokenised on-the-fly.
    vector_size :
        Dimensionality of the embedding space.
    window :
        Context window (words) used during training.
    min_count :
        Minimum corpus frequency for a token to enter the vocabulary.
    epochs :
        Training epochs.
    seed, workers :
        Forwarded to :class:`gensim.models.Word2Vec`. ``workers=1``
        keeps the run deterministic when ``seed`` is set.

    Raises
    ------
    RuntimeError
        If ``gensim`` is not installed.
    ValueError
        If the corpus is empty or the vocabulary collapses to nothing
        after pruning.
    """
    _require_gensim()

    sentences = [_tokenise(d) for d in documents if d and d.strip()]
    sentences = [s for s in sentences if s]
    if not sentences:
        raise ValueError("train_embeddings: no usable documents after tokenisation")

    model = Word2Vec(
        sentences=sentences,
        vector_size=int(vector_size),
        window=int(window),
        min_count=int(min_count),
        epochs=int(epochs),
        seed=int(seed),
        workers=int(workers),
        sg=1,  # skip-gram (Jansen Ch 16.4.2)
    )
    vocab_size = len(model.wv)
    if vocab_size == 0:
        raise ValueError("train_embeddings: empty vocabulary after min_count pruning")

    log.info(
        "word_embeddings.train complete",
        n_docs=len(sentences), vocab=vocab_size, vector_size=vector_size,
    )
    return EmbeddingResult(model=model, vector_size=int(vector_size), vocab_size=vocab_size)


def document_embedding(result: EmbeddingResult, document: str) -> np.ndarray:
    """Average-pool the trained word vectors over a document's tokens.

    Returns a ``(vector_size,)`` array; the zero vector when no token
    overlaps the trained vocabulary (so callers always get a well-formed
    feature even for cold-start documents).
    """
    _require_gensim()
    tokens = _tokenise(document or "")
    wv = result.model.wv  # type: ignore[attr-defined]
    vectors = [wv[t] for t in tokens if t in wv]
    if not vectors:
        return np.zeros(result.vector_size, dtype=float)
    return np.mean(np.stack(vectors), axis=0).astype(float)


def nearest_terms(
    result: EmbeddingResult,
    term: str,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Top-``k`` vocabulary terms closest to ``term`` by cosine similarity.

    Returns an empty list when ``term`` is out of vocabulary.
    """
    _require_gensim()
    wv = result.model.wv  # type: ignore[attr-defined]
    token = term.lower()
    if token not in wv:
        return []
    return [(str(t), float(s)) for t, s in wv.most_similar(token, topn=int(k))]
