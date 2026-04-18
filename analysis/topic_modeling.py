"""
analysis/topic_modeling.py — Latent Dirichlet Allocation on text corpora.

Implements the topic-modelling pipeline from Jansen, *Machine Learning
for Algorithmic Trading* (2nd ed.) Chapter 15: build a bag-of-words
representation of a news / filing corpus, fit ``LatentDirichletAllocation``
over a small number of topics, and project new documents onto that
topic basis.  Per-ticker topic distributions become alpha features
(or sentiment-blender inputs) without dragging in heavy NLP deps —
``sklearn`` is already pinned in ``requirements.txt``.

Public surface
--------------
``LDAResult``                — dataclass bundling the fitted vectorizer + LDA
``fit_lda(docs, n_topics)``  — train on a corpus
``infer_topics(result, docs)`` — project documents to topic-distribution rows
``top_terms_per_topic(result, n)`` — interpretability helper
``ticker_topic_distribution(result, docs_by_ticker)`` — per-ticker feature row

Reference
---------
    Jansen, *Machine Learning for Algorithmic Trading* (2nd ed.) Ch 15.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

try:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    _SKLEARN_AVAILABLE = True
except ImportError:
    LatentDirichletAllocation = None  # type: ignore[assignment,misc]
    CountVectorizer = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False


@dataclass(frozen=True)
class LDAResult:
    """Fitted LDA artefacts.  Keep ``vectorizer`` and ``model`` together
    so :func:`infer_topics` can apply the same vocabulary to new docs."""

    vectorizer: object   # CountVectorizer
    model: object        # LatentDirichletAllocation
    n_topics: int


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError(
            "scikit-learn is not installed. Run: pip install scikit-learn>=1.3.0"
        )


def fit_lda(
    documents: Sequence[str],
    n_topics: int = 10,
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.95,
    random_state: int = 42,
    max_iter: int = 20,
) -> LDAResult:
    """Fit an LDA model over a bag-of-words representation of ``documents``.

    Parameters
    ----------
    documents :
        Iterable of raw text documents (one per news item / filing /
        ticker-day, depending on the caller's granularity).
    n_topics :
        Number of latent topics to learn.
    max_features :
        Vocabulary size cap passed to :class:`CountVectorizer`.
    min_df, max_df :
        Document-frequency bounds for vocabulary pruning.
    random_state, max_iter :
        Forwarded to :class:`LatentDirichletAllocation`.

    Returns
    -------
    :class:`LDAResult` containing the fitted vectorizer + model.

    Raises
    ------
    ValueError
        If ``documents`` is empty or the vocabulary collapses to nothing
        after pruning (e.g. all-stopword corpus).
    """
    _require_sklearn()

    docs = [d for d in documents if d and d.strip()]
    if not docs:
        raise ValueError("fit_lda: no non-empty documents")

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
    )
    try:
        X = vectorizer.fit_transform(docs)
    except ValueError as exc:
        raise ValueError(f"fit_lda: vectorisation failed ({exc})") from exc

    if X.shape[1] == 0:
        raise ValueError("fit_lda: empty vocabulary after pruning")

    model = LatentDirichletAllocation(
        n_components=int(n_topics),
        random_state=random_state,
        max_iter=max_iter,
        learning_method="batch",
    )
    model.fit(X)

    log.info(
        "topic_modeling.fit_lda complete",
        n_docs=len(docs), vocab=X.shape[1], n_topics=n_topics,
    )
    return LDAResult(vectorizer=vectorizer, model=model, n_topics=int(n_topics))


def infer_topics(result: LDAResult, documents: Sequence[str]) -> np.ndarray:
    """Project ``documents`` onto the fitted topic basis.

    Returns a ``(len(documents), n_topics)`` array whose rows sum to
    ``1`` (LDA topic distributions). Empty / whitespace-only documents
    yield a uniform ``1/n_topics`` row so callers always get a row per
    input.
    """
    _require_sklearn()
    if not documents:
        return np.zeros((0, result.n_topics), dtype=float)

    cleaned: list[str] = []
    placeholder_idx: list[int] = []
    for i, d in enumerate(documents):
        if d and d.strip():
            cleaned.append(d)
        else:
            placeholder_idx.append(i)
            cleaned.append("")  # keep positional alignment

    out = np.full((len(documents), result.n_topics),
                  fill_value=1.0 / result.n_topics, dtype=float)

    real_idx = [i for i in range(len(documents)) if i not in set(placeholder_idx)]
    if real_idx:
        X = result.vectorizer.transform([cleaned[i] for i in real_idx])
        if X.nnz == 0:
            # No known vocabulary at all → leave uniform rows.
            return out
        scores = result.model.transform(X)  # already row-normalised
        for k, i in enumerate(real_idx):
            out[i] = scores[k]
    return out


def top_terms_per_topic(result: LDAResult, n: int = 10) -> list[list[str]]:
    """Return the ``n`` highest-weight vocabulary terms for each topic.

    Useful for human-readable topic labels / dashboard tooltips.
    """
    _require_sklearn()
    vocab = np.array(result.vectorizer.get_feature_names_out())
    components = np.asarray(result.model.components_)
    out: list[list[str]] = []
    for row in components:
        idx = np.argsort(row)[::-1][:n]
        out.append([str(t) for t in vocab[idx]])
    return out


def ticker_topic_distribution(
    result: LDAResult,
    docs_by_ticker: Mapping[str, Sequence[str]],
) -> dict[str, np.ndarray]:
    """Aggregate per-ticker topic exposure.

    For each ticker the per-document topic distributions are averaged
    into a single ``(n_topics,)`` vector summing to ``1``. Tickers with
    no usable documents get a uniform ``1/n_topics`` row.
    """
    out: dict[str, np.ndarray] = {}
    uniform = np.full(result.n_topics, 1.0 / result.n_topics)
    for ticker, docs in docs_by_ticker.items():
        if not docs:
            out[ticker] = uniform.copy()
            continue
        per_doc = infer_topics(result, list(docs))
        out[ticker] = per_doc.mean(axis=0) if per_doc.size else uniform.copy()
    return out
