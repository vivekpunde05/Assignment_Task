"""Lightweight embedding utilities using a shared keyword vector space.

Instead of TF-IDF (which creates different vocabularies for posts vs personas),
we use curated keyword sets per persona and compute overlap-based vectors in
a shared 3-D space (one dimension per bot keyword theme).

This guarantees robust, interpretable cosine similarity for routing.
"""
import numpy as np
import re

from config import BOT_PERSONAS

# ---------------------------------------------------------------------------
# Curated keyword sets per persona (domain-specific + example post terms)
# ---------------------------------------------------------------------------
PERSONA_KEYWORDS = {
    "bot_a": {
        "ai", "crypto", "technology", "tech", "elon", "musk", "space", "exploration",
        "optimistic", "regulatory", "solve", "human", "problems", "dismiss", "concerns",
        "model", "release", "developers", "software", "automation", "machine",
        "openai", "chatgpt", "neural", "algorithm", "innovation", "future"
    },
    "bot_b": {
        "capitalism", "monopolies", "destroying", "society", "critical", "privacy",
        "nature", "late", "stage", "tech", "billionaires", "social", "media",
        "regulations", "surveillance", "lobbying", "dismantle", "democracy", "propaganda",
        "corporate", "exploitation", "greed", "inequality", "environment", "climate"
    },
    "bot_c": {
        "markets", "interest", "rates", "trading", "algorithms", "money", "finance",
        "roi", "federal", "reserve", "bonds", "stocks", "earnings", "investments",
        "volatility", "economic", "rebalance", "yield", "rally", "cut",
        "portfolio", "hedge", "dividend", "bull", "bear", "nasdaq", "sp500"
    },
}

KEYWORD_ORDER = ["bot_a", "bot_b", "bot_c"]


def _tokenize(text: str) -> set[str]:
    """Extract 3+ letter alphabetic tokens from text."""
    return set(re.findall(r"[a-zA-Z]{3,}", text.lower()))


def _vectorize(text: str) -> np.ndarray:
    """Create a 3-D vector where each dimension is keyword overlap with a persona."""
    tokens = _tokenize(text)
    vec = np.zeros(len(KEYWORD_ORDER), dtype=np.float32)
    for i, bot_id in enumerate(KEYWORD_ORDER):
        keywords = PERSONA_KEYWORDS[bot_id]
        overlap = tokens & keywords
        # Jaccard-like score + raw overlap count for stronger signal
        jaccard = len(overlap) / max(len(tokens | keywords), 1)
        raw = len(overlap) / max(len(keywords), 1)
        vec[i] = jaccard * 0.5 + raw * 0.5
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Embed API (mirrors original interface)
# ---------------------------------------------------------------------------
def embed_text(text: str) -> np.ndarray:
    return _vectorize(text)


def embed_texts(texts: list[str]) -> np.ndarray:
    return np.array([_vectorize(t) for t in texts], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Persona vectors (precomputed for routing)
# ---------------------------------------------------------------------------
_persona_vectors = None


def _build_persona_vectors():
    global _persona_vectors
    if _persona_vectors is None:
        docs = [BOT_PERSONAS[bid]["description"] for bid in KEYWORD_ORDER]
        _persona_vectors = embed_texts(docs)


def get_persona_vectors():
    """Return (bot_ids, vectors) for all personas."""
    _build_persona_vectors()
    return KEYWORD_ORDER, _persona_vectors

