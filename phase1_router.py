"""Phase 1: Vector-Based Persona Matching (The Router).

Uses a lightweight pure-NumPy TF-IDF vectorizer to compute cosine similarities
between incoming posts and bot persona descriptions.
"""
import numpy as np
from typing import List, Dict

from config import BOT_PERSONAS, SIMILARITY_THRESHOLD
from embeddings import embed_text, get_persona_vectors, cosine_similarity


def route_post_to_bots(post_content: str, threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
    """Embed a post and return all bots whose persona similarity exceeds the threshold.

    Args:
        post_content: The text of the incoming post.
        threshold: Minimum cosine similarity to consider a bot "interested".

    Returns:
        List of dicts with keys: bot_id, name, similarity.
    """
    post_vec = embed_text(post_content)
    persona_ids, persona_vectors = get_persona_vectors()

    matches = []
    for bot_id, p_vec in zip(persona_ids, persona_vectors):
        sim = cosine_similarity(post_vec, p_vec)
        if sim >= threshold:
            matches.append(
                {
                    "bot_id": bot_id,
                    "name": BOT_PERSONAS[bot_id]["name"],
                    "similarity": round(sim, 4),
                }
            )

    # Sort descending by similarity
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


if __name__ == "__main__":
    # Quick sanity check
    post = "OpenAI just released a new model that might replace junior developers."
    print(f"\n[Phase 1 Test] Post: {post}")
    matched = route_post_to_bots(post, threshold=0.75)
    print("Matched bots:", matched)

