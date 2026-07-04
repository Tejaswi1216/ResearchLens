import re

import numpy as np

from app.embedding_service import model


def tokenize(text: str) -> set[str]:
    return set(
        re.findall(
            r"\b[a-zA-Z0-9\-]+\b",
            text.lower(),
        )
    )


def keyword_score(
    query: str,
    document: str,
) -> float:

    query_tokens = tokenize(query)
    document_tokens = tokenize(document)

    if not query_tokens:
        return 0.0

    matched_tokens = query_tokens.intersection(
        document_tokens
    )

    return len(matched_tokens) / len(query_tokens)


def hybrid_search(
    query: str,
    embedded_chunks: list[dict],
    top_k: int = 5,
    semantic_weight: float = 0.8,
    keyword_weight: float = 0.2,
) -> list[dict]:

    query_embedding = model.encode_query(
        query,
        normalize_embeddings=True,
    )

    results = []

    for chunk in embedded_chunks:

        chunk_embedding = np.array(
            chunk["embedding"],
            dtype=np.float32,
        )

        semantic_score = float(
            np.dot(
                query_embedding,
                chunk_embedding,
            )
        )

        lexical_score = keyword_score(
            query,
            chunk["text"],
        )

        final_score = (
            semantic_weight * semantic_score
            + keyword_weight * lexical_score
        )

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "semantic_score": semantic_score,
                "keyword_score": lexical_score,
                "final_score": final_score,
            }
        )

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return results[:top_k]