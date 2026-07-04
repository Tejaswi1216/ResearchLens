from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANKER_MODEL)


def rerank_results(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:

    if not candidates:
        return []

    pairs = [
        [query, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    reranked_results = []

    for candidate, score in zip(candidates, scores):
        result = candidate.copy()
        result["reranker_score"] = float(score)

        reranked_results.append(result)

    reranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return reranked_results[:top_k]