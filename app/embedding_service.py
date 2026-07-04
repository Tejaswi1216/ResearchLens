from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode_document(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "embedding": embedding.tolist(),
            }
        )

    return embedded_chunks