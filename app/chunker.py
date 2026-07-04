import re


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1200,
    overlap_sentences: int = 2,
) -> list[dict]:

    chunks = []
    chunk_id = 0

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        sentences = split_into_sentences(text)

        current_sentences = []
        current_length = 0

        for sentence in sentences:

            sentence_length = len(sentence)

            if (
                current_sentences
                and current_length + sentence_length > chunk_size
            ):
                chunk_text = " ".join(current_sentences)

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page_number": page_number,
                        "text": chunk_text,
                    }
                )

                chunk_id += 1

                current_sentences = current_sentences[
                    -overlap_sentences:
                ]

                current_length = sum(
                    len(item)
                    for item in current_sentences
                )

            current_sentences.append(sentence)
            current_length += sentence_length

        if current_sentences:

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": " ".join(current_sentences),
                }
            )

            chunk_id += 1

    return chunks