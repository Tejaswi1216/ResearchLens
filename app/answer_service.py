import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def build_context(results):

    context = ""

    for result in results:
        context += (
            f"[Page {result['page_number']}]\n"
            f"{result['text']}\n\n"
        )

    return context


def generate_answer(query, results):

    context = build_context(results)

    prompt = f"""
You are an AI research assistant.

Answer ONLY from the provided context.

Rules:

1. Do NOT use outside knowledge.

2. If the answer is not available in the context, reply exactly:

"The provided paper does not contain enough information."

3. Mention page numbers in your answer.

Question:
{query}

Context:
{context}

Answer:
"""

    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120.0,
    )

    response.raise_for_status()

    return response.json()["response"].strip()