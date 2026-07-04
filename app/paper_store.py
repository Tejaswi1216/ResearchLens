import json
from pathlib import Path


INDEX_DIR = Path("data/indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _get_index_path(filename: str) -> Path:
    safe_name = filename.replace(" ", "_").replace(".pdf", "")
    return INDEX_DIR / f"{safe_name}.json"


def save_paper_embeddings(
    filename: str,
    embedded_chunks: list[dict],
) -> None:

    index_path = _get_index_path(filename)

    with index_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            embedded_chunks,
            file,
            ensure_ascii=False,
        )


def get_paper_embeddings(
    filename: str,
) -> list[dict] | None:

    index_path = _get_index_path(filename)

    if not index_path.exists():
        return None

    with index_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def paper_is_indexed(filename: str) -> bool:
    return _get_index_path(filename).exists()