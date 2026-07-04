import re
from pathlib import Path

from pypdf import PdfReader


def clean_text(text: str) -> str:
    # Join words split across lines with hyphens
    text = re.sub(r"-\n(?=\w)", "", text)

    # Replace remaining newlines with spaces
    text = text.replace("\n", " ")

    # Remove isolated line-number artifacts
    text = re.sub(r"(?<=\D)\d+(?=\s)", " ", text)

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_text_from_pdf(file_path: Path) -> list[dict]:
    reader = PdfReader(str(file_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_text(raw_text)

        pages.append(
            {
                "page_number": page_number,
                "text": cleaned_text,
            }
        )

    return pages