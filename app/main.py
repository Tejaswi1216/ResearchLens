from pathlib import Path
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_pages
from app.embedding_service import generate_embeddings
from app.search_service import hybrid_search
from app.answer_service import generate_answer
from app.reranker import rerank_results
from app.paper_store import (
    get_paper_embeddings,
    paper_is_indexed,
    save_paper_embeddings,
)

app = FastAPI(
    title="ResearchLens API",
    description="AI Research Paper Intelligence and Experimentation Agent",
    version="1.0.0",
)


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "ResearchLens API is running"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
        )

    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename,
        "saved_path": str(file_path),
    }
@app.get("/papers/{filename}/extract")
def extract_pdf_text(filename: str):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF not found",
        )

    pages = extract_text_from_pdf(file_path)

    return {
        "filename": filename,
        "total_pages": len(pages),
        "pages": pages,
    }
@app.get("/papers/{filename}/chunks")
def get_pdf_chunks(filename: str):

    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF not found",
        )

    pages = extract_text_from_pdf(file_path)

    chunks = chunk_pages(pages)

    return {
        "filename": filename,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chunks": chunks,
    }
@app.get("/papers/{filename}/embeddings")
def get_pdf_embeddings(filename: str):

    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF not found",
        )

    pages = extract_text_from_pdf(file_path)

    chunks = chunk_pages(pages)

    embedded_chunks = generate_embeddings(chunks)

    embedding_dimension = (
        len(embedded_chunks[0]["embedding"])
        if embedded_chunks
        else 0
    )

    return {
        "filename": filename,
        "total_chunks": len(embedded_chunks),
        "embedding_dimension": embedding_dimension,
        "embedded_chunks": embedded_chunks,
    }
@app.get("/papers/{filename}/search")
def search_paper(
    filename: str,
    query: str,
    top_k: int = 5,
):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF not found",
        )

    embedded_chunks = get_paper_embeddings(filename)

    if embedded_chunks is None:
        raise HTTPException(
            status_code=400,
            detail="Paper is not indexed. Call the index endpoint first.",
        )

    candidates = hybrid_search(
        query=query,
        embedded_chunks=embedded_chunks,
        top_k=10,
    )

    results = rerank_results(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )

    return {
        "filename": filename,
        "query": query,
        "top_k": top_k,
        "results": results,
    }
@app.post("/papers/{filename}/index")
def index_paper(filename: str):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF not found",
        )

    if paper_is_indexed(filename):
        embedded_chunks = get_paper_embeddings(filename)

        return {
            "message": "Paper already indexed",
            "filename": filename,
            "total_chunks": len(embedded_chunks),
        }

    pages = extract_text_from_pdf(file_path)
    chunks = chunk_pages(pages)
    embedded_chunks = generate_embeddings(chunks)

    save_paper_embeddings(
        filename,
        embedded_chunks,
    )

    return {
        "message": "Paper indexed successfully",
        "filename": filename,
        "total_chunks": len(embedded_chunks),
        "embedding_dimension": (
            len(embedded_chunks[0]["embedding"])
            if embedded_chunks
            else 0
        ),
    }
@app.get("/papers/{filename}/ask")
def ask_paper(
    filename: str,
    query: str,
    top_k: int = 5,
):
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF not found",
        )

    embedded_chunks = get_paper_embeddings(filename)

    if embedded_chunks is None:
        raise HTTPException(
            status_code=400,
            detail="Paper is not indexed. Call the index endpoint first.",
        )

    candidates = hybrid_search(
        query=query,
        embedded_chunks=embedded_chunks,
        top_k=10,
    )

    results = rerank_results(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )

    answer = generate_answer(
        query=query,
        results=results,
    )

    return {
        "filename": filename,
        "query": query,
        "answer": answer,
        "sources": [
            {
                "page_number": item["page_number"],
                "chunk_id": item["chunk_id"],
            }
            for item in results
        ],
    }