from typing import List, Tuple
import io

import openai
import numpy as np
from PyPDF2 import PdfReader

from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 500  # characters

class PDFChunk:
    def __init__(self, page_content: str, page_num: int):
        self.page_content = page_content
        self.page_num = page_num

def ingest_pdf(file) -> Tuple[List[PDFChunk], np.ndarray]:
    pdf_reader = PdfReader(file)
    docs = []
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if not text:
            continue
        # Simple chunking by character count
        for start in range(0, len(text), CHUNK_SIZE):
            chunk_text = text[start:start + CHUNK_SIZE]
            docs.append(PDFChunk(chunk_text, i + 1))
    # Get embeddings for all chunks
    embeddings = get_embeddings([chunk.page_content for chunk in docs])
    return docs, embeddings

def get_embeddings(texts: List[str]) -> np.ndarray:
    # Use OpenAI's embedding endpoint
    results = []
    for text in texts:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response["data"][0]["embedding"]
        results.append(embedding)
    return np.array(results)

def search_similarity(query: str, docs: List[PDFChunk], embeddings: np.ndarray, top_k: int = 3) -> List[PDFChunk]:
    # Get embedding for query
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    query_emb = np.array(response["data"][0]["embedding"]).reshape(1, -1)
    sims = cosine_similarity(embeddings, query_emb).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [docs[i] for i in top_idx]