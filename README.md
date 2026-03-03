# Chat with PDF 🤖📄

A lightweight Streamlit app that lets you **chat with the contents of a PDF** using OpenAI embeddings + chat completion.

Upload a PDF, ask a question, and the app retrieves relevant chunks from the document before generating an answer.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start (Local)](#quick-start-local)
- [Configuration](#configuration)
- [Run with Docker](#run-with-docker)
- [Usage](#usage)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

---

## Features

- 📄 Upload and parse PDF files in the browser.
- 🧩 Split extracted text into chunks for retrieval.
- 🧠 Create OpenAI embeddings for document chunks.
- 🔎 Retrieve top-k relevant chunks via cosine similarity.
- 💬 Ask questions in a simple Streamlit chat-style interface.
- 🔐 Supports API key via environment variable or sidebar input.

---

## How It Works

1. **Ingestion** (`ingest.py`)
   - Reads PDF pages with `PyPDF2`.
   - Splits text into fixed-size chunks (`CHUNK_SIZE = 500` characters).
   - Generates embeddings for each chunk using OpenAI.

2. **Retrieval** (`ingest.py`)
   - Embeds the user question.
   - Computes cosine similarity between question embedding and chunk embeddings.
   - Returns the most relevant chunks.

3. **Answering** (`chatbot_app.py`)
   - Builds a prompt with retrieved context + user question.
   - Sends it to OpenAI ChatCompletion (`gpt-3.5-turbo`).
   - Displays response and chat history in Streamlit.

---

## Project Structure

```text
.
├── chatbot_app.py      # Streamlit UI and chat loop
├── ingest.py           # PDF parsing, chunking, embeddings, similarity search
├── constants.py        # Reads OPENAI_API_KEY from environment
├── requirements.txt    # Python dependencies
├── Dockerfile          # Containerized app runtime
└── README.md
```

---

## Prerequisites

- Python **3.10+**
- An OpenAI API key

---

## Quick Start (Local)

```bash
# 1) Clone
git clone https://github.com/kartikahlawat/Chat-with-PDF.git
cd Chat-with-PDF

# 2) (Optional) create & activate virtual env
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Set API key (optional if entering in sidebar)
export OPENAI_API_KEY="your_api_key_here"  # Windows PowerShell: $env:OPENAI_API_KEY="..."

# 5) Run app
streamlit run chatbot_app.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

---

## Configuration

The app checks for an API key in two places:

1. **Sidebar input** (runtime override)
2. `OPENAI_API_KEY` environment variable from `constants.py`

If neither is set, requests to OpenAI will fail.

---

## Run with Docker

```bash
# Build image
docker build -t chat-with-pdf .

# Run container
docker run --rm -p 8501:8501 \
  -e OPENAI_API_KEY="your_api_key_here" \
  chat-with-pdf
```

Then visit: `http://localhost:8501`

---

## Usage

1. Start the app.
2. Enter API key in sidebar (or rely on env var).
3. Upload a PDF.
4. Ask questions like:
   - “What are the main conclusions?”
   - “Summarize page 2.”
   - “What does the document say about X?”

---

## Known Limitations

- Uses fixed-size character chunking (no semantic splitting).
- Embeddings are generated per chunk request (can be slow/costly for large PDFs).
- No persistent vector database (embeddings live in memory per session).
- Uses legacy OpenAI API calls (`openai.Embedding.create`, `openai.ChatCompletion.create`).
- Minimal citation/grounding controls in responses.

---

## Troubleshooting

- **`Please enter your OpenAI API key`**
  - Add key in sidebar or set `OPENAI_API_KEY` before launch.

- **PDF uploads but answers are poor**
  - Try smaller, text-based PDFs.
  - OCR may be needed for scanned/image-only PDFs.

- **Import/dependency issues**
  - Recreate venv and reinstall:
    ```bash
    rm -rf .venv
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

---

## Future Improvements

- Migrate to latest OpenAI Python SDK patterns.
- Add token-aware chunking + overlap.
- Store embeddings in a vector DB (FAISS/Chroma/Pinecone).
- Add citations (page numbers) in final answers.
- Add multi-PDF support and conversation memory controls.

---

If you find this useful, feel free to open issues or contribute improvements.
