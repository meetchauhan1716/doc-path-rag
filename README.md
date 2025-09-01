# Project Overview: doc-path-rag

This project implements a **Retrieval-Augmented Generation (RAG)** system for software documentation using markdown files. It leverages **ChromaDB** for vector storage and retrieval, and integrates with **Ollama** models for embeddings and LLM responses.

## Folder Structure

- `data/` — Store your markdown files here. Supports nested folders.
- `db/chroma_langchain_db/` — ChromaDB vector store (auto-generated).
- `file_list.txt` — List of all markdown files found in `data/`.
- `markdown.txt` — Example output of markdown chunking.
- `prompt.yaml` — YAML configuration for prompt templates.
- Python scripts:
  - `ingest.py`
  - `bot.py`
  - `test-markdown-chunk.py`
  - `test-retrival-chunk.py`
  - `models.py`

---

## Main Python Scripts

### 1. `ingest.py` — Markdown Ingestion & Chunking

- **Purpose:** Scans the `data/` folder for markdown files, chunks them by headers, adds rich metadata, and stores them as vector embeddings in ChromaDB.
- **How it works:**
  - Finds all markdown files (recursively).
  - Chunks each file by header hierarchy.
  - Adds metadata (header path, content type, etc.).
  - Stores chunks in ChromaDB for fast retrieval.
  - Marks processed files by renaming (adds `_` prefix).
  - Can run once or in continuous monitoring mode.

**Use this script first to ingest your documentation.**

---

### 2. `bot.py` — Intelligent Document Retrieval & Q&A

- **Purpose:** Provides a command-line chatbot interface for querying the ingested documentation.
- **How it works:**
  - Loads prompt configuration from `prompt.yaml`.
  - Initializes Ollama models and ChromaDB retriever.
  - Supports switching between prompt modes (default, API, architecture).
  - Retrieves relevant chunks and answers user queries using LLM.
  - Displays sources and metadata for each answer.

**Use this script to interact with your documentation and get answers.**

---

### 3. `test-markdown-chunk.py` — Markdown Chunking Debug Tool

- **Purpose:** Standalone tool to chunk a single markdown file and inspect the results.
- **How it works:**
  - Chunks a specified markdown file.
  - Prints chunk summary and metadata.
  - Saves output to a text file for inspection.

**Use for debugging or inspecting chunking logic.**

---

### 4. `test-retrival-chunk.py` — Retrieval Debug Tool

- **Purpose:** Standalone tool to test document retrieval from ChromaDB.
- **How it works:**
  - Loads prompts and initializes models.
  - Retrieves relevant chunks for a user query.
  - Prints formatted context and metadata.

**Use for debugging retrieval and context formatting.**

---

### 5. `models.py` — Model Initialization

- **Purpose:** Centralizes Ollama model and embedding initialization.
- **How it works:**
  - Loads embedding and LLM models from environment variables or defaults.

**Used by all main scripts for model access.**

---

## Workflow

1. **Add markdown files** to the `data/` folder.
2. **Run `ingest.py`** to process and store files in ChromaDB.
3. **Query your documentation** using `bot.py` (or debug with the test scripts).

---

## Prompt Configuration

- **`prompt.yaml`** defines system instructions, response formats, and specialized prompt modes (API, architecture).
- Ensures answers are accurate, concise, and always cite sources.

---

## Which Script to Use

- **For ingestion:** `ingest.py`
- **For Q&A/chatbot:** `bot.py`
- **For debugging chunking:** `test-markdown-chunk.py`
- **For debugging retrieval:** `test-retrival-chunk.py`

---

## Example Usage

```bash
# Ingest all markdown files
python ingest.py

# Start the chatbot
python bot.py
```

---

## Data & DB Folders

- See `data/README.md` and `db/README.md` for details on how files and embeddings are managed.

---

Let me know if you want this saved to a markdown file or need a diagram!
