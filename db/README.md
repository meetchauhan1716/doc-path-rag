# ChromaDB Folder

The `chromadb` folder is where all processed markdown files are stored in the form of **vector embeddings with metadata**. These embeddings make it possible to **retrieve and search information** quickly and accurately.

---

## What is ChromaDB?
[ChromaDB](https://docs.trychroma.com/) is an open-source **vector database** designed for storing and retrieving embeddings (numerical representations of text, images, or other data).  
It is commonly used in AI/ML applications like **RAG (Retrieval-Augmented Generation)**, semantic search, and chatbots.  

Key features of ChromaDB:
- Stores documents as **vectors** (embeddings).
- Supports **rich metadata** for better filtering and context.
- Enables **fast similarity search** for finding relevant data.
- Easy to integrate with **LLMs (Large Language Models)**.

---

## Usage in This Project
- When markdown files are ingested, they are **converted into embeddings** with metadata.  
- These embeddings are stored inside the `chromadb` folder.  
- Later, they can be **retrieved for question answering, semantic search, or PathRAG workflows**.

---

## Example Workflow
1. Place markdown files in the `data` folder.
2. Run the ingestion script → files are processed into embeddings.
3. Embeddings + metadata are stored in `chromadb`.
4. During retrieval, queries are matched against stored embeddings to fetch the most relevant context.

---

## Why Use ChromaDB?
ChromaDB ensures that the system:
- Retrieves the **most relevant context** for your query.
- Reduces **hallucinations** in LLMs.
- Provides a **structured and scalable** way to store document knowledge.

