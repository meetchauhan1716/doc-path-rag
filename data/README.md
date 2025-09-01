# Data Folder

The `data` folder is used to store markdown (`.md`) files that will be processed by the system.

## Features
- You can add markdown files directly inside the `data` folder.
- You can also organize files inside **nested folders**.
- The system will **automatically scan** all files (including nested ones) and store their content in **ChromaDB** with embeddings and metadata.

## Usage
1. Place your markdown files in the `data` folder (or inside subfolders).
2. Run the ingestion script.
3. All files will be processed and saved in ChromaDB for retrieval.

## Example Structure
```

data/
│── guide.md
│── notes.md
└── tutorials/
├── setup.md
└── advanced.md

```

This makes it easy to manage and organize documents for **PathRAG** while ensuring everything is indexed automatically.

