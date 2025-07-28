# 📄 PDF Insight System - Adobe India Hackathon 2025 (Challenge 1b)

This project processes a collection of PDFs and extracts structured, searchable, and relevant answers for user-defined tasks using a lightweight LLM and LangGraph flow. It is optimized to work offline with local models (TinyLlama or MiniLM) and FAISS-based retrieval.

---

## 🧩 Architecture Overview

The pipeline consists of the following steps:

1. **Reads PDF and extracts structured text with layout info**
   → Extracts fonts, hierarchy, and positions using `pdf_processor()`
2. **Text Chunking**
   → Breaks the content into semantically meaningful sections using `chunker()`
3. **Embedding Generator**
   → Converts chunks into vector representations using TinyLlama or MiniLM
4. **Supabase Vector DB (Optional)**
   → Default: FAISS-based similarity search; can be replaced with Supabase
5. **LangGraph Retrieval**
   → Uses persona + task to retrieve contextually relevant chunks
6. **Ranking & Snippet Extraction**
   → Ranks top chunks and extracts cleaned responses from matching pages
7. **Output JSON Generator**
   → Saves final structured answer in `data/output/`

---

## 📁 Folder Structure

```
challenge1b/
├── ingest.py               # PDF → JSON converter using user-provided persona + task
├── answer.py               # Query → context retrieval → response
├── processors.py           # Core logic: pdf_processor, chunker, embedder, retriever, output_writer
├── langgraph_flow.py       # LangGraph workflow orchestration
│
├── db/
│   └── faiss_store.pkl     # FAISS index file (generated after embedding)
│
├── data/
│   └── collection/
│       ├── input/          # JSONs from ingest.py
│       └── output/         # Final structured answers
│
├── models/
│   ├── all-MiniLM-L6-v2/   # Embedding model (optional)
│   └── tinyllama/          # Quantized model weights (~650MB, optional)
│
├── requirements.txt
├── setup.py
└── README.md               # ← You’re reading it!
```

---

## ⚙️ Setup Instructions

1. **Clone the repo and create directory structure:**

   ```bash
   mkdir -p challenge1b/data/collection/input
   mkdir -p challenge1b/data/collection/output
   mkdir -p challenge1b/models
   mkdir -p challenge1b/db
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup (downloads embedding models):**

   ```bash
   python setup.py
   ```

4. **Place your PDFs** in:

   ```
   challenge1b/data/collection/
   ```

5. **Run the LangGraph flow:**

   ```bash
   python ingest.py
   python answer.py --interactive
   OR
   python langgraph_flow.py
   ```

---

## 🧪 Test Flow Description

For each PDF, the system expects:

* A **persona** (e.g., legal analyst, content writer)
* A **job to be done** (e.g., summarize case law, extract clauses)

These are passed through `ingest.py`, which generates:

* `test_case_name`: auto-generated based on persona/task
* `description`: high-level summary of the task
* `title`: extracted from the PDF (usually H1 or large-font heading)

The LangGraph engine then uses this context to:

* Retrieve matching chunks
* Rank and clean them
* Store the answer in `data/collection/output/`

---

## 📌 Notes

* Fully **offline**: uses FAISS, local models like TinyLlama or MiniLM
* Can handle **5 PDFs in under 1 minute** (on average machine)
* No external API calls (suitable for hackathon with offline-only constraints)
* Modular and extendable with LangGraph and persona-driven workflows
