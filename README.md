# 🧠 Multimodal RAG System

This repository contains a lightweight, modular Retrieval-Augmented Generation (RAG) system that supports **multimodal PDFs** containing text, tables, and images. It allows users to upload academic or research PDFs, extract and index their content, and query the documents using a language model grounded in retrieved evidence.

---

## ✨ Features

- ✅ Handles text, table (markdown), and image (caption + visual summary) content
- ⚡ Converts PDF content into 384-dimensional dense embeddings (IBM Granite)
- 📁 Stores and retrieves documents using FAISS for fast semantic search
- 🧠 Integrates a language model to generate grounded answers with citations
- 💾 Uses metadata caching to avoid redundant reprocessing
- 🖥️ Includes a Gradio-based UI for uploading PDFs and asking questions

---

## 📁 Project Structure

```
.
├── source_documents/        # Place your PDFs here
├── rag/                     # Core logic
│   ├── cache.py             # Load/save FAISS index and document cache
│   ├── ingestion.py         # Chunking, embedding, multimodal processing
│   ├── models.py            # LLMs, embedding model, tokenizer setup
│   ├── pipeline.py          # RAGPipeline class: wraps ingestion + querying
│   └── query.py             # Prompt templates and retrieval logic
├── app.py                   # Gradio interface
├── initial_ingest.py        # CLI script to ingest all PDFs in batch
├── requirements.txt         # Python dependencies
└── .env.template            # Example environment file (e.g., Replicate API token)
```

---

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag
```

2. **Set up your environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure environment variables**

Copy the `.env.template` file to `.env` and insert your Replicate API key or any other required secrets:
```bash
cp .env.template .env
```

---

## 🚀 Usage

### 1. Ingest all PDFs from `source_documents/`
```bash
python initial_ingest.py
```

### 2. Launch the Gradio UI
```bash
python app.py
```
- Upload additional PDFs
- Ask natural language questions about the content
- Get LLM answers with proper citations

---

## 🧪 Evaluation

Evaluation scripts (e.g. `evaluate.py`) can be used to benchmark system performance using ROUGE and cosine similarity metrics against reference answers.

---

## 📄 License

This project is released under the MIT License.

---

## 🤝 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)  
- [IBM Granite Models](https://huggingface.co/ibm-granite)  
- [Replicate](https://replicate.com)  
- [Gradio](https://www.gradio.app/)