# initial_ingest.py
"""Only to be called for the first time to chunk the original 8 PDFs via CLI"""
from pathlib import Path
from rag.pipeline import RAGPipeline
from rag.ingestion import ingest_new_pdfs

if __name__=="__main__":
    pipe = RAGPipeline(Path("./source_documents"))
    # ingest all at once
    _, _, processed = ingest_new_pdfs(pipe.index, pipe.cached_docs)
    print(f"✅ Embedded: {', '.join(processed)}" if processed else "⚠️ No new PDFs")

