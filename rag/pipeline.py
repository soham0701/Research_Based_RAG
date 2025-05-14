from pathlib import Path
from .cache import load_cached_docs, load_faiss_index
from .ingestion import ingest_new_pdfs
from .query import query_all_docs

# ─── Pipeline class ───────────────────────────────────────────────
class RAGPipeline:
    """This class loads cache documents and faiss index and handles the ingestion of new PDFs"""
    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.source_dir.mkdir(exist_ok=True)
        self.cached_docs = load_cached_docs()  #load docling docs from cache
        self.index       = load_faiss_index()  #load index from cache

    def uploaded_pdf(self, pdf_path: Path) -> str:
        if not pdf_path.exists():
            return f"❗ Not found: {pdf_path.name}"
        dest = self.source_dir / pdf_path.name
        if not dest.exists():
            dest.write_bytes(pdf_path.read_bytes())
        self.index, self.cached_docs, new = ingest_new_pdfs(self.index, self.cached_docs)   # if the new variable is not returned, it means that file is already processed
        return f"✅ Embedded: {', '.join(new)}" if new else f"ℹ️ Already: {pdf_path.name}"

    def query(self, query: str) -> str:
        if not query.strip():
            return "❗ Please enter a question."   #ensures that there is a question that is asked
        return query_all_docs(query)
