from pathlib import Path
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ─── Define the system variables here ───────────────────────────────────────────────
SOURCE_DIR = Path("./source_documents")
CACHE_DIR   = Path("./.cache")
DOC_CACHE   = CACHE_DIR / "docling_docs.pkl"
INDEX_CACHE = CACHE_DIR / "faiss_index"
EMBEDDER    = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-30m-english")

def load_cached_docs():
    return pickle.loads(DOC_CACHE.read_bytes()) if DOC_CACHE.exists() else {}

def save_cached_docs(docs):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_CACHE.write_bytes(pickle.dumps(docs))

def load_faiss_index():
    if not INDEX_CACHE.exists():
        return None
    return FAISS.load_local(str(INDEX_CACHE), EMBEDDER, allow_dangerous_deserialization=True)

def save_faiss_index(idx):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    idx.save_local(str(INDEX_CACHE))
