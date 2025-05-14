from pathlib import Path
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoProcessor
from ibm_granite_community.notebook_utils import get_env_var
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Replicate

# ─── Get the API key from .env file ───────────────────────────────────────
load_dotenv()
REPLICATE_TOKEN = get_env_var("REPLICATE_API_TOKEN")

# ─── Model identifiers ─────────────────────────────────────
EMBED_MODEL  = "ibm-granite/granite-embedding-30m-english"
LLM_MODEL    = "ibm-granite/granite-3.2-8b-instruct"
VISION_MODEL = "ibm-granite/granite-vision-3.2-2b"

# ─── Embeddings & Tokenizer ─────────────────────────────────
embedder  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

# ─── Text LLM for getting table summaries and final answer generation ────────────────────────────────────────
text_llm = Replicate(
    model=LLM_MODEL,
    replicate_api_token=REPLICATE_TOKEN,
    model_kwargs={"max_tokens": 1000, "min_tokens": 100},
)

# ─── Vision LLM for getting image summaries ─────────────────
vision_processor = AutoProcessor.from_pretrained(VISION_MODEL)
vision_llm = Replicate(
    model=VISION_MODEL,
    replicate_api_token=REPLICATE_TOKEN,
    model_kwargs={"max_tokens": 200},
)
