from typing import List
from .cache import load_faiss_index
from .models import text_llm
from langchain.prompts import PromptTemplate

# ─── System + User Prompt─────────────────────────────
SYSTEM_INSTRUCTIONS = """
You are an expert research assistant that answers questions and cites sources only at the end of your answer. 
"""

PROMPT = PromptTemplate(
    input_variables=["context_with_meta", "question"],
    template=f"""{SYSTEM_INSTRUCTIONS}

{{context_with_meta}}

QUESTION:
{{question}}

TASK:
1. Answer the given user QUESTION accuractely and relevantly using ONLY the CONTEXT provided.
2. Only at the end of your complete answer, cite the sources (that are given to you with the CONTEXT).
3. For the citations, inlcude ONLY the file name and the page number(s).
4. If you cannot find an answer, say “Unknown”.
5. If the answer is "Unknown", DO NOT CITE ANY SOURCES AT ALL


------Output Format------
Answer: 
Sources:
"""
)

def build_context_with_meta(docs: List) -> str:
    """
    Given a list of LangChain Documents, build a single context string where each chunk
    is prefixed by its metadata in square brackets.
    """
    blocks = []
    for d in docs:
        m = d.metadata
        # Base tag
        tag = f"[{m['filename']} | p.{m['pages']} | {m['type']}]"
        
        blocks.append(f"{tag}\n{d.page_content}")
    # Separate blocks by blank lines
    return "\n\n".join(blocks)

def query_all_docs(query: str) -> str:
    # 1) load index & retriever
    faiss_index = load_faiss_index()
    if faiss_index is None:
        raise RuntimeError("Index not found. Ingest first.")
    retriever = faiss_index.as_retriever(search_kwargs={"k": 4})

    # 2) fetch & tag docs
    docs = retriever.get_relevant_documents(query)
    context_with_meta = build_context_with_meta(docs)

    # 3) format the final prompt
    prompt_str = PROMPT.format(
        context_with_meta=context_with_meta,
        question=query
    )

    # 4) call the LLM directly
    answer = text_llm.invoke(prompt_str).strip()
    return answer