from pathlib import Path
import io
import base64

from pydantic import BaseModel, field_validator
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem

from .models import tokenizer, vision_processor, vision_llm, text_llm, embedder
from .cache  import SOURCE_DIR, save_cached_docs, save_faiss_index

# ─── Converter setup ───────────────────────────────────────────────
pdf_opts  = PdfPipelineOptions(do_ocr=False, generate_picture_images=True)
fmt_opts  = { InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts) }
converter = DocumentConverter(format_options=fmt_opts)

# ─── Pydantic class ───────────────────────────────────────────────
class ChunkMeta(BaseModel):
    """Defines a data validation class to check the output from Docling and the Image and Table summaries"""
    filename: str
    pages: List[int]
    type: str

    @field_validator("filename")
    def filename_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("filename cannot be empty")
        return v

    @field_validator("pages")
    def pages_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("pages list cannot be empty")
        return v
    
# ─── Converter function ───────────────────────────────────────────────
def convert_pdf_to_langchain(path: Path) -> list[Document]:
    """
    Takes each PDF and connverts them into a docling document. Separates texts, images and tables and attached metadata.
    Chunks the text and converts Images and Tables into summaries. 
    Finally, returns each chunk in a Langchain Document format
    """
    doc    = converter.convert(source=str(path)).document
    chunks = []

    # ─── 1) Text chunks ───────────────────────────────────────────────
    for ch in HybridChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=50).chunk(doc):
        items = ch.meta.doc_items
        # skip pure-table chunks
        if len(items) == 1 and isinstance(items[0], TableItem):
            continue

        # collect all the page numbers this chunk spans
        pages = sorted({ item.prov[0].page_no for item in items })

        validated_meta=ChunkMeta(
                filename=path.name,
                pages=pages,
                type="text"
            ).model_dump()

        chunks.append(Document(
            page_content=ch.text,
            metadata=validated_meta
        ))

    # ─── 2) Table chunks ──────────────────────────────────────────────
    for table_idx, tbl in enumerate(doc.tables, start=1):
        # collect all pages the table spans (usually one, but could be more)
        pages = sorted({ prov.page_no for prov in tbl.prov })

        # Export to Markdown (for prompting only)
        md = tbl.export_to_markdown(doc=doc)

        table_prompt=(
            "Here is a table in Markdown format:\n\n"
            f"{md}\n\n"
            "Please analyze this table and summarize in 3 sentences. "
        )
        table_summary=text_llm.invoke(table_prompt)
        
        validated_meta=ChunkMeta(
                filename=path.name,
                pages=pages,
                type="table"
            ).model_dump()

        chunks.append(Document(
            page_content=table_summary,
            metadata=validated_meta
        ))

    # ─── 3) Image chunks ──────────────────────────────────────────────
    def encode_image(img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    for image_idx, pic in enumerate(doc.pictures, start=1):
        pages = sorted({ prov.page_no for prov in pic.prov })
        img     = pic.get_image(doc)
        caption = pic.caption_text(doc)

        conv = [{
            "role": "user",
            "content": [
                {"type":"image"},
                {"type":"text",  "text": f"Caption: {caption}"},
                {"type":"text",  "text": "Please analyze this image and summarize in 3 sentences."},
            ],
        }]
        vision_prompt = vision_processor.apply_chat_template(
            conversation=conv,
            add_generation_prompt=True,
        )
        image_summary = vision_llm.invoke(vision_prompt, image=encode_image(img))

        validated_meta=ChunkMeta(
                filename=path.name,
                pages=pages,
                type="image"
            ).model_dump()

        chunks.append(Document(
            page_content=image_summary,
            metadata=validated_meta
        ))

    return chunks

# ─── Ingest New PDFs function ───────────────────────────────────────────────
def ingest_new_pdfs(faiss_index, cached_docs):
    """
    Scan SOURCE_DIR for all PDFs, process only new ones
    Returns (updated_index, updated_cache, [new_filenames]).
    """
    all_pdfs = sorted(SOURCE_DIR.glob("*.pdf"))
    to_proc  = [p for p in all_pdfs if p.name not in cached_docs]
    if not to_proc:
        return faiss_index, cached_docs, []

    new_chunks = []
    for p in to_proc:
        docs = convert_pdf_to_langchain(p)
        cached_docs[p.name] = docs
        new_chunks.extend(docs)

    texts = [d.page_content for d in new_chunks]
    metas = [d.metadata     for d in new_chunks]

    if faiss_index is None:
        faiss_index = FAISS.from_texts(texts, embedder, metadatas=metas)
    else:
        faiss_index.add_texts(texts, metadatas=metas)

    save_cached_docs(cached_docs)  #save the langchain docs in cache
    save_faiss_index(faiss_index)  #save the index in cache

    return faiss_index, cached_docs, [p.name for p in to_proc]
