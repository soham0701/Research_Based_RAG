from pathlib import Path
import gradio as gr
from rag.pipeline import RAGPipeline
from rag.cache import CACHE_DIR, EMBEDDER

pipeline = RAGPipeline(Path("./source_documents"))

def upload_pdf(path):      
    return pipeline.uploaded_pdf(Path(path))

def get_answers(q):        
    return pipeline.query(q)

# ─── GRADIO UI LAYOUT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with gr.Blocks() as demo:

        gr.Markdown("# Multimodal RAG")

        with gr.Accordion("ℹ️ What does this tool do?", open=False):
            gr.Markdown("This Retrieval-Augmented Generation (RAG) tool allows you to upload research PDFs containing **text**, **tables**, and **images**. "
                "It extracts structured content, indexes it semantically, and lets you ask questions that are answered using a language model "
                "based only on the documents you've uploaded."
            )

        # Section 1: PDF upload
        gr.Markdown("### 📥 Upload New PDF") 
        with gr.Row():
            pdf_uploader = gr.File(
                label="Select a PDF to ingest",
                file_count="single",
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("Upload New PDF")
        pdf_status = gr.Textbox(label="Upload Status", lines=1)
        upload_btn.click(fn=upload_pdf, inputs=pdf_uploader, outputs=pdf_status)

        gr.Markdown("---")

        # Section 2: Ask questions
        gr.Markdown("### ❓ Ask a Question")
        query_in  = gr.Textbox(label="Your question", lines=2, placeholder="Type here…")
        ask_btn   = gr.Button("Get Answers")
        answer_out = gr.Textbox(label="Answer & References", lines=10, placeholder="Ask me a query!")
        ask_btn.click(fn=get_answers, inputs=query_in, outputs=answer_out)

    demo.launch()
