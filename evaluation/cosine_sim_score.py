from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

# Load your model
model_name = "ibm-granite/granite-embedding-30m-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # shape: [1, 384]

# Your test samples
samples = [
    {
        "reference": "The HC Overview Brochure focuses on Guidehouse's comprehensive suite of healthcare technology solutions, including cybersecurity, advanced analytics, enterprise resource planning, interoperability, and digital transformation",
        "prediction": "The main focus of the HC Overview Brochure is to highlight Guidehouse Health IT (HIT) Solutions team, which is composed of over 700 technology experts including data engineers, analysts, software developers, digital strategists, and other professionals with extensive digital health experience. The team is led by a former health plan executive, Sziklai, who specializes in embedding digital transformation, cloud-based, and automation solutions into health plan models and their partnerships with healthcare organizations."
    },
    {
        "reference": "The tables show that the synthetic-text detector achieves high accuracy and macro-F1 scores in both summary-level and abstract-level data splits, indicating its effectiveness in identifying LLM-generated text ",
        "prediction": """The tables in the paper reveal that the synthetic-text detector performed exceptionally well. In the summary-level setup, all synthetic summaries were correctly identified (15 out of 15), while no real summaries were misclassified (0 out of 26). Similarly, in the abstract-level setup, all synthetic summaries were correctly identified (13 out of 13), with no real summaries misclassified (0 out of 5). 

This high accuracy suggests that the detector is effective in distinguishing between synthetic and real text, particularly in the context of summaries and abstracts generated using large language models like ChatGPT. 
"""
    },
    {
        "reference": "The paper includes a diagram that illustrates the architecture of the ActionCLIP model, showing the components of the video encoder, text encoder, and similarity calculation module ",
        "prediction": """The paper visually represents the architecture of the ActionCLIP model through several figures. Figure (b) provides textual prompt examples, while Figures (c) and (d) detail the in-network and pre-network visual prompts, respectively. The post-network visual prompts, including MeanP Conv1D LSTM, are detailed in Figures (e), (f), and (g)."""
    }
]

# Compute cosine similarities
scores = []
for sample in samples:
    ref_vec = embed(sample["reference"])
    pred_vec = embed(sample["prediction"])
    score = cosine_similarity(ref_vec, pred_vec).item()
    scores.append(score)
    print(f"Cosine similarity: {score:.4f}")

print(f"\nAverage cosine similarity: {sum(scores)/len(scores):.4f}")
