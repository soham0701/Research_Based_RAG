from rouge_score import rouge_scorer

samples = [
    {
        "query": "What evidence is presented to show that a smaller open-source language model can outperform a much larger model on several benchmarks?",
        "reference": "The GPT-J-6B model using AMA outperforms the few-shot GPT-3 175B model on 15 of 20 benchmarks, with an average improvement of 41% over the 6B model’s few-shot baseline ",
        "prediction": "Evidence is presented in the document ASK ME ANYTHING- A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS.pdf on page 8 and 9. The study compares the open-source GPT-J-6B model with the few-shot (k ∈ [32 .. 70]) GPT3-175B model across 20 benchmarks. Despite the significantly larger parameter count of GPT3-175B, the open-source 6B parameter GPT-J-6B model outperforms it on 15 of these benchmarks. Furthermore, the AMA (Ask Me Anything) strategy, applied to GPT-J-6B, provides an average improvement of 41% over the few-shot (k = 3) performance of the 6B parameter model, effectively closing the gap with the larger model on knowledge-intensive tasks."
    },
    {
        "query": "What do the tables in the paper reveal about the performance of the synthetic-text detector?",
        "reference": "The tables show that the synthetic-text detector achieves high accuracy and macro-F1 scores in both summary-level and abstract-level data splits, indicating its effectiveness in identifying LLM-generated text ",
        "prediction": """The tables in the paper reveal that the synthetic-text detector performed exceptionally well. In the summary-level setup, all synthetic summaries were correctly identified (15 out of 15), while no real summaries were misclassified (0 out of 26). Similarly, in the abstract-level setup, all synthetic summaries were correctly identified (13 out of 13), with no real summaries misclassified (0 out of 5). 

This high accuracy suggests that the detector is effective in distinguishing between synthetic and real text, particularly in the context of summaries and abstracts generated using large language models like ChatGPT. 
"""
    },
    {
        "query": "How does the paper visually represent the architecture of the ActionCLIP model?",
        "reference": "The paper includes a diagram that illustrates the architecture of the ActionCLIP model, showing the components of the video encoder, text encoder, and similarity calculation module ",
        "prediction": """The paper visually represents the architecture of the ActionCLIP model through several figures. Figure (b) provides textual prompt examples, while Figures (c) and (d) detail the in-network and pre-network visual prompts, respectively. The post-network visual prompts, including MeanP Conv1D LSTM, are detailed in Figures (e), (f), and (g)."""
        }
]

scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

total_rouge1 = 0

for sample in samples:
    scores = scorer.score(sample["reference"], sample["prediction"])
    r1 = scores['rouge1'].fmeasure

    total_rouge1 += r1


    print(f"Query: {sample['query']}")
    print(f"ROUGE-1 F1: {r1:.3f}")
  
# Compute average
n = len(samples)
avg_r1 = total_rouge1 / n


print("----- AVERAGE SCORES -----")
print(f"Average ROUGE-1 F1: {avg_r1:.3f}")
