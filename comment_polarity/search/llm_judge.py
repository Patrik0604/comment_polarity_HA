import json
import numpy as np
import pandas as pd
import openai

from ..models.prompting_model import openai_available


def llm_judge(query: str, results: pd.DataFrame, engine: str) -> int:
    if not openai_available():
        return 3

    payload = {
        "query": query,
        "engine": engine,
        "results": results["text"].tolist()
    }

    judge_prompt = (
        "You are a relevance judge. "
        "Rate the relevance of these results to the query on a scale of 1 (bad) to 5 (excellent). "
        "Return only the integer."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": json.dumps(payload, indent=2)}
        ],
        temperature=0
    )

    import re
    m = re.search(r"[1-5]", response.choices[0].message["content"])
    return int(m.group(0)) if m else 3


def evaluate_with_llm_judge(df, tfidf, tfidf_matrix, embed, embeddings):
    if not openai_available():
        print("[WARN] LLM-judge skipped")
        return {}

    from .search_engines import search_tfidf, search_embedding, search_hybrid

    queries = [
        "great product quality",
        "terrible customer support",
        "average content not too good not too bad"
    ]

    scores = {"tfidf": [], "embedding": [], "hybrid": []}

    for q in queries:
        scores["tfidf"].append(llm_judge(q, search_tfidf(q, df, tfidf, tfidf_matrix), "tfidf"))
        scores["embedding"].append(llm_judge(q, search_embedding(q, df, embed, embeddings), "embedding"))
        scores["hybrid"].append(llm_judge(q, search_hybrid(q, df, tfidf, tfidf_matrix, embed, embeddings), "hybrid"))

    print("\n=== LLM-Judge average scores (1–5) ===")
    for k, v in scores.items():
        print(f"{k}: {np.mean(v):.2f}")

    return scores
