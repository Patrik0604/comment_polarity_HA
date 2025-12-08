import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
from sentence_transformers import SentenceTransformer

from ..config import EMBED_MODEL_NAME


def build_search_engines(df):
    print("\n=== Building TF-IDF + Embedding search engines ===")

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["text"])

    embed = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embed.encode(df["text"].tolist(), convert_to_numpy=True, show_progress_bar=True)

    return tfidf, tfidf_matrix, embed, embeddings


def search_tfidf(query, df, tfidf, tfidf_matrix, k=5):
    q = tfidf.transform([query])
    sims = cosine_similarity(q, tfidf_matrix)[0]
    idx = sims.argsort()[::-1][:k]
    result = df.iloc[idx][["text", "label"]].copy()
    result["score"] = sims[idx]
    return result


def search_embedding(query, df, embed, embeddings, k=5):
    q_emb = embed.encode([query], convert_to_numpy=True)[0]
    sims = cosine_similarity([q_emb], embeddings)[0]
    idx = sims.argsort()[::-1][:k]
    result = df.iloc[idx][["text", "label"]].copy()
    result["score"] = sims[idx]
    return result


def search_hybrid(query, df, tfidf, tfidf_matrix, embed, embeddings, k=5, alpha=0.5):
    q_tfidf = tfidf.transform([query])
    s_tfidf = cosine_similarity(q_tfidf, tfidf_matrix)[0]

    q_emb = embed.encode([query], convert_to_numpy=True)[0]
    s_emb = cosine_similarity([q_emb], embeddings)[0]

    s_tfidf = minmax_scale(s_tfidf)
    s_emb = minmax_scale(s_emb)

    hybrid = alpha * s_tfidf + (1 - alpha) * s_emb
    idx = hybrid.argsort()[::-1][:k]

    result = df.iloc[idx][["text", "label"]].copy()
    result["tfidf_score"] = s_tfidf[idx]
    result["emb_score"] = s_emb[idx]
    result["hybrid_score"] = hybrid[idx]
    return result
