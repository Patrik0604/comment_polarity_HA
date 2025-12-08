from pathlib import Path
from joblib import dump

from comment_polarity.utils import setup_environment
from comment_polarity.config import (
    DATA_PATH,
    MODELS_DIR,
    CLASSIC_MODEL_PATH,
    NEURAL_MODEL_DIR,
)
from comment_polarity.data_loader import load_data, ensure_polarity, split_data
from comment_polarity.eda import run_eda
from comment_polarity.models.classic_model import train_classic_model
from comment_polarity.models.neural_model import train_neural_model
from comment_polarity.models.prompting_model import evaluate_prompting_model
from comment_polarity.search.search_engines import (
    build_search_engines,
    search_tfidf,
    search_embedding,
    search_hybrid,
)
from comment_polarity.search.llm_judge import evaluate_with_llm_judge


def main():
    # setup the env
    setup_environment()

    print(f"[INFO] Loading labeled training dataset from: {DATA_PATH}")
    df = load_data(DATA_PATH)
    df = ensure_polarity(df)

    # runEDA
    run_eda(df)

    # slip the comments.csv data into train and test data
    train_df, test_df = split_data(df)

    # check directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Classic TF-IDF + Logistic Regression model
    classic_model, classic_acc, classic_f1 = train_classic_model(train_df, test_df)
    print(f"[INFO] Saving classic model to: {CLASSIC_MODEL_PATH}")
    dump(classic_model, CLASSIC_MODEL_PATH)

    # Neural DistilBERT model
    neural_model, neural_tokenizer, label2id, id2label, neural_metrics = train_neural_model(train_df, test_df)
    print(f"[INFO] Saving neural model and tokenizer to: {NEURAL_MODEL_DIR}")
    neural_model.save_pretrained(NEURAL_MODEL_DIR)
    neural_tokenizer.save_pretrained(NEURAL_MODEL_DIR)

    # Prompt based LLM (Turned off because we don't have API key)
    evaluate_prompting_model(test_df)

    # Search engine
    tfidf_vectorizer, tfidf_matrix, embed_model, embeddings = build_search_engines(df)

    demo_query = "great product but slow delivery"
    print(f"\n=== DEMO SEARCH for query: '{demo_query}' ===")

    print("\n--- TF-IDF Search ---")
    print(search_tfidf(demo_query, df, tfidf_vectorizer, tfidf_matrix, k=5))

    print("\n--- Embedding Search ---")
    print(search_embedding(demo_query, df, embed_model, embeddings, k=5))

    print("\n--- Hybrid Search ---")
    print(search_hybrid(demo_query, df, tfidf_vectorizer, tfidf_matrix, embed_model, embeddings, k=5))

    # LLM judge, still turned off
    evaluate_with_llm_judge(df, tfidf_vectorizer, tfidf_matrix, embed_model, embeddings)

    print("\n=== Training & evaluation pipeline finished successfully ===")
    print("[INFO] You can now run `run_predict.py` to label new, unlabeled comments.")


if __name__ == "__main__":
    main()
