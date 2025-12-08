from pathlib import Path

import pandas as pd
import torch
from joblib import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from comment_polarity.utils import setup_environment
from comment_polarity.config import (
    DATA_DIR,
    CLASSIC_MODEL_PATH,
    NEURAL_MODEL_DIR,
)


def load_classic_model():
    if not CLASSIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classic model not found at {CLASSIC_MODEL_PATH}. "
            f"Run main.py first to train and save the models."
        )
    print(f"[INFO] Loading classic model from: {CLASSIC_MODEL_PATH}")
    return load(CLASSIC_MODEL_PATH)


def load_neural_model():
    if not NEURAL_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Neural model directory not found at {NEURAL_MODEL_DIR}. "
            f"Run main.py first to train and save the models."
        )
    print(f"[INFO] Loading neural model + tokenizer from: {NEURAL_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(NEURAL_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(NEURAL_MODEL_DIR)
    # id2label stored in config.json / model.config
    id2label = model.config.id2label
    # keys are strings like "0","1","2", convert to int-keyed dict
    id2label = {int(k): v for k, v in id2label.items()}
    return model, tokenizer, id2label


def predict_polarity_neural(text: str, model, tokenizer, id2label: dict) -> str:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**enc)
    pred_id = outputs.logits.argmax(dim=-1).item()
    return id2label[pred_id]


def main():
    setup_environment()

    # this is the real data for production
    input_path = DATA_DIR / "to_be_labeled.csv"
    output_path = DATA_DIR / "to_be_labeled_labeled.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file with unlabeled comments not found at: {input_path}\n"
            f"Create a CSV with at least a 'text' column."
        )

    print(f"[INFO] Reading unlabeled comments from: {input_path}")
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    # importinf the modells
    # classic_model = load_classic_model()

    neural_model, neural_tokenizer, id2label = load_neural_model()

    #polarity prediction using the neutal model
    print("Predicting polarity for new comments using DISTILBERT")
    df["predicted_polarity"] = df["text"].apply(
        lambda t: predict_polarity_neural(str(t), neural_model, neural_tokenizer, id2label)
    )

    print(f" Saving labeled output to: {output_path}")
    df.to_csv(output_path, index=False)

    print("\n completed")
    print(df.head())


if __name__ == "__main__":
    main()
