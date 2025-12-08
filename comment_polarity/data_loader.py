from pathlib import Path
import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, RANDOM_STATE


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found at: {path}")
    return pd.read_csv(path)


def weak_label_polarity(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    return "neutral"


def ensure_polarity(df: pd.DataFrame) -> pd.DataFrame:
    if "label" in df.columns:
        print("[INFO] 'label' column found. Using existing polarity labels.")
        return df
    print("[INFO] No labels found. Generating polarity with TextBlob...")
    df["label"] = df["text"].astype(str).apply(weak_label_polarity)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    print(f"[INFO] Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df
