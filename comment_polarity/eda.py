import pandas as pd

def run_eda(df: pd.DataFrame):
    print("\n=== EDA: First rows ===")
    print(df.head())

    print("\n=== Missing values ===")
    print(df.isna().sum())

    df["text_length"] = df["text"].astype(str).str.len()
    print("\n=== Text length stats ===")
    print(df["text_length"].describe())

    if "label" in df.columns:
        print("\n=== Polarity label distribution ===")
        print(df["label"].value_counts())
