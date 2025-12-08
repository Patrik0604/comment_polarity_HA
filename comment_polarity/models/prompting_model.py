import os
import openai
from sklearn.metrics import classification_report, accuracy_score, f1_score


def openai_available():
    key = os.environ.get("OPENAI_API_KEY", None)
    if not key:
        print("no api key, skip.")
        return False
    openai.api_key = key
    return True


def classify_polarity_with_llm(text):
    prompt = f"""
Classify the polarity (positive, negative, neutral) of the following comment:

\"\"\"{text}\"\"\"
"""

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a polarity classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    out = resp.choices[0].message["content"].lower()

    if "positive" in out: return "positive"
    if "negative" in out: return "negative"
    return "neutral"


def evaluate_prompting_model(test_df, max_samples=30):
    if not openai_available():
        return None, None

    sample = test_df.sample(min(max_samples, len(test_df)), random_state=42)

    y_true = sample["label"].tolist()
    y_pred = [classify_polarity_with_llm(t) for t in sample["text"]]

    print("\n--- Prompt-based LLM polarity classification ---")
    print(classification_report(y_true, y_pred))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    return acc, f1
