from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score


def train_classic_model(train_df, test_df):
    print("\nTF-IDF + LOGREG")

    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=200))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nclassic model report")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"[CLASSIC] Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return model, acc, f1
