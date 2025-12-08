import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from ..config import HF_MODEL_NAME, RANDOM_STATE


class PolarityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=128):
        self.texts = list(texts)
        self.labels = [label2id[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_neural_model(train_df, test_df):
    print("\ntraining the DISTILBERT")

    unique = sorted(train_df["label"].unique())
    label2id = {lbl: i for i, lbl in enumerate(unique)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    train_ds = PolarityDataset(train_df["text"], train_df["label"], tokenizer, label2id)
    test_ds = PolarityDataset(test_df["text"], test_df["label"], tokenizer, label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir="./polarity_model",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        seed=RANDOM_STATE
    )

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics
    )

    trainer.train()
    results = trainer.evaluate()
    print("\nDistilBERT model training results")
    print(results)

    return model, tokenizer, label2id, id2label, results
