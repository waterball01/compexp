import os
import sys

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL = "meta-llama/Llama-3.2-1B"

LABELS = ("entailment", "neutral", "contradiction")
LABEL_STOI = {"entailment": 0, "neutral": 1, "contradiction": 2}
LABEL_ITOS = {v: k for k, v in LABEL_STOI.items()}
MAX_SAMPLES = 10000


def format_example(premise, hypothesis):
    return f"Premise: {premise}\nHypothesis: {hypothesis}"


def load_tok(tok_path):
    with open(tok_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) % 2 != 0:
        raise RuntimeError("uneven src/hyp")

    fbase = os.path.splitext(os.path.basename(tok_path))[0]
    gt_csv = os.path.join(os.path.dirname(tok_path), "preds", f"6_{fbase}.csv")
    gt_df = pd.read_csv(gt_csv).iloc[:MAX_SAMPLES]
    lines = lines[: MAX_SAMPLES * 2]

    examples = []
    for i in range(0, len(lines), 2):
        premise = lines[i]
        hypothesis = lines[i + 1]
        label = gt_df.loc[i // 2, "gt"]
        if label not in LABEL_STOI:
            continue
        examples.append(
            {
                "text": format_example(premise, hypothesis),
                "label": LABEL_STOI[label],
            }
        )

    return examples


def tokenize_batch(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=256)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def main(tok_path):
    examples = load_tok(tok_path)
    mid = len(examples) // 2
    train_examples = examples[:mid]
    val_examples = examples[mid:]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        num_labels=len(LABELS),
        id2label=LABEL_ITOS,
        label2id=LABEL_STOI,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    train_ds = Dataset.from_list(train_examples).map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
    ).remove_columns(["text"])
    val_ds = Dataset.from_list(val_examples).map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
    ).remove_columns(["text"])

    nli_dir = os.path.dirname(os.path.dirname(tok_path))
    out_dir = os.path.join(nli_dir, "models", "llama32-1b-snli")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=out_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=True,
            logging_steps=100,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    metrics = trainer.evaluate()
    print(f"Val acc: {metrics['eval_accuracy']:.3f}")
    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <tokenized_path>")
    main(sys.argv[1])
