import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from llama import MODEL, ask

LABELS = ("entailment", "neutral", "contradiction")
MAX_SAMPLES = 10000

SINGLE_WORD_SYSTEM = (
    "Answer with a SINGLE WORD only. "
    "Do not add punctuation, explanation, or any other text. "
    "Given a premise and hypothesis, reply with exactly one of: "
    "entailment, neutral, contradiction."
)


def predict(premise, hypothesis):
    response = ask(
        SINGLE_WORD_SYSTEM,
        f"Premise: {premise}\nHypothesis: {hypothesis}",
    )
    text = response.strip().lower()
    for label in LABELS:
        if text == label or re.search(rf"\b{label}\b", text):
            return label
    return "unparseable"


def main(tok_path):
    with open(tok_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) % 2 != 0:
        raise RuntimeError("uneven src/hyp")

    fbase = os.path.splitext(os.path.basename(tok_path))[0]
    gt_csv = os.path.join(os.path.dirname(tok_path), "preds", f"6_{fbase}.csv")
    gt_df = pd.read_csv(gt_csv).iloc[:MAX_SAMPLES]
    lines = lines[: MAX_SAMPLES * 2]

    all_preds = []
    gt_labels = []
    for i in tqdm(range(0, len(lines), 2)):
        pre_raw = lines[i]
        hyp_raw = lines[i + 1]
        pred = predict(pre_raw, hyp_raw)
        gt = gt_df.loc[i // 2, "gt"]
        all_preds.append(pred)
        gt_labels.append(gt)

    hits = [p == g for p, g in zip(all_preds, gt_labels)]
    acc = np.mean(hits)
    print(f"Val acc: {acc:.3f}")

    mbase = MODEL.replace(":", "_").replace(".", "_")
    preds_file = f"{mbase}_{fbase}.csv"
    preds_folder = os.path.join(os.path.dirname(tok_path), "preds")
    os.makedirs(preds_folder, exist_ok=True)
    preds_file = os.path.join(preds_folder, preds_file)
    preds_df = pd.DataFrame({"gt": gt_labels, "pred": all_preds, "correct": hits})
    preds_df.to_csv(preds_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <tokenized_path>")
    main(sys.argv[1])
