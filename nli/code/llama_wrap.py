import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))
 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
 
import formula as FM
import settings
 
MODEL_NAME  = "meta-llama/Llama-3.2-1B"
TOK_FILE    = "data/analysis/snli_1.0_dev.tok"    # alternating prem / hyp lines
PREDS_FILE  = "data/analysis/preds/6_snli_1.0_dev.csv"
NUM_UNITS   = 20
BATCH_SIZE  = 4
MAX_LEN     = 128
DEVICE      = "cpu"
LABEL_MAP   = {"entailment": 0, "neutral": 1, "contradiction": 2}
OUT_CSV     = "exp/llama_compexp_results.csv"
 
 
class Hook:
    def __init__(self):
        self.out = None
    def register(self, model):
        mlp = model.model.layers[-2].mlp   # penultimate MLP block
        mlp.register_forward_hook(
            lambda m, i, o: setattr(self, "out", o[:, -1, :].detach().cpu().float())
        )
        print(f"Hooked layers[-2].mlp ({type(mlp).__name__})")
        return self
 
 
def load_data():
    lines = open(TOK_FILE).read().splitlines()
    prems = lines[0::2]
    hyps = lines[1::2]
 
    labels = []
    with open(PREDS_FILE) as f:
        for row in csv.DictReader(f):
            labels.append(LABEL_MAP.get(row["gt"], -1))
 
    pairs = []
    for p, h, l in zip(prems, hyps, labels):
        if l != -1:
            pairs.append({
                "prem_raw": p,                            
                "hyp_raw": h,                             
                "prem_tokens": set(p.lower().split()),    
                "hyp_tokens": set(h.lower().split()),   
                "label": l
            })
            
    print(f"Loaded {len(pairs)} tokenized pairs")
    return pairs
 
 
def extract(model, tokenizer, hook, pairs):
    all_acts, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            
            # Fix: Reconstruct prompts by pulling text directly out of the dictionary data structure
            prompts = [
                f"Premise: {item['prem_raw']}\nHypothesis: {item['hyp_raw']}\nRelationship:"
                for item in batch
            ]
            
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LEN).to(DEVICE)
            model(**enc)                              # triggers hook
            all_acts.append(hook.out.numpy())
            
            # Fix: Pull the actual integer label out of the dictionary key
            all_labels.extend(item["label"] for item in batch)
            
            if i % 100 == 0:
                print(f"  {i}/{len(pairs)}", end="\r")
    print()
    return np.concatenate(all_acts, axis=0), np.array(all_labels, dtype=np.int32)
 
 
def iou(a, b):
    return (a & b).sum() / ((a | b).sum() + 1e-9)
 
def build_concept_masks(pairs):
    prem_counts = Counter()
    hyp_counts = Counter()
    for item in pairs:
        prem_counts.update(item["prem_tokens"])
        hyp_counts.update(item["hyp_tokens"])
    
    top_prem_toks = [tok for tok, _ in prem_counts.most_common(30)]
    top_hyp_toks = [tok for tok, _ in hyp_counts.most_common(30)]
    
    masks = {}
    num_samples = len(pairs)
    
    for tok in top_prem_toks:
        masks[f"pre:tok:{tok}"] = np.array([tok in item["prem_tokens"] for item in pairs], dtype=bool)
        
    for tok in top_hyp_toks:
        masks[f"hyp:tok:{tok}"] = np.array([tok in item["hyp_tokens"] for item in pairs], dtype=bool)
        
    return masks
 
def eval_formula(f, masks):
    if f.mask is not None:
        return f.mask
    if isinstance(f, FM.Leaf):
        m = masks[list(masks.keys())[f.val]].copy()
    elif isinstance(f, FM.Not):
        m = ~eval_formula(f.val, masks)
    elif isinstance(f, FM.And):
        m = eval_formula(f.left, masks) & eval_formula(f.right, masks)
    elif isinstance(f, FM.Or):
        m = eval_formula(f.left, masks) | eval_formula(f.right, masks)
    else:
        raise ValueError(f"Unknown formula type {type(f)}")
    f.mask = m
    return m
 
def beam_search(target, masks, max_len=3, beam=10):
    n = len(masks)
    candidates = {}
    for i in range(n):
        for cls in (FM.Leaf, lambda v: FM.Not(FM.Leaf(v))):
            f = cls(i)
            f.mask = None
            score = iou(eval_formula(f, masks), target)
            f.mask = None   # clear so beam expansion re-evaluates fresh copies
            candidates[str(f)] = (score, f)
 
    beam_now = dict(Counter({k: v[0] for k, v in candidates.items()}).most_common(beam))
    best_score, best_f = max(candidates.values(), key=lambda x: x[0])
 
    for _ in range(max_len - 1):
        new_cands = {}
        for key in beam_now:
            _, existing = candidates[key]
            for i in range(n):
                for op in (FM.And, FM.Or):
                    for neg in (False, True):
                        leaf = FM.Leaf(i)
                        new_f = op(existing, FM.Not(leaf) if neg else leaf)
                        new_f.mask = None
                        s = iou(eval_formula(new_f, masks), target)
                        new_f.mask = None
                        new_cands[str(new_f)] = (s, new_f)
        candidates.update(new_cands)
        beam_now = dict(Counter({k: v[0] for k, v in candidates.items()}).most_common(beam))
        top_s, top_f = max(candidates.values(), key=lambda x: x[0])
        if top_s > best_score:
            best_score, best_f = top_s, top_f
 
    return best_f, best_score
 
def formula_to_str(f, mask_keys):
    if isinstance(f, FM.Leaf):
        return mask_keys[f.val]
    if isinstance(f, FM.Not):
        return f"NOT({formula_to_str(f.val, mask_keys)})"
    if isinstance(f, FM.And):
        return f"({formula_to_str(f.left, mask_keys)} AND {formula_to_str(f.right, mask_keys)})"
    if isinstance(f, FM.Or):
        return f"({formula_to_str(f.left, mask_keys)} OR {formula_to_str(f.right, mask_keys)})"
    return str(f)
 
def run_compexp(acts, pairs):
    masks = build_concept_masks(pairs)
    mask_keys = list(masks.keys())
    mask_vals = {k: v for k, v in masks.items()}
 
    thresholds = acts.mean(0) + 0.5 * acts.std(0)
    results = []
    for u in range(NUM_UNITS):
        target = acts[:, u] >= thresholds[u]
        if target.sum() < 5:
            results.append((u, "SKIP", 0.0, int(target.sum())))
            continue
        best_f, best_iou = beam_search(target, mask_vals,
                                       max_len=settings.MAX_FORMULA_LENGTH,
                                       beam=settings.BEAM_SIZE)
        label = formula_to_str(best_f, mask_keys)
        print(f"  unit {u:4d} | IoU={best_iou:.4f} | {label}")
        results.append((u, label, round(best_iou, 6), int(target.sum())))
    return results
 
 
def main():
    pairs = load_data()
 
    print(f"Loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    ).to(DEVICE if DEVICE != "cuda" else "cpu")   # device_map handles cuda
 
    hook = Hook().register(model)
 
    print("Extracting penultimate MLP activations...")
    acts, labels = extract(model, tokenizer, hook, pairs)
    print(f"Activations: {acts.shape}")
 
    print(f"\nRunning CompExp on units 0–{NUM_UNITS - 1}...")
    results = run_compexp(acts, pairs) 
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "formula", "iou", "n_firing"])
        w.writerows(results)
    print(f"\nSaved → {OUT_CSV}")
 
    ious = [r[2] for r in results if r[1] != "SKIP"]
    if ious:
        print(f"Mean IoU: {np.mean(ious):.4f}  |  Max: {np.max(ious):.4f}")
 
 
if __name__ == "__main__":
    main()