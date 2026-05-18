
import json
import numpy as np
from pathlib import Path

def compute_grouped_auc(run_dir):
    out_path = Path(run_dir) / "output.json"
    cfg_path = Path(run_dir) / "config.json"
    if not out_path.exists() or not cfg_path.exists():
        return None
    try:
        with open(cfg_path) as f: cfg = json.load(f)
        with open(out_path) as f: data = json.load(f)
        
        if isinstance(data, list): records = data
        elif isinstance(data, dict): records = data.get("records", [])
        else: return None
            
        if not records: return None
        
        files = cfg.get("train_files", [])
        c2l = {}
        for i, f in enumerate(files):
            fname = Path(f).name.lower()
            if "gamma" in fname: c2l["gamma"] = i
            elif "pi0" in fname: c2l["pi0"] = i
            elif "e-" in fname: c2l["e-"] = i
            
        if "pi0" not in c2l or "gamma" not in c2l: return None
        
        p0_idx, g_idx, e_idx = c2l["pi0"], c2l["gamma"], c2l.get("e-")
        
        # EM Grouping indices
        em_indices = [g_idx]
        if e_idx is not None: em_indices.append(e_idx)
        
        # Comparison: pi0 (pos) vs {gamma, e-} (neg)
        include_indices = [p0_idx] + em_indices
        
        labels, scores = [], []
        for r in records:
            lbl = int(r["label"])
            if lbl not in include_indices: continue
            
            logits = np.array(r["logits"])
            exps = np.exp(logits - np.max(logits))
            probs = exps / np.sum(exps)
            
            # Grouped EM score: P(pi0) / (P(pi0) + P(gamma) + P(e-))
            denom = probs[p0_idx] + probs[g_idx]
            if e_idx is not None: denom += probs[e_idx]
            
            score = probs[p0_idx] / denom if denom > 0 else 0
            labels.append(1 if lbl == p0_idx else 0)
            scores.append(score)
            
        labels, scores = np.array(labels), np.array(scores)
        if len(np.unique(labels)) < 2: return None
        
        n_pos, n_neg = np.sum(labels == 1), np.sum(labels == 0)
        ranks = np.argsort(scores)
        y_true_sorted = labels[ranks]
        pos_ranks = np.where(y_true_sorted == 1)[0]
        return (np.sum(pos_ranks + 1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    except: return None

save_dir = Path("save")
run_dirs = sorted(list(save_dir.glob("split_mamba*")))
results = []
for i, rd in enumerate(run_dirs):
    auc = compute_grouped_auc(rd)
    if auc is not None: 
        results.append((rd.name, auc))
        
results.sort(key=lambda x: x[1], reverse=True)
print("\n" + "="*70)
print(f"{'Ranking':<5} | {'Run Name':45s} | {'EM-Grouped AUC':15s}")
print("-" * 70)
for i, (name, auc) in enumerate(results):
    print(f"{i+1:<7} | {name:45s} | {auc:.5f}")
print("="*70)
