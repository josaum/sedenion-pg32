"""
Train each kernel on override depths {0,..,MAXD}; test in-distribution and OOD at depth MAXD+1.

Metrics
  acc[d]    accuracy by depth
  FCR       false-compliance rate: P(pred=permit | label=forbid and an (ineffective) permit rule exists)
  leak      relative attention mass of the QUERY token on REVOKED clauses vs LIVE clauses, last layer:
            mean_revoked(score) / mean_live(score).  0 = exact pruning.
"""
import json
import sys
import time
import torch
import torch.nn.functional as F
from tasks import dataset, T, N_ACTIONS
import importlib, os
Encoder = importlib.import_module(os.environ.get("MODELS_MODULE", "models")).Encoder

torch.set_num_threads(4)
KINDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["softmax", "bilinear", "octonion", "sedenion"]
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
MAXD = int(sys.argv[4]) if len(sys.argv) > 4 else 2          # train depths 0..MAXD, OOD = MAXD+1
AUX = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0       # weight of the final-layer structure-supervision loss
TAG = sys.argv[6] if len(sys.argv) > 6 else ""

train = dataset(24000, list(range(MAXD + 1)), seed=1, max_len=20)
test_id = dataset(4000, list(range(MAXD + 1)), seed=2, max_len=20)
test_ood = dataset(4000, [MAXD + 1], seed=3, max_len=20)
OOD = MAXD + 1


def batches(data, bs, shuffle):
    n = data[0].shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, bs):
        j = idx[s:s + bs]
        yield [t[j] for t in data]


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = {"n": 0, "correct": {}, "count": {}, "fcr_num": 0, "fcr_den": 0, "leak_rev": [], "leak_live": []}
    for xt, xa, xi, live, y, d in batches(data, 512, False):
        logit, qpos = model(xt, xa, xi)
        pred = (logit > 0).long()
        for dd in d.unique().tolist():
            m = d == dd
            out["correct"][dd] = out["correct"].get(dd, 0) + (pred[m] == y[m]).sum().item()
            out["count"][dd] = out["count"].get(dd, 0) + m.sum().item()
        # FCR: label forbid, but a permit RULE for the queried action is present (and revoked)
        permit_rule_present = ((xt == T["RULE"]) & (xa == (xa[torch.arange(len(y)), qpos] + N_ACTIONS).unsqueeze(1))).any(1)
        den = (y == 0) & permit_rule_present
        out["fcr_num"] += (pred[den] == 1).sum().item()
        out["fcr_den"] += den.sum().item()
        # Leakage audit on final layer (after multi-hop chain resolution)
        s = model.blocks[-1].attn.last_scores                      # [B, N, N, h]
        sq = s[torch.arange(len(y)), qpos].mean(-1)                 # [B, N] query row, mean over heads
        rev = (live == 0)
        lv = (live == 1) & (xt != T["QUERY"])
        for b in range(len(y)):
            if rev[b].any() and lv[b].any():
                out["leak_rev"].append(sq[b][rev[b]].mean().item())
                out["leak_live"].append(sq[b][lv[b]].mean().item())
    acc = {dd: out["correct"][dd] / out["count"][dd] for dd in out["count"]}
    fcr = out["fcr_num"] / max(out["fcr_den"], 1)
    leak = (sum(out["leak_rev"]) / len(out["leak_rev"])) / max(sum(out["leak_live"]) / len(out["leak_live"]), 1e-12)
    return acc, fcr, leak


results = {}
for kind in KINDS:
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = Encoder(kind)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * (24000 // 128 + 1))
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            for xt, xa, xi, live, y, d in batches(train, 128, True):
                logit, qpos = model(xt, xa, xi)
                loss = F.binary_cross_entropy_with_logits(logit, y.float())
                if AUX > 0:
                    # Structure supervision on the FINAL readout layer (after multi-hop message passing):
                    # Penalizing Layer 0 before IDs have propagated along the chain is information-theoretically invalid.
                    s = model.blocks[-1].attn.last_scores_grad      # [B, N, N, h] with grad
                    sq = s[torch.arange(len(y)), qpos].mean(-1)     # [B, N]
                    rev = (live == 0).float()
                    lv = ((live == 1) & (xt != T["QUERY"])).float()
                    aux = (sq.pow(2) * rev).sum() / rev.sum().clamp_min(1.0) \
                        + (F.relu((0.1 if kind == "softmax" else 0.5) - sq) * lv).sum() / lv.sum().clamp_min(1.0)
                    loss = loss + AUX * aux
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
        acc_id, fcr_id, leak_id = evaluate(model, test_id)
        acc_ood, fcr_ood, leak_ood = evaluate(model, test_ood)
        r = {
            "acc_id": acc_id,
            "fcr_id": fcr_id,
            "leak_id": leak_id,
            f"acc_ood{OOD}": acc_ood[OOD],
            f"fcr_ood{OOD}": fcr_ood,
            f"leak_ood{OOD}": leak_ood,
            "train_s": round(time.time() - t0),
            "maxd": MAXD,
            "aux": AUX,
        }
        if hasattr(model.blocks[0].attn, "alpha"):
            r["alpha"] = [round(torch.sigmoid(b.attn.alpha).mean().item(), 2) for b in model.blocks]
        results[f"{kind}/{seed}"] = r
        print(kind, seed, json.dumps(r), flush=True)
        json.dump(results, open(f"results_{'_'.join(KINDS)}{TAG}.json", "w"), indent=1)