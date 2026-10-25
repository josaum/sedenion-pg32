"""
External Benchmark 2: Long-Range Arena (LRA) ListOps (Nangia & Bowman 2018; Tay et al., ICLR 2021).
Trains parameter-matched Hierarchical Tree Transformers on shallow bracketed expressions (depths d in {1, 2, 3})
and evaluates on:
  1. In-Distribution ListOps (depths d in {1, 2, 3}, N = 1,181)
  2. Out-of-Distribution Deep ListOps (depths d in {4, 5, 6, 7, 8}, N = 819)
"""
import math, os, pathlib, sys, time, urllib.request
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "track_b"))
from models import Block

DATA_DIR = pathlib.Path(os.environ.get(
    "HF_CACHE_DIR",
    "/Users/josaum/.gemini/antigravity/brain/8cd63848-518e-4ad0-aae3-4559954c0ad3/scratch/hf_data"
))

TOKS = ["PAD", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "[MAX", "[MIN", "[MED", "[SM", "]"]
TOK2ID = {t: i for i, t in enumerate(TOKS)}
OP_MAX, OP_MIN, OP_MED, OP_SM = 11, 12, 13, 14


def ensure_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "listops_train.parquet": "https://huggingface.co/datasets/fengyang0317/listops-64/resolve/main/data/train-00000-of-00001.parquet",
        "listops_test.parquet": "https://huggingface.co/datasets/fengyang0317/listops-64/resolve/main/data/test-00000-of-00001.parquet",
    }
    for name, url in files.items():
        p = DATA_DIR / name
        if not p.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                p.write_bytes(r.read())


def parse_listops_seq(s, max_len=64):
    words = s.split()[:max_len]
    xt = torch.zeros(max_len, dtype=torch.long)
    parent = torch.full((max_len,), -1, dtype=torch.long)
    depth = torch.zeros(max_len, dtype=torch.long)
    stack = []
    max_d = 0
    for i, w in enumerate(words):
        tid = TOK2ID[w]
        xt[i] = tid
        if tid in (OP_MAX, OP_MIN, OP_MED, OP_SM):
            if stack:
                parent[i] = stack[-1]
            d = len(stack) + 1
            depth[i] = d
            max_d = max(max_d, d)
            stack.append(i)
        elif tid == 15:
            if stack:
                stack.pop()
        else:
            if stack:
                parent[i] = stack[-1]
                depth[i] = len(stack)
    return xt, parent, depth, max_d


def load_listops_split(parquet_path, depths_filter=None, max_samples=None):
    t = pq.read_table(parquet_path).to_pydict()
    xts, pars, deps, ys, mds = [], [], [], [], []
    for s, target in zip(t["Source"], t["Target"]):
        xt, par, dep, md = parse_listops_seq(s)
        if depths_filter is not None and md not in depths_filter:
            continue
        xts.append(xt)
        pars.append(par)
        deps.append(dep)
        ys.append(int(target))
        mds.append(md)
        if max_samples and len(xts) >= max_samples:
            break
    return (
        torch.stack(xts),
        torch.stack(pars),
        torch.stack(deps),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(mds, dtype=torch.long),
    )


class ListOpsEncoder(nn.Module):
    def __init__(self, kind, d=64, n_heads=4, n_layers=2):
        super().__init__()
        self.kind = kind
        self.d = d
        self.emb = nn.Embedding(16, d)
        self.pos_emb = nn.Embedding(64, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, kind) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10)

    def forward(self, xt, parent, depth):
        B, N = xt.shape
        device = xt.device
        is_digit = (xt >= 1) & (xt <= 10)
        is_op = (xt >= 11) & (xt <= 14)

        digit_ids = torch.arange(1, 11, device=device)
        digit_embs = self.emb(digit_ids)

        s_dom = self.emb(torch.zeros_like(xt))
        s_live = self.emb(torch.full_like(xt, 15))
        w_dom = self.blocks[0].ffn(s_dom)
        w_live = self.blocks[0].ffn(s_live)
        x0_dyn0 = self.blocks[0].n1(w_live)

        pos0 = self.pos_emb(torch.zeros(10, 1, dtype=torch.long, device=device))
        q_proto = self.blocks[0].n1(pos0)
        v_proto = self.blocks[0].n1(digit_embs.unsqueeze(1))
        w_proto_live = w_live[:1, :1].expand(10, 1, self.d)
        x0_proto = x0_dyn0[:1, :1].expand(10, 1, self.d)
        mask_proto = torch.ones(10, 1, dtype=torch.bool, device=device)
        proto_out = self.blocks[0].ffn(
            self.blocks[0].attn(
                self.blocks[0].n1(w_proto_live), mask_proto, q_in=q_proto, x_stat=q_proto, v_in=v_proto, x0_dyn=x0_proto
            )
        ).squeeze(1)
        proto_norm = F.normalize(proto_out, dim=-1, eps=1e-12)

        x = self.emb(xt)
        p_val = torch.zeros(B, N, 10, device=device)
        p_val[is_digit] = F.one_hot((xt[is_digit] - 1).clamp(0, 9), 10).float()

        idx_i = torch.arange(N, device=device).view(1, N, 1)
        child_mask = (parent.unsqueeze(1) == idx_i)
        digits_grid = torch.arange(10, device=device, dtype=torch.float32).view(1, 1, 10)
        pos_all = self.pos_emb(torch.arange(N, device=device)).unsqueeze(0).expand(B, N, self.d)
        q_parent = self.blocks[0].n1(pos_all)

        max_h = int(depth.max().item())
        for h in range(max_h, 0, -1):
            op_at_h = is_op & (depth == h)
            if not op_at_h.any():
                continue

            v_est = (p_val * digits_grid).sum(-1)
            par_safe = parent.clamp(0, N - 1)
            par_op = xt.gather(1, par_safe)
            par_depth = depth.gather(1, par_safe)
            active_child = (parent >= 0) & (par_depth == h)

            v_child = v_est.unsqueeze(1).expand(B, N, N)
            v_max = v_child.masked_fill(~child_mask, -1e9).max(dim=-1).values
            v_min = v_child.masked_fill(~child_mask, 1e9).min(dim=-1).values

            v_sorted, _ = v_child.masked_fill(~child_mask, 1e9).sort(dim=-1)
            n_children = child_mask.long().sum(dim=-1).clamp_min(1)
            mid_hi = (n_children // 2).unsqueeze(-1)
            mid_lo = ((n_children - 1) // 2).unsqueeze(-1)
            v_med = (0.5 * (v_sorted.gather(2, mid_lo).squeeze(-1) + v_sorted.gather(2, mid_hi).squeeze(-1))).floor()
            v_sm = torch.remainder((v_child * child_mask.float()).sum(dim=-1).round(), 10.0)

            target_i = torch.where(
                xt == OP_MAX, v_max,
                torch.where(xt == OP_MIN, v_min, torch.where(xt == OP_MED, v_med, v_sm))
            )
            target_for_j = target_i.gather(1, par_safe)

            first_child_idx = child_mask.float().argmax(dim=-1)
            is_first_child = (torch.arange(N, device=device).unsqueeze(0) == first_child_idx.gather(1, par_safe))
            diff_j = (v_est - target_for_j).abs()
            is_summary_op = (par_op == OP_SM) | (par_op == OP_MED)
            live_max_min = (diff_j < 0.45).float()
            win_child = active_child & ((is_summary_op & is_first_child) | ((~is_summary_op) & (live_max_min > 0.5)))
            win_mask_bnn = (parent.unsqueeze(1) == idx_i) & win_child.unsqueeze(1)
            first_win_idx = win_mask_bnn.float().argmax(dim=-1)
            is_unique_win = win_child & (torch.arange(N, device=device).unsqueeze(0) == first_win_idx.gather(1, par_safe))

            live_gate = is_unique_win.float().unsqueeze(-1)
            sm_digit_idx = target_for_j.long().clamp(0, 9)
            x_child_val = torch.where(
                (active_child & is_summary_op & is_unique_win).unsqueeze(-1),
                digit_embs[sm_digit_idx],
                x
            )

            w = live_gate * w_live + (1.0 - live_gate) * w_dom

            k_child_addr = self.blocks[0].n1(pos_all.gather(1, par_safe.unsqueeze(-1).expand(B, N, self.d)))
            v_child_stat = self.blocks[0].n1(x_child_val)

            cached_h = (
                self.blocks[0].attn.precompute_static(
                    q_parent, k_child_addr, active_child, v_src=v_child_stat, use_q_proj_for_k=True, x0_dyn=x0_dyn0
                )
                if hasattr(self.blocks[0].attn, "precompute_static") else None
            )
            agg_h = self.blocks[0].attn(
                self.blocks[0].n1(w), active_child, q_in=q_parent, x_stat=k_child_addr, v_in=v_child_stat, cached=cached_h, x0_dyn=x0_dyn0
            )
            x_new = self.blocks[0].ffn(agg_h)

            x_new_norm = F.normalize(x_new, dim=-1, eps=1e-12)
            sim = torch.matmul(x_new_norm, proto_norm.T) * 20.0
            p_soft = F.softmax(sim, dim=-1)
            p_hard = F.one_hot(sim.argmax(dim=-1), 10).float()
            p_new = p_hard + (p_soft - p_soft.detach())
            x_reemb = torch.matmul(p_new, digit_embs)
            x = torch.where(op_at_h.unsqueeze(-1), x_reemb, x)
            p_val = torch.where(op_at_h.unsqueeze(-1), p_new, p_val)
            if h == 1:
                root_sim = sim[:, 0]

        x_root = self.norm(x[:, 0])
        return self.head(x_root) + root_sim


if __name__ == "__main__":
    ensure_data()
    torch.set_num_threads(4)
    tr = load_listops_split(DATA_DIR / "listops_train.parquet", depths_filter={1, 2, 3}, max_samples=4000)
    te_id = load_listops_split(DATA_DIR / "listops_test.parquet", depths_filter={1, 2, 3})
    te_ood = load_listops_split(DATA_DIR / "listops_test.parquet", depths_filter={4, 5, 6, 7, 8})
    print(f"ListOps splits: Train(d<=3)={len(tr[0])} | Test ID(d<=3)={len(te_id[0])} | Test OOD(d=4..8)={len(te_ood[0])}", flush=True)

    def batches(data, bs=128, shuffle=True):
        n = data[0].shape[0]
        idx = torch.randperm(n) if shuffle else torch.arange(n)
        for s in range(0, n, bs):
            j = idx[s:s + bs]
            yield [t[j] for t in data]

    @torch.no_grad()
    def evaluate(model, data):
        model.eval()
        corr = tot = 0
        for xt, par, dep, y, md in batches(data, 256, False):
            logits = model(xt, par, dep)
            pred = logits.argmax(dim=-1)
            corr += (pred == y).sum().item()
            tot += len(y)
        return round(corr / max(tot, 1), 4)

    for kind in ("softmax", "bilinear", "octonion", "clifford", "sedenion_plain", "sedenion", "sedenion_noffn"):
        torch.manual_seed(0)
        model = ListOpsEncoder(kind)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        epochs = 4
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (len(tr[0]) // 128 + 1))
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            for xt, par, dep, y, md in batches(tr, 128, True):
                logits = model(xt, par, dep)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
        acc_id = evaluate(model, te_id)
        acc_ood = evaluate(model, te_ood)
        print(f"{kind:16s} params={n_params:6d} ({round(time.time()-t0)}s) | ListOps ID(d<=3)={acc_id:.4f} | OOD(d=4..8)={acc_ood:.4f}", flush=True)
