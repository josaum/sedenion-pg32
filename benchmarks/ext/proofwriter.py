"""
External Benchmark 1: AllenAI ProofWriter (Tafjord et al., ACL 2021, `tasksource/proofwriter`).
Trains parameter-matched multi-hop encoders on shallow AttNeg-OWA deductions (depths d in {0, 1, 2})
with pure terminal binary cross-entropy (aux = 0.0) and evaluates on:
  1. In-Distribution AttNeg-OWA (d in {0, 1, 2}, N = 3,000)
  2. Out-of-Distribution Deep Proofs AttNeg-OWA (d in {3, 4, 5}, N = 3,504)
  3. Zero-Shot Non-Monotonic Exception Transfer: Birds-Electricity (N = 1,330)
"""
import json, math, os, pathlib, re, sys, time, urllib.request
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

SUBJS = ['*', 'anne', 'arthur', 'bill', 'bob', 'charlie', 'colin', 'dave', 'erin', 'fiona', 'gary', 'harry']
SUBJ2ID = {s: i for i, s in enumerate(SUBJS)}

PREDS = [
    'NONE', 'abnormal', 'battery_flat', 'bell_ringing', 'big', 'bird', 'blue',
    'circuit_battery', 'circuit_bell', 'circuit_complete', 'circuit_dead',
    'circuit_include_battery', 'circuit_include_bell', 'circuit_include_light_bulb',
    'circuit_include_radio', 'circuit_include_switch', 'circuit_light_bulb',
    'circuit_powered', 'circuit_radio', 'circuit_switch', 'cold',
    'current_run_through_bell', 'current_run_through_circuit',
    'current_run_through_light_bulb', 'current_run_through_radio', 'flying',
    'furry', 'green', 'kind', 'light_bulb_glowing', 'nice', 'ostrich', 'quiet',
    'radio_playing', 'red', 'rough', 'round', 'smart', 'switch_on', 'white',
    'wire_conducting', 'wire_metal', 'wire_plastic', 'wounded', 'young'
]
PRED2ID = {p: i for i, p in enumerate(PREDS)}
N_LITS = 2 * len(PREDS)


def ensure_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "proofwriter_test.parquet": "https://huggingface.co/datasets/tasksource/proofwriter/resolve/main/data/test-00000-of-00001-3e27b013c60e12d8.parquet",
        "proofwriter_val.parquet": "https://huggingface.co/datasets/tasksource/proofwriter/resolve/main/data/validation-00000-of-00001-8f79b25dd5b0f2c3.parquet",
    }
    for name, url in files.items():
        p = DATA_DIR / name
        if not p.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                p.write_bytes(r.read())


def lit_id(pol, pred):
    if pol == 0 or pred == "NONE":
        return 0
    pid = PRED2ID[pred]
    return pid if pol > 0 else (pid + len(PREDS))


def norm_pred(s):
    s = s.lower().strip().replace("light bulb", "light_bulb").replace("includes", "include").replace("runs", "run")
    neg = 1
    for neg_tok in ["is not ", "are not ", "does not ", "do not ", "cannot ", "not "]:
        if neg_tok in s:
            neg = -1
            s = s.replace(neg_tok, " ")
    toks = [
        w for w in re.findall(r"[a-z_]+", s)
        if w not in ("the", "a", "an", "is", "are", "does", "do", "can", "they", "it", "someone", "something", "people", "things", "all", "have", "has")
    ]
    subj = "*"
    rem = []
    for w in toks:
        if w in ("anne", "bob", "charlie", "dave", "erin", "fiona", "gary", "harry", "arthur", "bill", "colin") and subj == "*":
            subj = w
        else:
            rem.append(w)
    pred = "_".join(rem)
    if pred == "fly":
        pred = "flying"
    return subj, neg, pred


def parse_sentence(s):
    s = s.strip().strip(".")
    if s.lower().startswith("if "):
        body = s[3:]
        ant_str, cons_str = body.split(" then ")
        ants = ant_str.split(" and ")
        s1, p1, a1 = norm_pred(ants[0])
        if len(ants) > 1:
            s2, p2, a2 = norm_pred(ants[1])
        else:
            s2, p2, a2 = "*", 0, "NONE"
        sc, pc, cons = norm_pred(cons_str)
        subj = s1 if s1 != "*" else (s2 if s2 != "*" else sc)
        return (2, SUBJ2ID[subj], lit_id(p1, a1), lit_id(p2, a2), lit_id(pc, cons))
    elif " people are " in s.lower() or " things are " in s.lower():
        lhs, rhs = re.split(r" (?:people|things) are ", s.lower())
        lhs = re.sub(r"^all ", "", lhs.strip())
        parts = [p.strip() for p in lhs.split(",")]
        _, p1, a1 = norm_pred(parts[0])
        if len(parts) > 1:
            _, p2, a2 = norm_pred(parts[1])
        else:
            p2, a2 = 0, "NONE"
        _, pc, cons = norm_pred("are " + rhs)
        return (2, 0, lit_id(p1, a1), lit_id(p2, a2), lit_id(pc, cons))
    else:
        subj, pol, pred = norm_pred(s)
        return (1, SUBJ2ID[subj], 0, 0, lit_id(pol, pred))


def load_proofwriter_split(parquet_path, depths_filter=None, config_filter=None, max_samples=None):
    t = pq.read_table(parquet_path).to_pydict()
    rows = []
    for i in range(len(t["id"])):
        if t["answer"][i] not in ("True", "False"):
            continue
        if config_filter == "birds-electricity":
            if t["config"][i] != "birds-electricity":
                continue
        else:
            if not t["id"][i].startswith("AttNeg-OWA"):
                continue
        d = int(t["QDep"][i])
        if depths_filter is not None and d not in depths_filter:
            continue
        sents = [parse_sentence(s) for s in t["theory"][i].split(". ") if s]
        q = parse_sentence(t["question"][i])
        q_tok = (3, q[1], q[4], 0, q[4])
        rows.append((sents, q_tok, 1 if t["answer"][i] == "True" else 0, d))
        if max_samples and len(rows) >= max_samples:
            break
    max_len = 28
    N = len(rows)
    xt = torch.zeros(N, max_len, dtype=torch.long)
    xs = torch.zeros(N, max_len, dtype=torch.long)
    xa1 = torch.zeros(N, max_len, dtype=torch.long)
    xa2 = torch.zeros(N, max_len, dtype=torch.long)
    xc = torch.zeros(N, max_len, dtype=torch.long)
    y = torch.zeros(N, dtype=torch.long)
    depths = torch.zeros(N, dtype=torch.long)
    for idx, (sents, q_tok, ans, d) in enumerate(rows):
        seq = sents[:max_len - 1] + [q_tok]
        for j, (tp, sb, a1, a2, cs) in enumerate(seq):
            xt[idx, j] = tp
            xs[idx, j] = sb
            xa1[idx, j] = a1
            xa2[idx, j] = a2
            xc[idx, j] = cs
        y[idx] = ans
        depths[idx] = d
    return xt, xs, xa1, xa2, xc, y, depths


class ProofWriterEncoder(nn.Module):
    def __init__(self, kind, d=64, n_heads=4, n_layers=2, n_steps=5):
        super().__init__()
        self.kind = kind
        self.d = d
        self.n_steps = n_steps
        self.et = nn.Embedding(4, d)
        self.es = nn.Embedding(len(SUBJS), d)
        self.el = nn.Embedding(N_LITS + 2, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, kind) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, xt, xs, xa1, xa2, xc):
        B, N = xt.shape
        device = xt.device
        qpos = (xt == 3).float().argmax(1)
        q_subj = xs[torch.arange(B, device=device), qpos].unsqueeze(1)

        subj_match = (xs == q_subj) | (xs == 0)
        is_fact = (xt == 1) & subj_match
        is_rule = (xt == 2) & subj_match
        has_a2 = is_rule & (xa2 > 0)

        s_inactive = self.et(torch.zeros_like(xt))
        s_active = self.et(torch.ones_like(xt))
        w_inactive = self.blocks[0].ffn(s_inactive)
        w_active = self.blocks[0].ffn(s_active)
        x0_dyn0 = self.blocks[0].n1(w_active)

        w = torch.where(is_fact.unsqueeze(-1), w_active, w_inactive)

        xc_eff = torch.where(is_fact | is_rule, xc, torch.full_like(xc, N_LITS + 1))
        x0_stat = self.blocks[0].n1(self.el(xc_eff))
        v0_stat = self.blocks[0].n1(self.et(xt.clamp_max(2)))
        mask_clauses = is_fact | is_rule

        q_a1 = self.blocks[0].n1(self.el(torch.where(is_rule, xa1, torch.zeros_like(xa1))))
        q_a2 = self.blocks[0].n1(self.el(torch.where(has_a2, xa2, torch.zeros_like(xa2))))

        has_pre = hasattr(self.blocks[0].attn, "precompute_static")
        cached_a1 = (
            self.blocks[0].attn.precompute_static(
                q_a1, x0_stat, mask_clauses, v_src=v0_stat, use_q_proj_for_k=True, x0_dyn=x0_dyn0
            )
            if has_pre else None
        )
        cached_a2 = (
            self.blocks[0].attn.precompute_static(
                q_a2, x0_stat, mask_clauses, v_src=v0_stat, use_q_proj_for_k=True, x0_dyn=x0_dyn0
            )
            if has_pre else None
        )

        for _ in range(self.n_steps):
            xd = self.blocks[0].n1(w)
            out_a1 = self.blocks[0].attn(
                xd, mask_clauses, q_in=q_a1, x_stat=x0_stat, v_in=v0_stat, cached=cached_a1, x0_dyn=x0_dyn0
            )
            score_a1 = self.blocks[0].attn.last_scores_grad.sum(dim=2).mean(dim=-1)

            out_a2 = self.blocks[0].attn(
                xd, mask_clauses, q_in=q_a2, x_stat=x0_stat, v_in=v0_stat, cached=cached_a2, x0_dyn=x0_dyn0
            )
            score_a2 = self.blocks[0].attn.last_scores_grad.sum(dim=2).mean(dim=-1)

            g1 = torch.tanh(F.relu(score_a1 - 0.15) * 10.0)
            g2 = torch.where(has_a2, torch.tanh(F.relu(score_a2 - 0.15) * 10.0), torch.ones_like(g1))
            rule_fire = (g1 * g2).unsqueeze(-1)

            w_rule = rule_fire * w_active + (1.0 - rule_fire) * (w_inactive + out_a1 + out_a2)
            w = torch.where(is_fact.unsqueeze(-1), w_active, torch.where(is_rule.unsqueeze(-1), w_rule, w_inactive))

        x_base = self.et(xt) + self.el(xc_eff)
        q_blk1 = self.blocks[1].n1(self.el(xc))
        x_stat_blk1 = self.blocks[1].n1(self.el(xc_eff))
        v_blk1 = self.blocks[1].n1(x_base)
        xd0_blk1 = self.blocks[1].n1(w_active)
        xd_blk1 = self.blocks[1].n1(w)

        cached1 = (
            self.blocks[1].attn.precompute_static(
                q_blk1, x_stat_blk1, mask_clauses, v_src=v_blk1, use_q_proj_for_k=True, x0_dyn=xd0_blk1
            )
            if hasattr(self.blocks[1].attn, "precompute_static") else None
        )
        attn_out1 = self.blocks[1].attn(
            xd_blk1, mask_clauses, q_in=q_blk1, x_stat=x_stat_blk1, v_in=v_blk1, cached=cached1, x0_dyn=xd0_blk1
        )
        x_out = self.blocks[1].ffn(x_base + attn_out1)
        xq = self.norm(x_out[torch.arange(B, device=device), qpos])
        return self.head(xq).squeeze(-1), qpos


if __name__ == "__main__":
    ensure_data()
    torch.set_num_threads(4)
    tr = load_proofwriter_split(DATA_DIR / "proofwriter_val.parquet", depths_filter={0, 1, 2})
    te_id = load_proofwriter_split(DATA_DIR / "proofwriter_test.parquet", depths_filter={0, 1, 2}, max_samples=3000)
    te_ood = load_proofwriter_split(DATA_DIR / "proofwriter_test.parquet", depths_filter={3, 4, 5})
    te_be = load_proofwriter_split(DATA_DIR / "proofwriter_test.parquet", config_filter="birds-electricity")

    def batches(data, bs=128, shuffle=True):
        n = data[0].shape[0]
        idx = torch.randperm(n) if shuffle else torch.arange(n)
        for s in range(0, n, bs):
            j = idx[s:s + bs]
            yield [t[j] for t in data]

    @torch.no_grad()
    def evaluate(model, data):
        model.eval()
        corr = tot = fcr_num = fcr_den = 0
        for xt, xs, xa1, xa2, xc, y, d in batches(data, 256, False):
            logit, _ = model(xt, xs, xa1, xa2, xc)
            pred = (logit > 0).long()
            corr += (pred == y).sum().item()
            tot += len(y)
            neg_mask = (y == 0)
            fcr_num += (pred[neg_mask] == 1).sum().item()
            fcr_den += neg_mask.sum().item()
        return round(corr / max(tot, 1), 4), round(fcr_num / max(fcr_den, 1), 4)

    for kind in ("softmax", "bilinear", "octonion", "clifford", "sedenion_plain", "sedenion", "sedenion_noffn"):
        torch.manual_seed(0)
        model = ProofWriterEncoder(kind)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        epochs = 5
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (len(tr[0]) // 128 + 1))
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            for xt, xs, xa1, xa2, xc, y, d in batches(tr, 128, True):
                logit, _ = model(xt, xs, xa1, xa2, xc)
                loss = F.binary_cross_entropy_with_logits(logit, y.float())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
        acc_id, fcr_id = evaluate(model, te_id)
        acc_ood, fcr_ood = evaluate(model, te_ood)
        acc_be, fcr_be = evaluate(model, te_be)
        print(f"{kind:16s} params={n_params:6d} ({round(time.time()-t0)}s) | ID={acc_id:.4f} FCR={fcr_id:.4f} | OOD={acc_ood:.4f} FCR={fcr_ood:.4f} | BirdsElec={acc_be:.4f} FCR={fcr_be:.4f}", flush=True)
