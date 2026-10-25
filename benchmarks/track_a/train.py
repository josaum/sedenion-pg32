"""
Track A Training & Evaluation Benchmark:
Yang-Mills Scattering Amplitudes, BCJ Jacobi Duality (R_BCJ) & Collinear Pole Error (E_pole)
(Section 5.2 & Table 2 of Agourakis 2026, "Sedenion PG(3,2) Neural Topologies").

Models are trained strictly on hard-scattering amplitude regression (|s_ij| in [0.08, 1.20])
using MSE loss on A_hat = n_s/s + n_t/t + n_u/u (zero direct supervision on n_s, n_t, n_u,
zero BCJ penalty, and zero collinear-pole training samples), then evaluated on:
  1. R_BCJ  = E[||n_s + n_t + n_u||_2] / E[||n_s||_2 + ||n_t||_2 + ||n_u||_2]
  2. E_pole = E[||A_hat - A*||_2] / E[||A*||_2] for collinear singularities |s_ij| < 1e-4
  3. E_id   = E[||A_hat - A*||_2] / E[||A*||_2] on held-out hard-scattering kinematics
"""
import json
import pathlib
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tasks import generate_ym_dataset
from models import YangMillsSurrogate


def iter_batches(data: dict, batch_size: int, shuffle: bool, seed: int = 0):
    n = data["x_in"].shape[0]
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)
    else:
        idx = torch.arange(n)
    for i in range(0, n, batch_size):
        sl = idx[i : i + batch_size]
        yield {k: v[sl] for k, v in data.items()}


@torch.no_grad()
def evaluate(model: YangMillsSurrogate, data: dict, batch_size: int = 256):
    model.eval()
    err_num, err_den = 0.0, 0.0
    bcj_num, bcj_den = 0.0, 0.0
    for b in iter_batches(data, batch_size, shuffle=False):
        amp_pred, n_pred = model(b)
        amp_true = b["amp_true"]
        err_num += (amp_pred - amp_true).norm(dim=-1).sum().item()
        err_den += amp_true.norm(dim=-1).sum().item()

        n_sum = n_pred.sum(dim=1).norm(dim=-1)  # ||n_s + n_t + n_u||_2
        n_mag = n_pred.norm(dim=-1).sum(dim=1)  # ||n_s||_2 + ||n_t||_2 + ||n_u||_2
        bcj_num += n_sum.sum().item()
        bcj_den += n_mag.sum().item()

    rel_err = err_num / max(err_den, 1e-12)
    r_bcj = bcj_num / max(bcj_den, 1e-12)
    return rel_err, r_bcj


def train_and_eval(
    kind: str,
    seed: int = 0,
    epochs: int = 25,
    n_train: int = 8192,
    n_test: int = 2048,
    batch_size: int = 128,
    lr: float = 3e-3,
):
    torch.manual_seed(seed)
    train_data = generate_ym_dataset(n_train, seed=100 + seed, pole_regime=False)
    test_id = generate_ym_dataset(n_test, seed=200 + seed, pole_regime=False)
    test_pole = generate_ym_dataset(n_test, seed=300 + seed, pole_regime=True, pole_min=1e-6, pole_max=1e-4)

    model = YangMillsSurrogate(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_ep = (n_train + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_ep)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for b in iter_batches(train_data, batch_size, shuffle=True, seed=seed * 1000 + ep):
            amp_pred, _ = model(b)
            loss = F.mse_loss(amp_pred, b["amp_true"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

    e_id, r_bcj_id = evaluate(model, test_id)
    e_pole, r_bcj_pole = evaluate(model, test_pole)
    dt = round(time.time() - t0, 2)
    return {
        "kind": kind,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "E_id": round(e_id, 6),
        "R_BCJ_id": round(r_bcj_id, 6),
        "E_pole": round(e_pole, 6),
        "R_BCJ": round(r_bcj_pole, 6),
        "train_s": dt,
    }


if __name__ == "__main__":
    torch.set_num_threads(4)
    kinds = [
        "Dense-R16",
        "Quat-H4",
        "Oct-O2",
        "Clifford-Cl04",
        "Sed-Free",
        "Sed-PG32-Moreno",
        "Sed-PG32-Moreno-NoFFN",
    ]
    seeds = [0, 1]
    results = {}
    for k in kinds:
        results[k] = {}
        for s in seeds:
            r = train_and_eval(k, seed=s)
            results[k][str(s)] = r
            print(f"{k:23s} seed={s} | R_BCJ={r['R_BCJ']:.6f} | E_pole={r['E_pole']:.6f} | E_id={r['E_id']:.6f} ({r['train_s']}s)", flush=True)

    out_path = pathlib.Path(__file__).resolve().parent / "results_track_a.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved Track A results to {out_path}")
