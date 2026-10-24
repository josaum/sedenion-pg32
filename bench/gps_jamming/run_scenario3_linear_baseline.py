"""Scenario III rerun with a linear-gate baseline: identical scaffold, the sedenion path's
closed-form scalar gate g(phi) = (1+cos phi)/sqrt(2), and no hypercomplex products."""
import json, pathlib, sys, time, torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.argv = [sys.argv[0]]
import run_gps_bench as g
import torch.nn.functional as F
from models import Encoder
from tasks import T

def run(kinds, epochs=6, n_train=8000):
    torch.set_num_threads(8)
    train = g.make_crpa_dataset(n_train, [0, 1, 2], seed=101, max_slots=20)
    test_id = g.make_crpa_dataset(2000, [0, 1, 2], seed=102, max_slots=20)
    test_ood = g.make_crpa_dataset(2000, [3], seed=103, max_slots=20)
    def batches(data, bs=128, shuffle=True):
        n = data[0].shape[0]; idx = torch.randperm(n) if shuffle else torch.arange(n)
        for s in range(0, n, bs):
            j = idx[s:s+bs]; yield [t[j] for t in data]
    @torch.no_grad()
    def evaluate(model, data):
        model.eval(); corr = tot = fn = fd = 0
        for xt, xa, xi, live, y, d in batches(data, 256, False):
            logit, _ = model(xt, xa, xi); pred = (logit > 0).long()
            corr += (pred == y).sum().item(); tot += len(y)
            m = (y == 0); fn += (pred[m] == 1).sum().item(); fd += m.sum().item()
        return round(corr / tot, 4), round(fn / max(fd, 1), 4)
    out = {}
    for kind in kinds:
        t0 = time.time(); torch.manual_seed(0); model = Encoder(kind)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (len(train[0]) // 128 + 1))
        for _ in range(epochs):
            model.train()
            for xt, xa, xi, live, y, d in batches(train):
                logit, _ = model(xt, xa, xi); loss = F.binary_cross_entropy_with_logits(logit, y.float())
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        a_id, f_id = evaluate(model, test_id); a_ood, f_ood = evaluate(model, test_ood)
        out[kind] = dict(params=n_params, acc_id=a_id, flr_id=f_id, acc_ood=a_ood, flr_ood=f_ood, train_s=round(time.time() - t0))
        print(f"{kind:22s} params={n_params:6d} ID={a_id:.4f} FLR={f_id:.4f} | OOD={a_ood:.4f} FLR={f_ood:.4f} ({out[kind]['train_s']}s)", flush=True)
    return out

if __name__ == "__main__":
    res = run(["sedenion", "linear_gate", "linear_gate_moreno", "sedenion_noffn", "linear_gate_noffn", "linear_gate_moreno_noffn"])
    (HERE / "results_scenario3_linear_baseline.json").write_text(json.dumps(res, indent=1))
