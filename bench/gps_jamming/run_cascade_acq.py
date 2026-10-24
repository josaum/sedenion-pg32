"""1023-chip GPS L1 C/A acquisition on the 2-stage sedenion cascade output (0.5 deg drifting +80 dB jammer,
K=64 window, SNR -20 dB), vs rank-2 projection, 1-stage sedenion and LSMI-MVDR. Same estimation for all
(top-2 eigenvectors of the same R_hat).  Leakage taken as the window-mean jammer residual."""
import json, math, pathlib, importlib.util
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
t = load("run_tracking_cascade"); f = load("run_fair_baseline")
rb = t.rb; cd = rb.cd_mul; unit = rb.unit; rng = np.random.default_rng(7)
NOISE_STD = 10.0
st = {k: dict(hit=0, sinr=[], null=[]) for k in ["sed_1stage", "sed_2stage_cascade", "proj_rank2", "mvdr_lsmi"]}
N = 256
for _ in range(N):
    th0 = rng.uniform(-60, 60); th_s = th0 + (20 if th0 < 35 else -20); u_sat = unit(rb.c8_to_r16(rb.steer(th_s)))
    thetas = th0 + np.linspace(0, 0.5, 64); X = t.snapshots(thetas, u_sat); R = X.T @ X / 64
    ev, V = np.linalg.eigh(R); u1 = unit(V[:, -1]); u2 = unit(V[:, -2])
    T1, v1, _ = t.sed_weights(u1, u_sat); y1 = lambda x: cd(T1 @ x, v1); r_sat = y1(u_sat)
    T2, v2, _ = t.sed_weights(y1(u2), r_sat); y2 = lambda r: cd(T2 @ r, v2); s2 = unit(y2(r_sat)); s1 = unit(r_sat)
    w_mv = unit(np.linalg.solve(R + 1e-2 * np.eye(16), u_sat)); Q, _ = np.linalg.qr(np.stack([u1, u2], 1)); wp = unit(u_sat - Q @ (Q.T @ u_sat))
    xs = [unit(rb.c8_to_r16(rb.steer(th))) for th in thetas]
    cases = {
        "sed_1stage": (np.mean([abs(y1(x) @ s1) for x in xs]), abs(r_sat @ s1)),
        "sed_2stage_cascade": (np.mean([abs(y2(y1(x)) @ s2) for x in xs]), abs(y2(r_sat) @ s2)),
        "proj_rank2": (np.mean([abs(x @ wp) for x in xs]), abs(u_sat @ wp)),
        "mvdr_lsmi": (np.mean([abs(x @ w_mv) for x in xs]), abs(u_sat @ w_mv)),
    }
    for k, (leak, sat) in cases.items():
        h, s = f.acquisition(sat, leak, NOISE_STD); st[k]["hit"] += h; st[k]["sinr"].append(s); st[k]["null"].append(rb.db(leak / sat))
res = {k: dict(p_d=round(v["hit"] / N, 4), post_corr_sinr_db=round(float(np.mean(v["sinr"])), 1),
               null_rel_sat_db=round(float(np.mean(v["null"])), 1)) for k, v in st.items()}
(HERE / "results_cascade_acq.json").write_text(json.dumps(res, indent=1))
for k, v in res.items(): print(f"{k:20s} {v}")
