"""
Fair-baseline rerun of Table 2 (Scenario I) for the TAES revision.

Asymmetry removed: EVERY method estimates the jammer signature from the SAME K noisy,
drifting snapshots.  The sedenion nuller takes u_hat = dominant eigenvector of R_hat
(then Moreno-projects it); MVDR/LSMI invert R_hat; orthogonal projection uses u_hat.
Null depth is evaluated against the jammer signature at the LAST snapshot (drift included).

Also: half-sphere key modulation vs. a continuous-gain projection beam (same sweep shape?).

numpy only.  Writes results_fair_baseline.json.
"""
import json, math, pathlib, importlib.util
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rb", HERE / "run_robustness.py")
rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)
cd_mul, unit, moreno_project, nullspace_partner, householder, db, steer, c8_to_r16 = (
    rb.cd_mul, rb.unit, rb.moreno_project, rb.nullspace_partner, rb.householder, rb.db, rb.steer, rb.c8_to_r16)
rng = np.random.default_rng(2026)

G2_TAPS = {1: (2, 6)}
def ca_code(prn=1):
    g1 = [1]*10; g2 = [1]*10; t1, t2 = G2_TAPS[prn]; out = []
    for _ in range(1023):
        out.append(1.0 if (g1[9] ^ (g2[t1-1] ^ g2[t2-1])) == 0 else -1.0)
        f1 = g1[2] ^ g1[9]; f2 = g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9]
        g1 = [f1]+g1[:9]; g2 = [f2]+g2[:9]
    return np.array(out)
CA = ca_code(); L = 1023; TAU = 317; CA_SHIFT = np.roll(CA, TAU)

def acquisition(sat_gain, jam_leak, noise_std):
    """Correlate 1023 chips: satellite at amplitude sat_gain, jammer leakage amplitude jam_leak."""
    jam_amp = 10 ** (80 / 20)
    y = CA_SHIFT * sat_gain + rng.standard_normal(L) * jam_amp * jam_leak + rng.standard_normal(L) * noise_std / 4
    corr = np.abs(np.fft.irfft(np.fft.rfft(y) * np.conj(np.fft.rfft(CA)), n=L)) / L
    hit = int(np.argmax(corr) == TAU)
    off = np.ones(L, bool); off[TAU] = False
    sinr = 10 * math.log10(max(corr[TAU]**2 / max(np.mean(corr[off]**2), 1e-20), 1e-20))
    return hit, sinr


def table2_fair(n_trials=256, jsr_db=80.0, snr_db=-20.0, drift_deg_total=0.5):
    noise_std = 10 ** (-snr_db / 20); jam_amp = 10 ** (jsr_db / 20)
    methods = ["MVDR-SMI (K, delta=1e-2)", "Orth. projection (u_hat from R_hat)", "Sed-PG32-Moreno (u_hat from R_hat)",
               "Sed-PG32-Moreno (clean u, paper's Table 2)"]
    out = {}
    for K in [4, 16, 64]:
        st = {m: dict(supp=[], sinr=[], hits=0) for m in methods}
        for _ in range(n_trials):
            th0 = rng.uniform(-60, 60); th_s = th0 + (20 if th0 < 35 else -20)
            u_sat = unit(c8_to_r16(steer(th_s)))
            thetas = th0 + np.linspace(0, drift_deg_total, K)
            X = np.stack([rng.standard_normal() * jam_amp * c8_to_r16(steer(t)) for t in thetas]) \
                + rng.standard_normal((K, 16)) * noise_std / 4 \
                + np.outer(CA_SHIFT[:K], u_sat)          # satellite present in the snapshots too
            x_now = unit(c8_to_r16(steer(thetas[-1])))
            R = X.T @ X / K
            evals, evecs = np.linalg.eigh(R); u_hat = unit(evecs[:, -1])
            if u_hat @ x_now < 0: u_hat = -u_hat
            for m in methods:
                if m.startswith("MVDR"):
                    w = unit(np.linalg.solve(R + 1e-2 * np.eye(16), u_sat))
                    leak = abs(x_now @ w); sat = abs(u_sat @ w)
                elif m.startswith("Orth"):
                    w = unit(u_sat - (u_hat @ u_sat) * u_hat)
                    leak = abs(x_now @ w); sat = abs(u_sat @ w)
                else:
                    src = u_hat if "R_hat" in m else x_now
                    uzd = moreno_project(src + 1e-3 * rng.standard_normal(16))
                    T = householder(unit(src), uzd)
                    v = nullspace_partner(uzd, rng.standard_normal(7))
                    out_j = cd_mul(T @ x_now, v); out_s = cd_mul(T @ u_sat, v)
                    s_hat = unit(out_s)                      # single matched beam, fair vs scalar beamformers
                    leak = abs(out_j @ s_hat); sat = np.linalg.norm(out_s)
                st[m]["supp"].append(db(leak))
                h, s = acquisition(sat, leak, noise_std); st[m]["hits"] += h; st[m]["sinr"].append(s)
        out[f"K={K}"] = {m: dict(null_depth_db=round(float(np.mean(s["supp"])), 1),
                                 post_corr_sinr_db=round(float(np.mean(s["sinr"])), 1),
                                 p_det=round(s["hits"] / n_trials, 3)) for m, s in st.items()}
    return out


def halfsphere_vs_projection(n_trials=256):
    """Does half-sphere key modulation do anything a continuous projection sweep cannot?"""
    phis = np.linspace(0, math.pi, 9); sed = np.zeros(len(phis)); prj = np.zeros(len(phis)); nrm = np.zeros(len(phis))
    for _ in range(n_trials):
        u = moreno_project(rng.standard_normal(16))
        a = u[:8] * math.sqrt(2); b = u[8:] * math.sqrt(2)
        # two orthonormal seeds c, c_perp in W_ab^perp
        def seed():
            v = nullspace_partner(u, rng.standard_normal(7)); return v[:8] * math.sqrt(2)
        c = unit(seed()); cp = seed(); cp = unit(cp - (cp @ c) * c)
        p = cd_mul(cd_mul(a, c), b); pp = cd_mul(cd_mul(a, cp), b)
        x = unit(rng.standard_normal(16))  # a generic spoofer-direction probe... use u itself as the query
        for i, ph in enumerate(phis):
            k = np.concatenate([c, -math.cos(ph) * p + math.sin(ph) * pp]) / math.sqrt(2)
            sed[i] += np.linalg.norm(cd_mul(u, k)); nrm[i] += np.linalg.norm(k)
            # projection analogue: beam w(ph) = cos(ph/2) w_pass + sin(ph/2) w_null, unit norm
            w_pass = u; w_null = unit(x - (x @ u) * u)
            w = math.cos(ph / 2) * w_pass + math.sin(ph / 2) * w_null
            prj[i] += abs(u @ w)
    return [dict(phi_deg=round(math.degrees(p), 1), sed_gain=round(float(s / n_trials), 4),
                 sed_key_norm=round(float(n / n_trials), 4), proj_gain=round(float(q / n_trials), 4),
                 sqrt_1_plus_cos=round(math.sqrt(1 + math.cos(p)), 4))
            for p, s, n, q in zip(phis, sed, nrm, prj)]


if __name__ == "__main__":
    res = dict(table2_fair=table2_fair(), halfsphere_vs_projection=halfsphere_vs_projection())
    (HERE / "results_fair_baseline.json").write_text(json.dumps(res, indent=1))
    for K, d in res["table2_fair"].items():
        print("\n==", K)
        for m, v in d.items(): print(f"  {m:45s} {v}")
    print("\n== half-sphere vs projection sweep")
    for r in res["halfsphere_vs_projection"]: print(" ", r)
