"""
Experiment 1: the sedenion product as a BILINEAR feature extractor (signal enters twice).

Feature:  F(tau) = (1/T) sum_t  x(t) . x(t - tau)   in R^16, for lags tau in LAGS.
  e0 component  = <x(t), x(t-tau)>  (trace of the lag-tau correlation)       -- "e0 only"
  15 imaginary  = CD-structured antisymmetric cross-channel lag correlations   -- "imag only"
Computed in one shot by the analog multiplier rig; question: is it a useful summary of the
256-number lag-correlation matrix for interference DETECTION and CLASSIFICATION?

Classes: none | CW tone | linear chirp | pulsed noise | wideband noise | BPSK jammer |
         repeater (delayed authentic PRN, strong) | spoofer (PRN, wrong delay, moderate power)
Baselines (same classifier, LDA with shrinkage):
  energy (1) | spectral kurtosis (1) | covariance eigenvalues (8) | lag-trace (len(LAGS))
  | random 16-dim projection of the full lag-correlation per lag (matched dimension)
  | full lag-correlation matrices (256 x len(LAGS), the unstructured ceiling)
"""
import json, math, pathlib, sys, importlib.util
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("okc", HERE / "okc_revised_crpa_validation.py")
okc = importlib.util.module_from_spec(spec); spec.loader.exec_module(okc)
cd_mul, r16, I16 = okc.cd_mul, okc.r16, okc.I16
L = 1023; CA = okc.CA
LAGS = [0, 1, 2, 4, 8, 16, 32]
CLASSES = ["none", "cw", "chirp", "pulsed", "wideband", "bpsk", "repeater", "spoofer"]


# ------------------------------------------------------------ batched sedenion product
MUL_T = np.zeros((16, 16, 16))
for i in range(16):
    for j in range(16):
        MUL_T[i, j] = cd_mul(I16[i], I16[j])          # e_i . e_j
_rr = np.random.default_rng(99)
RAND_T = np.zeros((16, 16, 16))
_perm = _rr.permutation(16)
for k in range(16):
    for i in range(16):
        j = _perm[(i + k) % 16]                      # Latin-square index pattern (each (i,k) one j; each (i,j) one k)
        RAND_T[i, j, k] = _rr.choice([-1.0, 1.0])
def rand_mul_batch(X, Y):
    return np.einsum("ti,tj,ijk->tk", X, Y, RAND_T)
# lower rungs of the Cayley–Dickson tower on the same 16 real coordinates: H^4 (4 quaternion blocks), O^2
def _blk_tensor(bs):
    T = np.zeros((16, 16, 16))
    for b in range(16 // bs):
        for i in range(bs):
            for j in range(bs):
                e_i = np.zeros(16); e_i[b * bs + i] = 1; e_j = np.zeros(16); e_j[b * bs + j] = 1
                prod = cd_mul(e_i[b * bs:(b + 1) * bs], e_j[b * bs:(b + 1) * bs])   # bs-dim CD product
                T[b * bs + i, b * bs + j, b * bs:(b + 1) * bs] = prod
    return T
QUAT_T = _blk_tensor(4); OCT_T = _blk_tensor(8)
def quat_mul_batch(X, Y): return np.einsum("ti,tj,ijk->tk", X, Y, QUAT_T)
def oct_mul_batch(X, Y):  return np.einsum("ti,tj,ijk->tk", X, Y, OCT_T)
def sed_mul_batch(X, Y):                              # X, Y: T x 16  ->  T x 16
    return np.einsum("ti,tj,ijk->tk", X, Y, MUL_T)


# ------------------------------------------------------------ scene generation
def waveform(kind, rng, T=L):
    t = np.arange(T)
    if kind == "cw":      return np.exp(1j * (2 * np.pi * rng.uniform(-0.2, 0.2) * t + rng.uniform(0, 2 * np.pi)))
    if kind == "chirp":   k = rng.uniform(1e-5, 2e-4); return np.exp(1j * (np.pi * k * t ** 2 + rng.uniform(0, 2 * np.pi)))
    if kind == "pulsed":  on = (np.floor(t / rng.integers(20, 80)) % 2 == 0); return on * okc.cnormal(rng, T)
    if kind == "wideband": return okc.cnormal(rng, T)
    if kind == "bpsk":    return np.sign(rng.standard_normal(T)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    if kind == "repeater": return np.roll(CA, rng.integers(2, 40)) * np.exp(1j * (2 * np.pi * rng.uniform(-0.01, 0.01) * t + rng.uniform(0, 2 * np.pi)))
    if kind == "spoofer":  return np.roll(CA, rng.integers(40, 900)) * np.exp(1j * (2 * np.pi * rng.uniform(-0.01, 0.01) * t + rng.uniform(0, 2 * np.pi)))
    raise ValueError(kind)

def scene(kind, rng, snr_sat_db=-20.0):
    th_s = rng.uniform(-60, 60); a_s = okc.steer(th_s)
    noise_std = 10 ** (-snr_sat_db / 20)
    X = np.outer(np.roll(CA, okc.TAU), a_s) + noise_std * okc.cnormal(rng, (L, 8))
    if kind != "none":
        inr_db = rng.uniform(5, 30)                                # interference-to-noise ratio
        amp = noise_std * 10 ** (inr_db / 20)
        th_j = rng.uniform(-60, 60)
        X = X + amp * np.outer(waveform(kind, rng), okc.steer(th_j))
    return X                                                      # L x 8 complex


# ------------------------------------------------------------ feature sets
rng_cproj = (np.random.default_rng(11).standard_normal((8, 64)) + 1j * np.random.default_rng(12).standard_normal((8, 64))) / 8
def feats_all(X, rng_proj):
    Xr = np.stack([r16(x) for x in X])                            # L x 16
    F_sed, F_trace, F_full, F_rand, F_rs, F_q, F_o, F_diag, F_crand = [], [], [], [], [], [], [], [], []
    for tau in LAGS:
        A, B = Xr[tau:], Xr[:L - tau]
        s = sed_mul_batch(A, B).mean(0); F_sed.append(s); F_rs.append(rand_mul_batch(A, B).mean(0))
        F_q.append(quat_mul_batch(A, B).mean(0)); F_o.append(oct_mul_batch(A, B).mean(0))
        Ac, Bc = X[tau:], X[:L - tau]
        Rc = Ac.T @ Bc.conj() / (L - tau)                         # 8x8 complex lag correlation
        F_diag.append(np.concatenate([np.diag(Rc).real, np.diag(Rc).imag]))
        F_crand.append(np.concatenate([(rng_cproj @ Rc.ravel()).real, (rng_cproj @ Rc.ravel()).imag]))
        R = A.T @ B / (L - tau)                                   # 16 x 16 lag correlation
        F_trace.append(np.trace(R)); F_full.append(R.ravel()); F_rand.append(rng_proj @ R.ravel())
    E = np.mean(Xr ** 2) * 16 + 1e-12
    F_sed = np.concatenate(F_sed) / E; F_trace = [f / E for f in F_trace]
    F_full = [f / E for f in F_full]; F_rand = [f / E for f in F_rand]; F_rs = np.concatenate(F_rs) / E
    F_q = np.concatenate(F_q) / E; F_o = np.concatenate(F_o) / E
    F_diag = np.concatenate(F_diag) / E; F_crand = np.concatenate(F_crand) / E
    e0_idx = np.array([k * 16 for k in range(len(LAGS))])
    im_idx = np.array([k * 16 + m for k in range(len(LAGS)) for m in range(1, 16)])
    R0 = Xr.T @ Xr / L
    P = np.abs(np.fft.fft(X[:, 0])) ** 2
    sk = np.mean(P ** 2) / np.mean(P) ** 2 - 2
    return {"sed_full (16/lag)": F_sed, "sed_e0_only": F_sed[e0_idx], "sed_imag_only": F_sed[im_idx],
            "energy (1)": np.array([np.log(np.mean(np.abs(X) ** 2))]), "spectral_kurtosis (1)": np.array([sk]),
            "cov_eigenvalues (8)": np.log(np.linalg.eigvalsh(X.T @ X.conj() / L).real + 1e-12),
            "lag_trace (7)": np.array(F_trace), "random_proj (16/lag)": np.concatenate(F_rand), "random_pm1_xor (16/lag)": F_rs, "quaternion_H4 (16/lag)": F_q, "octonion_O2 (16/lag)": F_o,
            "complex_diag_autocorr (16/lag)": F_diag, "complex_random_proj (16/lag)": F_crand,
            "full_lagcorr (256/lag)": np.concatenate(F_full)}


# ------------------------------------------------------------ LDA with shrinkage
def lda_fit(F, y, shrink=0.1):
    classes = np.unique(y); mu = np.stack([F[y == c].mean(0) for c in classes])
    Sw = sum(np.cov((F[y == c] - mu[i]).T, bias=True) * (y == c).sum() for i, c in enumerate(classes)) / len(y)
    Sw = np.atleast_2d(Sw); Sw = (1 - shrink) * Sw + shrink * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])
    W = np.linalg.solve(Sw, mu.T)                                 # d x C
    b = -0.5 * np.sum(mu * W.T, 1) + np.log(np.array([(y == c).mean() for c in classes]))
    return classes, W, b
def lda_predict(model, F):
    classes, W, b = model; return classes[np.argmax(F @ W + b, 1)]


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(2026); rng_proj = np.random.default_rng(7).standard_normal((16, 256)) / 16
    data = {}; labels = []
    for c in CLASSES:
        for _ in range(n_per):
            f = feats_all(scene(c, rng), rng_proj)
            for k, v in f.items(): data.setdefault(k, []).append(v)
            labels.append(c)
    y = np.array(labels); n = len(y)
    res = {}
    for seed in range(3):
        perm = np.random.default_rng(seed).permutation(n); tr, te = perm[: n // 2], perm[n // 2:]
        feats = {k: np.stack(v) for k, v in data.items()}
        # PCA-16 per lag of the full lag-correlation (fit on train): data-driven 16-dim summary
        Ffull = feats["full_lagcorr (256/lag)"]; parts = []
        for li in range(len(LAGS)):
            blk = Ffull[:, li * 256:(li + 1) * 256]; pca = PCA(16, random_state=0).fit(blk[tr]); parts.append(pca.transform(blk))
        feats["pca16_of_lagcorr (16/lag)"] = np.concatenate(parts, 1)
        for k, F in feats.items():
            mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9; Fz = (F - mu) / sd
            rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=2).fit(Fz[tr], y[tr])
            pred = rf.predict(Fz[te]); acc = float((pred == y[te]).mean())
            res.setdefault(k, dict(dim=F.shape[1], acc=[]))["acc"].append(acc)
    for k, d in sorted(res.items(), key=lambda kv: -np.mean(kv[1]["acc"])):
        d["acc_mean"] = round(float(np.mean(d["acc"])), 3); d["acc_std"] = round(float(np.std(d["acc"])), 3)
        print(f"{k:28s} dim={d['dim']:5d}  8-class acc = {d['acc_mean']:.3f} ± {d['acc_std']:.3f}", flush=True)
    (HERE / "results_bilinear_features.json").write_text(json.dumps(res, indent=1))
