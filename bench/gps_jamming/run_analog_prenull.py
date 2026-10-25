"""
Experiment 2: pre-ADC (analog baseband) jammer removal with a fixed +-1 network and per-channel gains.

Chain:  antennas -> I/Q demod -> per-channel complex gain D (phase shifter + attenuator, quantised,
        with hardware error) -> FIXED passive/op-amp network N -> ADC (b bits, AGC to peak) -> digital
        WL-MVDR (SMI, K=64) on the ADC channels -> correlator.
Networks:
  sed_blade : N = R(v), v = (e_c - e_d)/sqrt2 primitive blade: 16 outputs, kernel 4-D.  D steers the
              jammer (2 real directions {u, Ju}) into ker R(v) -- all 16 ADC channels jammer-free.
  difference: D makes the jammer common-mode (D = diag(1/a_j)), N = adjacent differences: 7 complex
              = 14 real outputs, jammer-free.  The textbook baseline.
  none      : D = I, N = I (digital-only CRPA; the ADC sees the full jammer).
Metrics vs JSR and ADC bits: pre-ADC jammer suppression, ADC clipping fraction, post-correlation
SINR and P_d after digital beamforming.  Hardware: 6-bit phase, 0.25 dB attenuator steps, 2 deg / 0.2 dB
random error.  One jammer (circular), one satellite.
"""
import json, math, pathlib, sys, importlib.util
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bs = load("run_block_stage"); okc = bs.okc; rect = bs.rect
r16, c8, J16, I16, cd_mul = bs.r16, bs.c8, bs.J16, bs.I16, bs.cd_mul

# fixed blade network R(v), v = (e_1 - e_10)/sqrt2  (a primitive zero divisor), and its kernel
V_BLADE = np.zeros(16); V_BLADE[1] = 1 / math.sqrt(2); V_BLADE[10] = -1 / math.sqrt(2)
R_BLADE = np.stack([cd_mul(I16[i], V_BLADE) for i in range(16)], 1)
_u, _s, _vt = np.linalg.svd(R_BLADE); K_BLADE = _vt[12:].T          # 16 x 4
# dense network from a generic cone point (kernel admits ~constant-modulus vectors on channels 1-7)
U_DENSE = np.load(HERE / "analog_prenull_cone_point.npy")
R_DENSE = np.stack([cd_mul(I16[i], U_DENSE) for i in range(16)], 1)
_u2, _s2, _vt2 = np.linalg.svd(R_DENSE); K_DENSE = _vt2[12:].T
_rk = np.random.default_rng(5); _best = None; _bm = 0
for _ in range(4000):
    _k = c8(K_DENSE @ _rk.standard_normal(4)); _m = np.abs(_k[1:]); _r = _m.min() / _m.max()
    if _r > _bm: _bm, _best = _r, _k / np.abs(_k[1:]).max()
K_DENSE_VEC = _best                                                   # fixed target kernel vector, |k_0| = 0


def quantise_gain(g, phase_bits=6, att_step_db=0.25, phase_err_deg=2.0, gain_err_db=0.2, rng=None):
    ph = np.round(np.angle(g) / (2 * np.pi / 2 ** phase_bits)) * (2 * np.pi / 2 ** phase_bits)
    mag_db = np.round(20 * np.log10(np.abs(g) + 1e-15) / att_step_db) * att_step_db
    if rng is not None:
        ph = ph + np.radians(phase_err_deg) * rng.standard_normal(len(g))
        mag_db = mag_db + gain_err_db * rng.standard_normal(len(g))
    return 10 ** (mag_db / 20) * np.exp(1j * ph)


def design_D(network, a_j_hat, rng):
    """per-channel complex gains that put the jammer into the network's kernel."""
    if network == "none":
        return np.ones(8, complex)
    if network == "difference":
        g = 1.0 / a_j_hat                                           # jammer becomes (1,...,1): common mode
    elif network == "sed_dense":
        g = K_DENSE_VEC / a_j_hat                                   # channel 0 -> gain 0 (antenna dropped)
    elif network == "sed_blade":
        # want D a_j in ker R(v) AND D (i a_j) in ker R(v): choose k in K (2-D complex-structured part)
        # pick k in K_BLADE with J k also in K_BLADE (the kernel is J-invariant for cone points), unit modulus-ish
        K = K_BLADE
        # find k = K alpha maximising min channel modulus (so attenuators stay in range)
        best, bestm = None, -1
        for _ in range(400):
            al = rng.standard_normal(4); k = K @ al; kc = c8(k)
            m = np.min(np.abs(kc)) / (np.linalg.norm(kc) + 1e-15)
            if m > bestm: best, bestm = kc / np.linalg.norm(kc), m
        g = best / a_j_hat
    g = g / np.max(np.abs(g))                                       # attenuators only (max gain 0 dB)
    gq = quantise_gain(g, rng=rng); gq[np.abs(g) < 1e-9] = 0.0; return gq


def apply_network(network, Y):                                      # Y: T x 8 complex (after D)
    if network == "none":       return np.stack([r16(y) for y in Y])                 # T x 16 real
    if network == "difference": return np.stack([r16(y[1:] - y[:-1]) for y in Y])   # T x 14 real
    if network == "sed_blade":  return np.stack([r16(y) for y in Y]) @ R_BLADE.T     # T x 16 real
    if network == "sed_dense":  return np.stack([r16(y) for y in Y]) @ R_DENSE.T     # T x 16 real (rank 12)


def adc(Z, bits, agc_headroom_db=6.0):
    """b-bit uniform quantiser, full scale set by AGC at the peak over channels (+ headroom)."""
    fs = np.max(np.abs(Z)) * 10 ** (agc_headroom_db / 20); step = 2 * fs / 2 ** bits
    Zq = np.clip(np.round(Z / step), -2 ** (bits - 1), 2 ** (bits - 1) - 1) * step
    clip = float(np.mean(np.abs(Z) >= fs * (1 - 2 / 2 ** bits)))
    return Zq, clip


def trial(rng, network, jsr_db, bits, K=64, snr_db=-20.0, eps=1e-3):
    th_j = rng.uniform(-60, 60); th_s = th_j + (20 if th_j < 35 else -20)
    a_j = okc.steer(th_j) * np.exp(1j * rng.uniform(0, 2 * np.pi)); a_s = okc.steer(th_s)
    a_j_hat = okc.estimate_with_error(a_j, eps, rng) * np.linalg.norm(a_j)
    amp = 10 ** (jsr_db / 20); noise = 10 ** (-snr_db / 20)
    T = okc.L
    jam = amp * okc.cnormal(rng, T); n = noise * okc.cnormal(rng, (T, 8))
    sat = np.roll(okc.CA, okc.TAU) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    X = np.outer(jam, a_j) + np.outer(sat, a_s) + n
    D = design_D(network, a_j_hat, rng)
    Y = X * D[None, :]
    Z = apply_network(network, Y)
    # pre-ADC suppression: jammer power at network output relative to input, per unit satellite
    Zj = apply_network(network, np.outer(jam, a_j) * D[None, :]); Zs = apply_network(network, np.outer(sat, a_s) * D[None, :])
    pre_supp = 10 * math.log10(np.mean(np.abs(np.outer(jam, a_j)) ** 2) / max(np.mean(Zj ** 2), 1e-30)) \
               - 10 * math.log10(np.mean(np.abs(np.outer(sat, a_s)) ** 2) / max(np.mean(Zs ** 2), 1e-30))
    Zq, clip = adc(Z, bits)
    # digital stage: real-linear MVDR-SMI on the ADC channels (satellite steering mapped through D and N)
    s_dir = apply_network(network, np.ones((1, 1)) * a_s[None, :] * D[None, :])[0]
    s_dirJ = apply_network(network, (1j * np.ones((1, 1))) * a_s[None, :] * D[None, :])[0]
    R = Zq[:K].T @ Zq[:K] / K + 1e-3 * np.trace(Zq[:K].T @ Zq[:K] / K) / Zq.shape[1] * np.eye(Zq.shape[1])
    A = np.stack([s_dir, s_dirJ], 1); Ri = np.linalg.pinv(R)
    W = Ri @ A @ np.linalg.pinv(A.T @ Ri @ A)                       # LCMV: W^T s_dir = e1, W^T s_dirJ = e2
    out = Zq @ W; y = out[:, 0] + 1j * out[:, 1]
    corr = np.abs(np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(okc.CA)))) / T
    hit = int(np.argmax(corr) == okc.TAU); off = np.ones(T, bool); off[okc.TAU] = False
    p2o = 10 * math.log10(max(corr[okc.TAU] ** 2 / max(np.mean(corr[off] ** 2), 1e-30), 1e-30))
    return dict(pre_adc_supp_db=pre_supp, clip_frac=clip, peak_to_offpeak_db=p2o, hit=hit, n_channels=Z.shape[1])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    rng = np.random.default_rng(2026); res = {}
    for bits in [8, 12]:
        for jsr in [60, 80, 100]:
            for net in ["none", "difference", "sed_dense"]:
                rows = [trial(rng, net, jsr, bits) for _ in range(n)]
                d = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
                res[f"bits={bits} JSR={jsr} {net}"] = d
                print(f"bits={bits:2d} JSR=+{jsr} {net:11s} pre-ADC supp {d['pre_adc_supp_db']:6.1f} dB | clip {d['clip_frac']:.3f} | "
                      f"peak/off {d['peak_to_offpeak_db']:6.1f} dB | Pd {d['hit']:.3f} | ch {int(d['n_channels'])}", flush=True)
    (HERE / "results_analog_prenull.json").write_text(json.dumps(res, indent=1))
