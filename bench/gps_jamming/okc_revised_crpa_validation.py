#!/usr/bin/env python3
"""Revised CRPA benchmark: complex baseband, arbitrary phase, and paired acquisition.

NumPy only. Run directly with Python even though this distribution uses a .txt suffix:
    python revised_crpa_validation.py.txt --trials 128 --output revised_results

The script preserves the submitted Sedenion algebra but adds constraints for the
orthogonal phase quadrature. A single 4-D fibre cannot impose two complex jammer
nulls plus phase lock. The dual-jammer experiment therefore tests a two-stage
cascade explicitly, without assuming that the second stage preserves the first
null under estimation error.
"""
import argparse
import csv
import json
import math
import pathlib

import numpy as np

I8 = np.eye(8)
I16 = np.eye(16)
J16 = np.block([[np.zeros((8, 8)), -I8], [I8, np.zeros((8, 8))]])
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])
IQ = np.array([1, 2])
L = 1023
TAU = 317


def cd_conj(x):
    y = -x.copy()
    y[..., 0] = x[..., 0]
    return y


def cd_mul(x, y):
    n = x.shape[-1]
    if n == 1:
        return x * y
    h = n // 2
    a, b = x[..., :h], x[..., h:]
    c, d = y[..., :h], y[..., h:]
    return np.concatenate(
        [cd_mul(a, c) - cd_mul(cd_conj(d), b),
         cd_mul(d, a) + cd_mul(b, cd_conj(c))], axis=-1)


def runit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cunit(x):
    return x / np.linalg.norm(x)


def r16(z):
    return np.concatenate([z.real, z.imag])


def c8(x):
    return x[:8] + 1j * x[8:]


def steer(theta_deg):
    idx = np.arange(8)
    return np.exp(1j * np.pi * idx * np.sin(math.radians(theta_deg))) / math.sqrt(8)


def householder(x, y):
    w = x - y
    if np.linalg.norm(w) < 1e-14:
        return np.eye(16)
    w = w / np.linalg.norm(w)
    return np.eye(16) - 2 * np.outer(w, w)


def cnormal(rng, shape=()):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2)


def fiber_basis(rng):
    """Four kernel vectors at canonical u_zd=(e3,e12)/sqrt(2)."""
    u = np.zeros(16)
    u[3] = u[12] = 1 / math.sqrt(2)
    a, b = u[:8] * math.sqrt(2), u[8:] * math.sqrt(2)
    ab = cd_mul(a, b)
    basis = []
    while len(basis) < 4:
        c = np.zeros(8)
        c[1:] = rng.standard_normal(7)
        for q in (a, b, ab):
            c -= (c @ q) / (q @ q) * q
        for q in basis:
            c -= (c @ q) * q
        c[0] = 0
        if np.linalg.norm(c) > 1e-6:
            basis.append(runit(c))
    return np.stack([np.concatenate([c, cd_mul(cd_mul(a, c), b)]) for c in basis])


def sed_stage(u_hat, sat_ref, rng, quadrature=None):
    """Real 16x16 Sedenion stage with optional quadrature and phase constraints.

    u_hat itself is annihilated over all 16 outputs. If quadrature is supplied,
    its two I/Q readout components are constrained to zero. Satellite Q is zero.
    """
    u = runit(u_hat)
    uzd = np.zeros(16)
    uzd[3] = uzd[12] = 1 / math.sqrt(2)
    uanti = np.zeros(16)
    uanti[3], uanti[12] = 1 / math.sqrt(2), -1 / math.sqrt(2)
    T1 = householder(u, uzd)
    sat_perp = sat_ref - (u @ sat_ref) * u
    if np.linalg.norm(sat_perp) < 1e-10:
        raise ValueError("Satellite and jammer signatures are numerically collinear")
    T2 = householder(runit(T1 @ sat_perp), uanti)
    T = T2 @ T1
    V = fiber_basis(rng)
    ys = np.stack([cd_mul(T @ sat_ref, vk) for vk in V])
    rows = [ys[:, 2]]
    if quadrature is not None:
        yq = np.stack([cd_mul(T @ quadrature, vk) for vk in V])
        rows += [yq[:, 1], yq[:, 2]]
    Lc = np.stack(rows)
    _, _, vt = np.linalg.svd(Lc)
    null = vt[len(rows):]
    g_i = ys[:, 1]
    alpha = null.T @ (null @ g_i)
    if np.linalg.norm(alpha) < 1e-12:
        alpha = null[0]
    alpha = runit(alpha)
    if alpha @ g_i < 0:
        alpha = -alpha
    v = runit(alpha @ V)
    M = np.stack([cd_mul(T @ e, v) for e in I16], axis=1)
    return M


def sed_single(u_hat, sat, rng, complex_null):
    q = r16(1j * c8(u_hat)) if complex_null else None
    M = sed_stage(u_hat, sat, rng, q)
    return M[IQ, :]


def sed_cascade(u1, u2, sat, rng):
    q1, q2 = r16(1j * c8(u1)), r16(1j * c8(u2))
    M1 = sed_stage(u1, sat, rng, q1)
    M2 = sed_stage(M1 @ u2, M1 @ sat, rng, M1 @ q2)
    return (M2 @ M1)[IQ, :]


def complex_lcmv(jammers, sat):
    A = np.stack([c8(x) for x in jammers] + [c8(sat)], axis=1)
    f = np.zeros(A.shape[1], complex)
    f[-1] = 1
    w = A @ np.linalg.solve(A.conj().T @ A, f)
    return w_to_h(w)


def complex_projection(jammers, sat):
    Q, _ = np.linalg.qr(np.stack([c8(x) for x in jammers], axis=1))
    z = c8(sat)
    w = z - Q @ (Q.conj().T @ z)
    return w_to_h(w)


def complex_mvdr(R, sat):
    w = np.linalg.solve(R + 1e-2 * np.eye(8), c8(sat))
    return w_to_h(w)


def w_to_h(w):
    return np.stack([np.concatenate([w.real, w.imag]),
                     np.concatenate([-w.imag, w.real])])


def h_to_z(row):
    return row[0] + 1j * row[1]


def noise_gain(H):
    """RMS I/Q gain for independent equal-variance real input components."""
    return math.sqrt((np.linalg.norm(H[0]) ** 2 + np.linalg.norm(H[1]) ** 2) / 2)


def complex_linearity_error(H):
    return np.linalg.norm(H @ J16 - J2 @ H, "fro") / max(np.linalg.norm(H, "fro"), 1e-15)


def apply_phase(H, z, phase):
    return h_to_z(H @ r16(z * phase))


def null_and_carrier_metrics(H, sat, jammers, phases):
    zs = h_to_z(H @ sat)
    if abs(zs) < 1e-12:
        raise ValueError("Satellite output is numerically zero")
    nulls = []
    for jammer in jammers:
        z = c8(jammer)
        residuals = [abs(apply_phase(H, z, p)) / abs(zs) for p in phases]
        nulls.append(-20 * math.log10(max(math.sqrt(np.mean(np.square(residuals))), 1e-16)))
    z0 = c8(sat)
    carrier_errors = [abs(apply_phase(H, z0, p) - zs * p) / abs(zs) for p in phases]
    return nulls, math.sqrt(np.mean(np.square(carrier_errors))), zs


def ca_code():
    g1, g2 = [1] * 10, [1] * 10
    out = []
    for _ in range(L):
        out.append(1.0 if (g1[9] ^ (g2[1] ^ g2[5])) == 0 else -1.0)
        f1 = g1[2] ^ g1[9]
        f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1, g2 = [f1] + g1[:9], [f2] + g2[:9]
    return np.array(out, float)


CA = ca_code()
CA_SHIFT = np.roll(CA, TAU)


def acquire(sat_amp, leak_amp, jam_amp, noise_sigma, jam_noise, thermal_noise):
    y = CA_SHIFT * sat_amp + jam_amp * leak_amp * jam_noise + noise_sigma * thermal_noise
    corr = np.abs(np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(CA)))) / L
    hit = int(np.argmax(corr) == TAU)
    off = np.ones(L, bool)
    off[TAU] = False
    peak_offpeak = 10 * math.log10(max(corr[TAU] ** 2 / max(np.mean(corr[off] ** 2), 1e-30), 1e-30))
    sinr = 10 * math.log10(max(L * sat_amp ** 2 / max((jam_amp * leak_amp) ** 2 + noise_sigma ** 2, 1e-30), 1e-30))
    return hit, peak_offpeak, sinr


def make_scene(rng, dual=False):
    while True:
        th1 = rng.uniform(-60, 60)
        ths = rng.uniform(-60, 60)
        u1, us = steer(th1), steer(ths)
        corr = abs(np.vdot(u1, us))
        if 0.1 <= corr <= 0.3:
            break
    if not dual:
        return [u1], us, [th1]
    while True:
        th2 = rng.uniform(-60, 60)
        u2 = steer(th2)
        if abs(th2 - th1) >= 10 and abs(np.vdot(u2, us)) < 0.5:
            return [u1, u2], us, [th1, th2]


def estimate_with_error(z, eps, rng):
    return cunit(z + eps * cunit(cnormal(rng, 8)))


def controlled_trial(rng, eps=1e-3):
    jammers, sat, _ = make_scene(rng, False)
    u = estimate_with_error(jammers[0], eps, rng)
    uh, sh = r16(u), r16(sat)
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, 16, endpoint=False))
    methods = {
        "sed_real_signature_only": sed_single(uh, sh, rng, False),
        "sed_complex_iq": sed_single(uh, sh, rng, True),
        "lcmv_complex": complex_lcmv([uh], sh),
    }
    rows = []
    for name, H in methods.items():
        nulls, carrier_error, zs = null_and_carrier_metrics(H, sh, [r16(jammers[0])], phases)
        rows.append(dict(method=name, null1_db=nulls[0], carrier_relative_error=carrier_error,
                         complex_linearity_error=complex_linearity_error(H),
                         noise_gain=noise_gain(H), sat_amplitude=abs(zs)))
    return rows


def heldout_data(rng, dual=True, K=64, eval_K=16):
    jammers, sat, angles = make_scene(rng, dual)
    J = len(jammers)
    powers = [80.0, 76.0][:J]
    amps = [10 ** (p / 20) for p in powers]
    # Stationary spatial signatures isolate arbitrary-phase suppression from
    # drift. Evaluation uses new temporal realizations, not training samples.
    train_angles = [np.full(K, th) for th in angles]
    eval_angles = [np.full(eval_K, th) for th in angles]
    X = np.zeros((K, 8), complex)
    for j in range(J):
        X += np.stack([amps[j] * cnormal(rng) * steer(th) for th in train_angles[j]])
    X += 10 * cnormal(rng, (K, 8))
    X += np.sign(rng.standard_normal((K, 1))) * sat[None, :]
    # Rows store x^T and beam output is w^H x, hence R=E[x x^H].
    R = X.T @ X.conj() / K
    evals, V = np.linalg.eigh(R)
    estimates = [V[:, -j - 1] for j in range(J)]
    eval_signatures = [[steer(th) for th in seq] for seq in eval_angles]
    return estimates, sat, eval_signatures, R


def heldout_trial(rng, phases):
    estimates, sat, eval_signatures, R = heldout_data(rng, True)
    sh = r16(sat)
    u1, u2 = r16(estimates[0]), r16(estimates[1])
    methods = {
        "sed_2stage_complex_experimental": sed_cascade(u1, u2, sh, rng),
        "lcmv_complex": complex_lcmv([u1, u2], sh),
        "projection_complex": complex_projection([u1, u2], sh),
        "mvdr_complex": complex_mvdr(R, sh),
    }
    # Common random numbers make acquisition comparisons paired across methods.
    jam_noise = cnormal(rng, L)
    thermal_noise = cnormal(rng, L)
    rows = []
    for name, H in methods.items():
        ng = noise_gain(H)
        zs = h_to_z(H @ sh)
        residuals = [[], []]
        carrier_errors = []
        for t in range(len(eval_signatures[0])):
            signatures = [eval_signatures[j][t] for j in range(2)]
            for p in phases:
                for j in range(2):
                    residuals[j].append(abs(apply_phase(H, signatures[j], p)) / abs(zs))
                carrier_errors.append(abs(apply_phase(H, sat, p) - zs * p) / abs(zs))
        nulls = [-20 * math.log10(max(math.sqrt(np.mean(np.square(r))), 1e-16)) for r in residuals]
        leak_rms = math.sqrt(np.mean(np.concatenate(residuals) ** 2))
        sat_norm = abs(zs) / ng
        leak_norm = leak_rms * sat_norm
        hit, peak_offpeak, sinr = acquire(sat_norm, leak_norm, 1e4, 10.0,
                                          jam_noise, thermal_noise)
        rows.append(dict(method=name, null1_db=nulls[0], null2_db=nulls[1],
                         carrier_relative_error=math.sqrt(np.mean(np.square(carrier_errors))),
                         complex_linearity_error=complex_linearity_error(H), noise_gain=ng,
                         sat_amplitude=abs(zs), normalized_sat_amplitude=sat_norm,
                         normalized_leak_rms=leak_norm, detection_hit=hit,
                         peak_to_offpeak_db=peak_offpeak, postcorr_sinr_db=sinr))
    return rows


def wilson(hits, n, z=1.959963984540054):
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def distribution(values, bootstrap_rng):
    x = np.asarray(values, float)
    n = len(x)
    idx = bootstrap_rng.integers(0, n, (2000, n))
    means = x[idx].mean(axis=1)
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.quantile(x, 0.1)), p90=float(np.quantile(x, 0.9)),
                mean_ci95=[float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))])


def summarize(controlled, heldout, trials, seed):
    bootstrap_rng = np.random.default_rng(seed + 991)
    out = {"config": {"trials_per_experiment": trials, "seed": seed,
                      "signal_model": "proper complex baseband; arbitrary temporal phase",
                      "heldout": "64 training snapshots; 16 new stationary-scene snapshots; disjoint temporal realizations",
                      "single_stage_dual_complex_feasibility": {
                          "available_real_constraints": 4,
                          "required_real_constraints_for_two_complex_nulls_and_phase_lock": 7,
                          "used": "experimental two-stage cascade; dual-complex feasibility remains unproven"}},
           "controlled_single_jammer": {}, "heldout_dual_jammer": {}}
    for method in sorted({r["method"] for group in controlled for r in group}):
        rows = [r for group in controlled for r in group if r["method"] == method]
        out["controlled_single_jammer"][method] = {
            key: distribution([r[key] for r in rows], bootstrap_rng)
            for key in ["null1_db", "carrier_relative_error", "complex_linearity_error", "noise_gain"]
        }
    for method in sorted({r["method"] for group in heldout for r in group}):
        rows = [r for group in heldout for r in group if r["method"] == method]
        entry = {key: distribution([r[key] for r in rows], bootstrap_rng)
                 for key in ["null1_db", "null2_db", "carrier_relative_error",
                             "complex_linearity_error", "noise_gain", "postcorr_sinr_db",
                             "peak_to_offpeak_db"]}
        hits = sum(r["detection_hit"] for r in rows)
        entry["detection"] = {"p_d": hits / len(rows), "wilson_ci95": wilson(hits, len(rows))}
        out["heldout_dual_jammer"][method] = entry
    return out


def write_raw(path, controlled, heldout):
    fields = ["experiment", "trial", "method", "null1_db", "null2_db",
              "carrier_relative_error", "complex_linearity_error", "noise_gain",
              "sat_amplitude", "normalized_sat_amplitude", "normalized_leak_rms",
              "detection_hit", "peak_to_offpeak_db", "postcorr_sinr_db"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for experiment, rows in [("controlled", controlled), ("heldout", heldout)]:
            for trial, group in enumerate(rows):
                for row in group:
                    writer.writerow(dict(experiment=experiment, trial=trial, **row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("revised_results"))
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be at least 2")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, 8, endpoint=False))
    controlled, heldout = [], []
    for trial in range(args.trials):
        controlled.append(controlled_trial(rng))
        heldout.append(heldout_trial(rng, phases))
        if (trial + 1) % 16 == 0 or trial + 1 == args.trials:
            print(f"completed {trial + 1}/{args.trials} paired trials", flush=True)
    summary = summarize(controlled, heldout, args.trials, args.seed)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    write_raw(args.output / "raw_trials.csv", controlled, heldout)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
