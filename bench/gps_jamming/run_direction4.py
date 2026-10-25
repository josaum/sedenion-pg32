"""
Direction 4: PG(3,2)-indexed weight codebooks and switching.

Part A: the incidence structure as a transition graph.
  points (15): any two points lie on exactly one line  -> point graph is complete (K15)
  lines  (35): two lines are adjacent if they meet      -> degree 18 of 34
Part B: codebook experiment.  A jammer slews across [-60, 60] deg; the receiver holds a codebook of
  N entries (N = 15, 35), each a 2x16 real readout H_i precomputed for a DOA cell, selected by cell.
  Entries: (i) sedenion block stage (cell-centre gauge), (ii) WL-LCMV, (iii) complex LCMV;
  cell layouts: uniform in angle (PG-indexed = uniform with a relabelling), Lloyd VQ on steering vectors.
  Metrics over the sweep: mean and worst-case suppression, satellite gain/phase jump at every switch.
"""
import json, math, pathlib, sys, importlib.util, itertools
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
bs = load("run_block_stage"); rect = bs.rect; okc = bs.okc
r16, c8, J16 = bs.r16, bs.c8, bs.J16


# ------------------------------------------------------------------ Part A: PG(3,2) graphs
def pg32():
    pts = list(range(1, 16))
    lines = sorted({tuple(sorted((i, j, i ^ j))) for i in pts for j in pts if i < j})
    return pts, lines

def part_a():
    pts, lines = pg32()
    on_line = {(i, j): sum(1 for L in lines if i in L and j in L) for i in pts for j in pts if i < j}
    deg_line = [sum(1 for M in lines if M != L and set(L) & set(M)) for L in lines]
    return dict(n_points=len(pts), n_lines=len(lines),
                point_pairs_on_exactly_one_line=all(v == 1 for v in on_line.values()),
                point_graph="complete K15 (every transition allowed)",
                line_graph_degree=sorted(set(deg_line)), line_graph_edges=sum(deg_line) // 2)


# ------------------------------------------------------------------ Part B: codebooks
def entry(kind, centre_deg, sat):
    u = r16(okc.steer(centre_deg)); sh = r16(sat)
    if kind == "sed_block":  return bs.sed_block_cascade([u, J16 @ u], sh)
    if kind == "wl_lcmv":    return rect.wl_lcmv([u, J16 @ u], sh)
    if kind == "complex_lcmv": return okc.complex_lcmv([u], sh)
    raise ValueError(kind)

def centres_uniform(N):
    edges = np.linspace(-60, 60, N + 1); return (edges[:-1] + edges[1:]) / 2, edges

def centres_vq(N, rng, iters=50):
    """Lloyd VQ on steering vectors (distance 1 - |<a_i, a_j>|^2), 1-D DOA; returns centres and edges."""
    samples = np.linspace(-60, 60, 2000); A = np.stack([okc.steer(t) for t in samples])
    c = np.sort(rng.choice(samples, N, replace=False))
    for _ in range(iters):
        C = np.stack([okc.steer(t) for t in c]); d = 1 - np.abs(A.conj() @ C.T) ** 2
        lab = np.argmin(d, 1)
        c = np.array([samples[lab == k].mean() if np.any(lab == k) else c[k] for k in range(N)]); c.sort()
    edges = np.concatenate([[-60], (c[:-1] + c[1:]) / 2, [60]])
    return c, edges

def sweep(kind, centres, edges, sat, slew_pts=1200, phases=8):
    Hs = [entry(kind, t, sat) for t in centres]
    sh = r16(sat); ph = np.exp(1j * np.linspace(0, 2 * np.pi, phases, endpoint=False))
    thetas = np.linspace(-60, 60, slew_pts); supp = []; gains = []; idx_prev = None; switches = []
    for th in thetas:
        k = min(int(np.searchsorted(edges, th, side="right") - 1), len(centres) - 1); H = Hs[k]
        zs = okc.h_to_z(H @ sh)
        leak = math.sqrt(np.mean([abs(okc.apply_phase(H, okc.steer(th), p)) ** 2 for p in ph])) / abs(zs)
        supp.append(-20 * math.log10(max(leak, 1e-16))); gains.append(zs)
        if idx_prev is not None and k != idx_prev:
            z0, z1 = gains[-2], gains[-1]
            switches.append((abs(math.degrees(np.angle(z1 / z0))), 20 * math.log10(abs(z1) / abs(z0))))
        idx_prev = k
    ng = np.mean([okc.noise_gain(H) for H in Hs])
    return dict(mean_supp_db=round(float(np.mean(supp)), 1), worst_supp_db=round(float(np.min(supp)), 1),
                p10_supp_db=round(float(np.percentile(supp, 10)), 1),
                switch_phase_jump_deg_max=round(float(max(s[0] for s in switches)), 4) if switches else 0.0,
                switch_gain_jump_db_max=round(float(max(abs(s[1]) for s in switches)), 3) if switches else 0.0,
                n_switches=len(switches), mean_noise_gain=round(float(ng), 3))


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    res = {"A_incidence_graphs": part_a()}
    print("== A:", res["A_incidence_graphs"])
    sat = okc.steer(25.0)
    for N in [15, 35]:
        for layout in ["uniform (PG-indexed)", "lloyd_vq"]:
            centres, edges = centres_uniform(N) if layout.startswith("uniform") else centres_vq(N, rng)
            for kind in ["sed_block", "wl_lcmv", "complex_lcmv"]:
                key = f"N={N} {layout} {kind}"; res[key] = sweep(kind, centres, edges, sat)
                print(f"   {key:40s} {res[key]}", flush=True)
    # reference: continuous (no codebook) tracking with the same methods
    for kind in ["sed_block", "wl_lcmv"]:
        centres = np.linspace(-60, 60, 1200); edges = np.concatenate([[-60], (centres[:-1] + centres[1:]) / 2, [60]])
        key = f"continuous (1200 cells) {kind}"; res[key] = sweep(kind, centres, edges, sat)
        print(f"   {key:40s} {res[key]}")
    (HERE / "results_direction4.json").write_text(json.dumps(res, indent=1))
