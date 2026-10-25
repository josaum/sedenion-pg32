# Pivot 1 + Pivot 2: Dual-Pol Sedenion Jammer Excision — Lean proofs and simulation (2026-10-25)

Everything below is reproduced by `lean/` (Lean 4.15.0 + Mathlib v4.15.0, `lake build`, `lake env lean Axioms.lean`)
and `bench/gps_jamming/run_dualpol_pivot12.py` (numpy only; writes `results_dualpol_pivot12.json`).

## What was proven (SedenionRobust.lean, no `sorry`, axioms = propext / Classical.choice / Quot.sound)

| Theorem | Statement |
|---|---|
| `right_mul_adjoint` | ⟨x·v, y⟩ = ⟨x, y·v*⟩, i.e. R(v)ᵀ = R(v*): the reverse path of the 32-port S-matrix is the same passive network with the conjugate weight (reciprocity). |
| `SedenionReal.nullspace` (imported) | (a,b) imaginary, ‖a‖=‖b‖=1, ⟨a,b⟩=0, c ⊥ {1,a,b,ab} ⇒ (a,b)·(c,(ac)b) = 0. This is Theorem 5.1 of the PG(3,2) paper, already in `bench/lean-real`; `SedenionRobust` now imports it instead of re-proving it. |
| `SedenionReal.partner_norm` (imported) | ‖(ac)b‖² = ‖c‖², hence ‖(c,(ac)b)‖² = 2‖c‖². |
| `dualpol_colinear_preservation` | Jammer on the cone annihilated AND ‖(a_sat,0)·v‖² = ‖a_sat‖² for the unit annihilator (0.0000 dB), any a_sat (Δθ = 0° included). |
| `dualpol_passive_rf_stage` | For v_rf = v/√2: ‖X·v_rf‖² ≤ ‖X‖² for all X (passive), jammer still annihilated, satellite passes at ‖a_sat‖²/2. |
| `jam_linear_on_cone` | Under the fixed wiring RHCP I/Q→(e1,e2), LHCP I/Q→(e5,e6), every sample of a tilt-ψ linearly polarised wave is on the Moreno cone. |
| `dualpol_tilt_annihilation` | **One fixed** v(ψ) = e3 + sin2ψ·e12 − cos2ψ·e15 annihilates **every complex sample** X_J(ψ,s) of the jammer. |
| `v_tilt_norm` | ‖v(ψ)‖² = 2. |
| `dualpol_tilt_colinear_preservation` | Closed-form Δθ = 0° theorem: jammer waveform nulled, ‖(a_sat,0)·v(ψ)‖² = 2‖a_sat‖² (unit weight ⇒ exactly ‖a_sat‖²). |
| `dualpol_tilt_passive_rf_stage` | v_rf(ψ) = v(ψ)/2: passive on all inputs, nulls the waveform, satellite at ‖a_sat‖²/2; nonzero couplings {1/2, ±sin2ψ/2, ∓cos2ψ/2}. |

The pre-existing hypothesis-form theorems (`rhcp_satellite_norm_preservation`, `lhcp_satellite_norm_preservation`,
`colinear_dualpol_jammer_null_and_sat_preserve`, `rf_smatrix_passivity_bound`) are unchanged.

`SedenionReal.lean` itself needed a repair to import: `crpa_antipartner_product` and `spoofer_half_sphere_passband` did not compile on Mathlib v4.15.0 (their `linarith` step saw mismatched atoms). They are now proven from `nullspace` through three `scale`-bilinearity lemmas and compile with the standard axioms. `AxiomsReal.lean` still imports a `SedenionDim` module that is not in the repository.

## What the simulation showed

**Pivot 1 (S-matrix).** Canonical blade: singular values of S21 = R(v)/√2 are 1⁽⁴⁾, 0.7071⁽⁸⁾, 0⁽⁴⁾;
min eig(I − S32ᵀS32) = +2.2e-16; S32 = S32ᵀ exactly; couplings per port = [2] with magnitude [0.5].
A general-position v(c) is equally passive and symmetric but has **14 couplings per port**; the 2-coupler stripline claim holds for the canonical / tilt blades only.

**Pivot 2 — a trap, then the fix.** With the LHCP I/Q pair wired into (e4,e5) the jammer's I and Q samples have **no common annihilator**
(common-kernel dim 0): a single zero divisor nulls one sample, not the waveform. Wired into the partner plane (e5,e6) the common kernel is
2-dimensional and contains the closed-form v(ψ) above. This is a fixed design-time wiring, not a data-dependent Householder gauge, so the “T = I” claim stands with that caveat.

**Δθ = 0°.** Satellite loss 0.0000 dB; jammer suppression 321.6 dB (float64 floor, δ = 0 case).

**Fair baseline (honest result).** A conventional dual-pol 2-port LCMV (wᴴp_sat = 1, wᴴp_J = 0) also nulls the jammer at Δθ = 0° with 0 dB
distortion, and both it and the sedenion stage + LS readout pay the same **3.01 dB noise-gain penalty**.
The sedenion advantage over spatial MVDR is real (polarisation separability, angle-independent); the advantage over polarisation-domain LCMV is not in SINR — it is in the hardware story (passive, pre-ADC, 2 couplers per lane) and in the proofs.

| Δθ | spatial MVDR (8-el ULA) SINR | dual-pol sedenion | dual-pol LCMV |
|---|---|---|---|
| 0.0° | -80.00 dB | -3.01 dB | -3.01 dB |
| 0.5° | -24.18 dB | -3.01 dB | -3.01 dB |
| 1.0° | -18.19 dB | -3.01 dB | -3.01 dB |
| 2.0° | -12.26 dB | -3.01 dB | -3.01 dB |
| 5.0° | -4.89 dB | -3.01 dB | -3.01 dB |
| 10.0° | -0.79 dB | -3.01 dB | -3.01 dB |
| 30.0° | -0.02 dB | -3.01 dB | -3.01 dB |

(input SNR 0 dB, J/N 80 dB)

**Robustness.** Jammer axial ratio (not perfectly linear) → leaves the cone; suppression tracks `leakage_bound`:

| axial ratio | suppression | −10log₁₀(2‖δ‖²) |
|---|---|---|
| 40 dB | 37.0 dB | 37.0 dB |
| 30 dB | 27.0 dB | 27.0 dB |
| 20 dB | 17.0 dB | 17.0 dB |
| 10 dB | 7.4 dB | 7.3 dB |
| 6 dB | 4.0 dB | 3.7 dB |
| 3 dB | 1.8 dB | 1.3 dB |

Satellite loss vs receive-antenna XPD: 40 dB → 0.024 dB, 30 dB → 0.076 dB, 25 dB → 0.136 dB, 20 dB → 0.244 dB, 15 dB → 0.434 dB.
Coupler calibration error ‖δ_v‖: 1e-04 → 79.8 dB (bound 77.0), 1e-03 → 59.3 dB (bound 57.0), 1e-02 → 40.7 dB (bound 37.0), 3e-02 → 31.8 dB (bound 27.4), 1e-01 → 19.9 dB (bound 17.0).

**CRLB, GPS L1 C/A code phase** (β_rms from a ±1 MHz front end, T = 1 ms, C/N0 = 45 dB-Hz, J/N = 80 dB):

| Δθ | no mitigation | spatial MVDR | dual-pol sedenion | dual-pol LCMV |
|---|---|---|---|---|
| 0.0° | 126997 m | 126997.2 m | 17.96 m | 17.96 m |
| 1.0° | 126997 m | 103.1 m | 17.96 m | 17.96 m |
| 5.0° | 126997 m | 22.3 m | 17.96 m | 17.96 m |
| 30.0° | 126997 m | 12.7 m | 17.96 m | 17.96 m |

## Limits to state in the abstract / TAES text
- Polarisation-only excision needs the jammer to be (near-)linearly polarised; a circularly polarised jammer is off the cone and is not excised.
- The 2-coupler-per-lane passive network is for the canonical / tilt blades; general-position Moreno blades need 14 couplings per lane.
- The 0 dB satellite insertion loss is a distortionless-response property, shared with polarisation LCMV; the SINR cost (3.01 dB noise gain) is identical.
- v(ψ) depends on the jammer tilt ψ (two PIN-diode-selectable coupler values, sin2ψ/2 and cos2ψ/2); tilt estimation error enters through `leakage_bound`.
