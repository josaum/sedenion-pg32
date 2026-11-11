# Sedenion Algebras and PG(3,2) Incidence Structures for Non-Associative Geometric Deep Learning

[![DOI: Alternativity & Pontryagin](https://zenodo.org/badge/DOI/10.5281/zenodo.24124222.svg)](https://doi.org/10.5281/zenodo.24124222)
[![DOI: Main Paper (latest)](https://zenodo.org/badge/DOI/10.5281/zenodo.23588273.svg)](https://doi.org/10.5281/zenodo.23588273)
[![DOI: GNSS Anti-Jamming](https://zenodo.org/badge/DOI/10.5281/zenodo.23603006.svg)](https://doi.org/10.5281/zenodo.23603006)
[![DOI: Free Cells](https://zenodo.org/badge/DOI/10.5281/zenodo.21665205.svg)](https://doi.org/10.5281/zenodo.21665205)
[![DOI: Sedenion Triangular Root](https://zenodo.org/badge/DOI/10.5281/zenodo.21246673.svg)](https://doi.org/10.5281/zenodo.21246673)

Reference implementation, Lean 4 formalisations (312 machine-checked theorems in `lean/` — 172 zero-axiom theorems in `lean/pg32/` and 140 real-analytic theorems in `lean/real/` — plus 204 zero-axiom companion hardware/RF/optimal-control theorems in `v3sp3r/proofs/MaxwellExterior.lean`, **516 total**, **376 with zero axioms**, zero `sorry`), and benchmark suite for:

1. **Pedro Barone Agourakis & Dionisio Chiuratto Agourakis** (July 7, 2026). *The Sedenion Triangular Root: A Child-Originated Partial Desimplicialization Operator on Cayley–Dickson Bitmasks*. Zenodo. [`10.5281/zenodo.21246673`](https://doi.org/10.5281/zenodo.21246673)
2. **Pedro Barone Agourakis & Dionisio Chiuratto Agourakis** (July 29, 2026). *Zero-Divisor-Free Coordinate Cells in the Cayley–Dickson Tower*. Zenodo. [`10.5281/zenodo.21665205`](https://doi.org/10.5281/zenodo.21665205)
3. **Dionisio Chiuratto Agourakis** (2026). *Sedenion Algebras and PG(3,2) Incidence Structures for Non-Associative Geometric Deep Learning: Algebraic Foundations, Machine-Checked Theorems, and Dual-Track Validation*. Zenodo. Concept DOI [`10.5281/zenodo.23588273`](https://doi.org/10.5281/zenodo.23588273) (v2026.10.25: [`10.5281/zenodo.23652148`](https://doi.org/10.5281/zenodo.23652148); v1: [`10.5281/zenodo.23588274`](https://doi.org/10.5281/zenodo.23588274))
4. **Dionisio Chiuratto Agourakis** (2026). *Sedenion PG(3,2) Zero-Divisor Beamforming for GNSS Anti-Jamming and Anti-Spoofing: Matrix-Inversion-Free Null Steering, Associative Quaternion Phase Hinges, and Neural CRPA Validation*. Zenodo. [`10.5281/zenodo.23603006`](https://doi.org/10.5281/zenodo.23603006)
5. **Dionisio Chiuratto Agourakis** (November 11, 2026). *Beyond Associativity and Alternativity in 16D Sedenion Signal Processing: Mutually-Annihilating Filter Cascades, the Cubic Alternativity-Defect Calculus $N_u$, and Closed-Form Pontryagin Optimal Control*. Zenodo Concept DOI [`10.5281/zenodo.24124221`](https://doi.org/10.5281/zenodo.24124221) (v1.0.0: [`10.5281/zenodo.24124222`](https://doi.org/10.5281/zenodo.24124222) · [PDF](papers/Agourakis_2026_Alternativity_Defect_Pontryagin_and_Annihilating_Cascades.pdf) · [Zenodo Package](zenodo/alternativity_pontryagin_2026/))

## Layout

```
papers/            manuscripts and proposals (LaTeX + PDF)
sedenion_pg32/     Python package: Cayley–Dickson tables, adjoint autograd, PG(3,2) router,
                   Moreno projection, constructive nullspace partner, Triton kernel + PyTorch mirror
tests/             pytest suite for the package
benchmarks/        algebraic_baselines.py (Table 2), track_a/ (Yang–Mills BCJ), track_b/ (deontic
                   multi-hop pruning, No-FFN ablation), ext/ (ProofWriter, ListOps), gps_jamming/
                   (CRPA null steering, phase-2 defences, dual-pol excision, Q1.14 RTL)
lean/pg32/         finite development: Lean 4.15 core only (172 theorems across Core, Geometry, Census,
                   TriangularRootFreeCells; 172/172 exhaustive #print axioms checks), 172/172 theorems with zero axioms ("does not depend on any axioms"; 0 propext, 0 Classical.choice, 0 Quot.sound)
lean/real/         real-analytic development: Lean 4.15 + Mathlib v4.15.0 (140 theorems; 60 top-level #print axioms checks)
                   SedenionReal.lean   — Theorem 5.1 nullspace, Hurwitz composition, vertical quaternion blocks Q_{u,3}, SO(2) bivector invariance, CRPA theorems
                   SedenionAlt/Rank/Dim — alternativity, kernel characterisation, rank L(u) = 12 (rank–nullity)
                   SedenionRobust.lean — universal adjoint L(w)^T = L(w*), R(x)^T = R(x*), leakage bounds, 8⊕4⊕4 transceiver, dual-pol RHCP/LHCP excision
reports/           analysis notes that back numbers in the papers
zenodo/            deposit metadata, packages and upload scripts (dual-pol supplement + 2026 Alternativity/Pontryagin zero-axiom bundle)
```

| Document / Hardware Artifact | Source | PDF / Netlist |
|---|---|---|
| **Beyond Associativity & Alternativity: $N_u$ Calculus, $w_1\cdot w_2=0$ Cascades & Pontryagin Control (IEEE TSP Manuscript)** | [`papers/Agourakis_2026_Alternativity_Defect_Pontryagin_and_Annihilating_Cascades.tex`](papers/Agourakis_2026_Alternativity_Defect_Pontryagin_and_Annihilating_Cascades.tex) | [**PDF**](papers/Agourakis_2026_Alternativity_Defect_Pontryagin_and_Annihilating_Cascades.pdf) · [Zenodo DOI `10.5281/zenodo.24124222`](https://doi.org/10.5281/zenodo.24124222) · [Zenodo Bundle](zenodo/alternativity_pontryagin_2026/Agourakis_2026_Zenodo_Deposit_Alternativity_Pontryagin_ZeroAxiom.zip) |
| **DARPA-PS-26-09 STO Proposal Abstract — Pre-ADC Analog Crossbar (5 pp.)** | [`papers/DARPA_PS_26_09_Abstract_Native_Sedenion_Analog_Crossbar.tex`](papers/DARPA_PS_26_09_Abstract_Native_Sedenion_Analog_Crossbar.tex) | [**PDF**](papers/DARPA_PS_26_09_Abstract_Native_Sedenion_Analog_Crossbar.pdf) |
| **DARPA STO / MTO / DSO & ONR Global Heilmeier White Paper (2 pp.)** | [`papers/DARPA_ONR_WhitePaper_Sedenion_GNSS.tex`](papers/DARPA_ONR_WhitePaper_Sedenion_GNSS.tex) | [**PDF**](papers/DARPA_ONR_WhitePaper_Sedenion_GNSS.pdf) |
| **Native 16D Sedenion Signal Processing (IEEE TSP Manuscript, 6 pp.)** | [`papers/Agourakis_2026_Native_Sedenion_Signal_Processing.tex`](papers/Agourakis_2026_Native_Sedenion_Signal_Processing.tex) | [**PDF**](papers/Agourakis_2026_Native_Sedenion_Signal_Processing.pdf) |
| **GNSS CRPA Anti-Jamming & Anti-Spoofing (IEEE TAES Manuscript)** | [`papers/Agourakis_2026_Sedenion_GNSS_AntiJamming.tex`](papers/Agourakis_2026_Sedenion_GNSS_AntiJamming.tex) | [**PDF**](papers/Agourakis_2026_Sedenion_GNSS_AntiJamming.pdf) |
| **Foundational PG(3,2) & Non-Associative GDL Manuscript (16 pp.)** | [`papers/Agourakis_2026_Sedenion_PG32.tex`](papers/Agourakis_2026_Sedenion_PG32.tex) | [**PDF**](papers/Agourakis_2026_Sedenion_PG32.pdf) |
| **1-Stage & 2-Stage Mutually-Annihilating Analog Crossbar Suite (5-Sheet PDF, KiCad, SPICE3, BOM)** | [`hardware/Native_Sedenion_Hardware_BOM_and_Schematic.md`](hardware/Native_Sedenion_Hardware_BOM_and_Schematic.md) | [**5-Sheet Schematic PDF**](hardware/schematic/Sedenion_PreADC_Analog_Crossbar_Schematic.pdf) · [1-Stage KiCad](hardware/schematic/sedenion_analog_crossbar.kicad_sch) · [2-Stage Cascade KiCad](hardware/schematic/sedenion_2stage_annihilating_cascade.kicad_sch) · [2-Stage SPICE](hardware/schematic/sedenion_2stage_annihilating_cascade.cir) · [84-Blade + $J_3$ ROM](hardware/crossbar_84_blades_rom.json) |
| **SystemVerilog RTL Cores: 1-Cycle Beamformer & 2-Cycle Pontryagin / Cascade / `[16,8,6]_3` FEC** | [`benchmarks/gps_jamming/rtl/sedenion_pontryagin_cascade.sv`](benchmarks/gps_jamming/rtl/sedenion_pontryagin_cascade.sv) | [1-Cycle Beamformer](benchmarks/gps_jamming/rtl/sedenion_crpa_beamformer.sv) · [Golden RTL/SPICE Verifier](hardware/verify_2stage_hardware_and_rtl.py) · [Verification JSON](benchmarks/gps_jamming/results_phase2_defense.json) |
| **Background IP & USPTO Provisional Patent Specification** | [`patent/USPTO_Provisional_Patent_Native_Sedenion_DSP.tex`](patent/USPTO_Provisional_Patent_Native_Sedenion_DSP.tex) | [**USPTO PDF**](patent/USPTO_Provisional_Patent_Native_Sedenion_DSP.pdf) · [Cover PTO/SB/16](patent/PTO_SB_16_Provisional_Cover_Sheet.pdf) · [Micro-Entity PTO/SB/15A](patent/PTO_SB_15A_Micro_Entity_Certification.pdf) |
| **INPI Brazil Patent Specification (Convenção de Paris Art. 4)** | [`patent/INPI_Pedido_Patente_Sedenion_DSP_PTBR.tex`](patent/INPI_Pedido_Patente_Sedenion_DSP_PTBR.tex) | [**INPI PDF**](patent/INPI_Pedido_Patente_Sedenion_DSP_PTBR.pdf) · [Partes `e-Patentes`](patent/inpi_partes/) |
| **Dual-Pol Pivot 1 & Pivot 2 Report (Passive $S_{32}$ & $\Delta\theta=0^\circ$ Excision)** | [`reports/DUALPOL_PIVOT12_REPORT.md`](reports/DUALPOL_PIVOT12_REPORT.md) | [Results JSON](benchmarks/gps_jamming/results_dualpol_pivot12.json) |

## Quick start

```bash
pip install torch pytest onnx onnxruntime
PYTHONPATH=. python -m pytest -q tests                 # algebraic & geometric unit tests (incl. 0.0 ULP sign-bit XOR)
python benchmarks/algebraic_baselines.py               # Table 2: CLR, PSI, rank L(u) = 12 (CPU, < 1 min)
python benchmarks/track_a/train.py                     # Table 2: Track A 4-gluon Yang–Mills BCJ & collinear pole
python benchmarks/track_b/train.py                     # Table 3: Track B multi-hop contradiction pruning & No-FFN ablation
python benchmarks/ext/proofwriter.py                   # Table 4: ProofWriter OOD depth & Birds-Electricity
python benchmarks/ext/listops.py                       # Table 4: Long-Range Arena ListOps OOD tree reduction
python benchmarks/gps_jamming/run_gps_bench.py         # companion paper: +80 dB JSR CRPA null steering, RTK hinges, anti-spoofing
python benchmarks/gps_jamming/run_native_sedenion_dsp.py # NSSP: K=1 associator spoofer detector, 84-blade SPICE crossbar, ZDKM, Q1.14 RTL
python benchmarks/gps_jamming/run_phase2_defense.py    # mutual-coupling gauge, LSMI K=4..64 comparison, M=1..4 multi-jammer, XCZU28DR RTL
python benchmarks/gps_jamming/run_dualpol_pivot12.py out.json   # dual-pol RHCP/LHCP excision, passive S32-matrix, Δθ=0° CRLB (numpy only)
```

Benchmark scripts write their `results_*.json` next to themselves; the committed JSON files are the runs reported in the papers.

## Lean 4 verification

```bash
# finite PG(3,2) development: 8+7 Fano bifurcation, 672/2520 associator census, BCJ stratum theorems
cd lean/pg32 && lake build SedenionPG32 && LEAN_PATH=.lake/build/lib lean Axioms.lean     # 172/172 theorems "does not depend on any axioms" (zero axioms)

# real-analytic development (Mathlib): nullspace, CRPA gains, leakage bounds, dual-pol excision
cd lean/real && lake exe cache get && lake build && lake env lean Axioms.lean            # propext, Classical.choice, Quot.sound only
```

`lean/real/Axioms.lean` audits every theorem the papers cite from the real development, including the
§5.5 rank–nullity chain (`SedenionAlt` → `SedenionRank` → `SedenionDim`: `square_identity`,
`adjoint_identity`, `kernel_char`, `ker_Lu_eq`, `finrank_ker_Lu = 4`, `finrank_range_Lu = 12`).

## Citation

```bibtex
@misc{agourakis2026triangular,
  author       = {Agourakis, Pedro Barone and Agourakis, Dionisio Chiuratto},
  title        = {{The Sedenion Triangular Root: A Child-Originated Partial Desimplicialization Operator on Cayley--Dickson Bitmasks}},
  year         = {2026},
  month        = jul,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21246673},
  url          = {https://doi.org/10.5281/zenodo.21246673}
}

@misc{agourakis2026freecells,
  author       = {Agourakis, Pedro Barone and Agourakis, Dionisio Chiuratto},
  title        = {{Zero-Divisor-Free Coordinate Cells in the Cayley--Dickson Tower}},
  year         = {2026},
  month        = jul,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21665205},
  url          = {https://doi.org/10.5281/zenodo.21665205}
}

@misc{agourakis2026sedenion,
  author       = {Agourakis, Dionisio Chiuratto},
  title        = {{Sedenion Algebras and PG(3,2) Incidence Structures for Non-Associative Geometric Deep Learning: Algebraic Foundations, Machine-Checked Theorems, and Dual-Track Validation}},
  year         = {2026},
  month        = oct,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23588274},
  url          = {https://doi.org/10.5281/zenodo.23588274}
}

@misc{agourakis2026gnss,
  author       = {Agourakis, Dionisio Chiuratto},
  title        = {{Sedenion PG(3,2) Zero-Divisor Beamforming for GNSS Anti-Jamming and Anti-Spoofing: Matrix-Inversion-Free Null Steering, Associative Quaternion Phase Hinges, and Neural CRPA Validation}},
  year         = {2026},
  month        = oct,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23603006},
  url          = {https://doi.org/10.5281/zenodo.23603006}
}

@misc{agourakis2026pontryagin,
  author       = {Agourakis, Dionisio Chiuratto},
  title        = {{Beyond Associativity and Alternativity in 16D Sedenion Signal Processing: Mutually-Annihilating Filter Cascades, the Cubic Alternativity-Defect Calculus $N_u$, and Closed-Form Pontryagin Optimal Control}},
  year         = {2026},
  month        = nov,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24124222},
  url          = {https://doi.org/10.5281/zenodo.24124222},
  note         = {Concept DOI: 10.5281/zenodo.24124221; 204/204 zero-axiom Lean 4.15.0 theorems (376 zero-axiom / 516 total unified across sedenion-pg32 + v3sp3r)}
}
```
