# Zenodo Archival Deposit & Reproducibility Package (November 2026)

**Title:** *Beyond Associativity and Alternativity in 16D Sedenion Signal Processing: Mutually-Annihilating Filter Cascades, the Cubic Alternativity-Defect Calculus $N_u$, and Closed-Form Pontryagin Optimal Control*  
**Author:** Dionisio Chiuratto Agourakis (`jai` — `dionisio.ca@gmail.com`)  
**License:** CC-BY-4.0 (Manuscripts & Proofs) / MIT (Source Code)  
**Companion Zenodo DOIs:** [`10.5281/zenodo.23588273`](https://doi.org/10.5281/zenodo.23588273) (Concept DOI, `continues`) · [`10.5281/zenodo.23652148`](https://doi.org/10.5281/zenodo.23652148) (`v2026.10.25`, `continues`) · [`10.5281/zenodo.23588274`](https://doi.org/10.5281/zenodo.23588274) (`v1`, `references`) · [`10.5281/zenodo.23603006`](https://doi.org/10.5281/zenodo.23603006) (GNSS Anti-Jamming, `continues`) · [`10.5281/zenodo.21246673`](https://doi.org/10.5281/zenodo.21246673) (Sedenion Triangular Root, `references`)  
**Companion Software & Proof Repository:** [`https://github.com/josaum/sedenion-pg32`](https://github.com/josaum/sedenion-pg32) (`isSupplementedBy`)

---

## 1. Archival File Manifest & Verified SHA-256 Checksums

| File | Role in Deposit | SHA-256 Checksum |
|---|---|---|
| `Agourakis_2026_Alternativity_Defect_Pontryagin_and_Annihilating_Cascades.pdf` | Primary IEEEtran Preprint PDF (Primary Preview File on Zenodo) | `554996459bab987cd360177766f009d8f710b3c24983a739783e35428d9e7222` |
| `Agourakis_2026_Alternativity_Defect_Pontryagin_and_Annihilating_Cascades.tex` | IEEEtran LaTeX Source | `352a22a8d51b6fd5476d22223633598dc6522a7de042fc2bd82484faef5b692d` |
| `DARPA_DSO_STO_WhitePaper_Pontryagin_Annihilating_Cascades.pdf` | 2-Page Heilmeier Executive Concept White Paper (Uncosted) | `002e4e5ccfb9ea30cdca1e980f105ecd5eeda33ae02ad93324b6dd2e9413ca74` |
| `DARPA_DSO_STO_WhitePaper_Pontryagin_Annihilating_Cascades.tex` | 2-Page Heilmeier White Paper LaTeX Source | `e2837a4e16598650a3651b577c13af5f3723c397aeb1d776181934dd6448cc28` |
| `.zenodo.json` | Flat Zenodo Deposit Metadata (5 Related DOIs + GitHub Supplement) | `c5871c4da366ced753c66a51fb6a06cdf98f0b8840e0944b0dfa6bb5b461fdf2` |
| `upload_to_zenodo.py` | Zero-Dependency Python Deposit Helper | `7c95e3936192d5f677bae486b18ad2fccb3664e2e25ef47e267c5e4fa902a0e3` |
| `proofs/MaxwellExterior.lean` | Constructive First-Principles Lean 4.15.0 Development (`204/204` Zero-Axiom Theorems) | `efd197b622242c9ce89db1b2db73b3ab0fb95461ac470f77f52ae053222f3bb5` |
| `proofs/Axioms.lean` | Automated `#print axioms` Kernel Auditor (`ALL_204_ZERO_AXIOMS_CERTIFIED`) | `b1e3bfb967b6c50ae26c09767934b0bdd367efd7c1c5d868547be5f3c0b423d7` |
| `proofs/README.md` | Lean 4 Proof Suite Build & Verification Guide | `ab2a2b271d6e9ee1d3ac2825c2b265b34288de2bb9948d5097c8dad76c093c7a` |
| `code/sedenion_pg32_rust.rs` | Native Rust C-ABI Implementation (`N_u`, Cascades, PMP, $J_3(u)$, $[16,8,6]_3$ FEC) | `c7325ad044c9716989f0e5ebf98f00a21299f6f7161be425a99bb6647eb2391c` |
| `code/sedenion_pg32_dsp.py` | Python Reference & `ctypes` Native Bridge (`256/256` Basis Parity) | `7171dd0360e5502b2259d8b92d48ba52d2aea484c4cc66af81976a34ab1c79c7` |
| `code/prove_sedenion_merit.py` | Reproducible Quantitative Merit & Numerical Verification Suite | `74f1bd8a057c81b278b4f7caebf98c3b59bd91beb07b8e32ded06ca1aa46157c` |
| `code/sedenion_pg32_embedded.h` | Zero-Heap Header-Only C99 Library (`STM32WB55RG` Cortex-M4F & `ESP32-S2` Xtensa LX7) | `8535149fd019446898b572d637ee2f54059d9d9427142a00fb10f08bee1164b7` |
| `hardware/Sedenion_PreADC_Analog_Crossbar_Schematic.pdf` | 5-Sheet ANSI-B Engineering Schematic PDF (`SCH-NSSP-100..103` + `SCH-NSSP-200-A` 2-Stage Cascade) | `d6bc73c6fbcce76935a7b1bba180a07bc257dc525e95f4573b82ddd3d8a07c8a` |
| `hardware/Sedenion_PreADC_Analog_Crossbar_Schematic.tex` | 5-Sheet Circuitikz Schematic LaTeX Source (`256/256` Bit-Exact ROM) | `fbadfb2b57b5e1f288fd660ed8253c09a146faf5bf7a61c4efb27f3f53fb9916` |
| `hardware/sedenion_2stage_annihilating_cascade.cir` | SPICE3 Netlist: 2-Stage Mutually-Annihilating Cascade (`LT5400AIMS8E-1` + `ADL5565`) | `cc38efed9d78d8ca494b624b15c9727599f36a26fa20689f8f42de0c73571f25` |
| `hardware/sedenion_2stage_annihilating_cascade.kicad_sch` | KiCad 7/8 Schematic: 2-Stage Mutually-Annihilating Cascade (`SCH-NSSP-200-A`) | `51279d803f903c508bc99826628d90d962ee8a5775721d578287d8be99228ee6` |
| `hardware/BOM_PreADC_Analog_Crossbar.csv` | 23-Line Production CSV Bill of Materials (1-Stage + 2-Stage Cascade + RFSoC) | `ac760716548ec5f1fd6431516e678cc7c7caf9982bf1e6c186ac984996ff24f9` |
| `hardware/crossbar_84_blades_rom.json` | 84-Blade / 14-Sextet / $J_3(u)$ Hardware Control ROM JSON | `63a30f87e407e5c59b12fe8e227d9807e645f7e8f7e9111eeaa8786a9081c52e` |
| `hardware/sedenion_pontryagin_cascade.sv` | Synthesizable 2-Cycle `Q1.14` SystemVerilog RTL Core (Cascade + Pontryagin $\psi$ + `[16,8,6]_3` FEC) | `7ee70ededac9b5d4cb1643d6b4c9b857201dd159a65beb5a9e638817037c9c4f` |
| `hardware/verify_2stage_hardware_and_rtl.py` | Bit-Exact SPICE3, 256-Bit ROM & `Q1.14` RTL Golden Verification Suite | `8c704f5de98bcee39ef4af13244b4f52d1255f404151f5525b58b2426981affb` |

*(Note: All three itemized-ROM solicitation documents — `DARPA_PS_26_09_Abstract_Pontryagin_Annihilating_Cascades.pdf`, `DARPA_DSO_HR001125S0013_Abstract_NonAssociative_Pontryagin.pdf`, and `DARPA_DSO_HR001125S0013_AttachmentB_SummarySlide.pdf` — are excluded from the public Zenodo zip so proposer labor/cost tables remain restricted to DARPA portal submissions.)*

---

## 2. Theorem-to-Code & Cross-Repository Crosswalk (`376` Zero-Axiom / `516` Total Lean 4.15.0 Theorems)

| Paper Theorem | `MaxwellExterior.lean` (`204/204` Zero-Axiom) | Companion `sedenion-pg32` Bridge (`172` Zero-Ax + `140` Mathlib) | Native Rust / Embedded C Function | Verified Metric |
|---|---|---|---|---|
| **Thm 1** ($\mathbb{S}^{\mathrm{op}}$ & $S_3$ Reflection) | `cayley_dickson_opposite_anti_automorphism_and_s3_reflection_automorphism` (5.44) | `SedenionPG32/Defs.lean` (`256/256` basis parity) | `sedenion_apply_s3_psi`, `sedenion_g2_s3_invariant_j3` | $256/256$ basis pairs exact |
| **Thm 1 / §II-C** (Sextic $J_3$ & 14 Sextets) | `sextic_g2_s3_invariant_j3_and_14_box_kite_sextet_partition` (5.45) | `SedenionPG32/Main.lean` (`84` blades, `7` box-kites) | `sedenion_g2_s3_invariant_j3` | $J_3(e_1)=+1$, $J_3(e_9)=-1$, $14 \times 6 = 84$ |
| **Thm 2** ($N_u$ Symmetry, Gram & $N_u^3 = 4\Lambda^2 N_u$) | `sedenion_alternativity_defect_operator_and_gram_identity` (5.37), `alternativity_defect_cubic_minimal_polynomial_and_moore_penrose` (5.38) | `SedenionRank.square_identity` ($2(T_1 d, T_2 c) \equiv N_u$), `SedenionDim.finrank_range_Lu = 12` | `sedenion_alternativity_defect_nu`, `sedenion_svd_free_mmse`, `sedenion_moore_penrose_zd` | $L(u)^{-1}$ err $2.87\times 10^{-14}$; $L(u)^+$ err $1.45\times 10^{-15}$ |
| **Thms 3–4** (Co-Axial Dichotomy & $w_1\cdot w_2=0$ Cascade) | `non_associative_annihilating_filter_cascade_and_coaxial_dichotomy` (5.39) | `SedenionReal.nullspace` ($(a'p)b' = -n(u)p$) | `sedenion_coaxial_joint_null`, `sedenion_non_associative_cascade_null` | $w_1\cdot w_2 = 0$, jammer $0.0$, $\|[e_2,w_1,w_2]\|_2 = 2.0$ |
| **Thm 5** (Pontryagin Optimal Control $w^*$) | `pontryagin_adjoint_switching_and_quadratic_concavity_certificate` (5.42), `pontryagin_closed_form_sedenion_optimal_control_certified` (5.43) | `SedenionPG32/Hyperplane.lean` ($8+7$ bang-bang geometry) | `sedenion_pontryagin_max_control`, `sedenion_pontryagin_zd_sphere_control`, `sedenion_pontryagin_bang_bang_84` | $1.11\times 10^{-16}$ PMP residual; $0.0$ ZD-sphere leakage |
| **Thm 6** ($\mathcal{V}_9$ Isometry & $[16,8,6]_3$ FEC) | `hurwitz_radon_nine_frame_anticommutation` (5.40), `extremal_ternary_self_dual_code_16_8_6` (5.41) | `SedenionPG32/TriangularRootFreeCells.lean` (`8` ZD-free cells) | `sedenion_hurwitz_radon_spread9`, `sedenion_ternary_16_8_6_encode`, `sedenion_ternary_16_8_6_decode` | $513/513$ $\le 2$-trit error cosets corrected |

---

## 3. One-Command Reproducibility

```bash
# 1. Verify all 204 theorems in Lean 4.15.0 with ZERO axioms (0 propext, 0 Quot.sound, 0 Classical.choice, 0 sorry)
cd proofs
~/.elan/bin/lean -o MaxwellExterior.olean MaxwellExterior.lean
LEAN_PATH=. ~/.elan/bin/lean Axioms.lean
# Expected output: TOTAL=204 ZERO=204 / ALL_204_ZERO_AXIOMS_CERTIFIED

# 2. Run C-ABI / Python / Embedded-C cross-verification & merit suite
cd ..
uv run pytest tests/scripts/test_pure_logic.py
uv run python scripts/prove_sedenion_merit.py
```

---

## 4. Minting the Zenodo DOI (`upload_to_zenodo.py` or `curl`)

### Option A — One-Command Python Helper (Identical to `sedenion-pg32/zenodo/upload_to_zenodo.py`)

```bash
export ZENODO_TOKEN="<YOUR_ZENODO_PERSONAL_ACCESS_TOKEN>"
python3 docs/sedenion_pg32_papers_and_patent/upload_to_zenodo.py \
  docs/sedenion_pg32_papers_and_patent/Agourakis_2026_Alternativity_Defect_Pontryagin_and_Annihilating_Cascades.pdf \
  docs/sedenion_pg32_papers_and_patent/Agourakis_2026_Zenodo_Deposit_Alternativity_Pontryagin_ZeroAxiom.zip
# Add --sandbox to test on sandbox.zenodo.org first, or --publish to mint DOI immediately.
```

### Option B — Direct `curl` Bucket API

```bash
export ZENODO_TOKEN="<YOUR_ZENODO_PERSONAL_ACCESS_TOKEN>"
cd docs/sedenion_pg32_papers_and_patent

DEPOSIT_JSON=$(curl -s -X POST "https://zenodo.org/api/deposit/depositions" \
  -H "Authorization: Bearer ${ZENODO_TOKEN}" \
  -H "Content-Type: application/json" \
  -d @zenodo_deposit_2026_alternativity_pontryagin.json)

BUCKET_URL=$(echo "${DEPOSIT_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['links']['bucket'])")
DEPOSIT_ID=$(echo "${DEPOSIT_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

curl -X PUT "${BUCKET_URL}/Agourakis_2026_Alternativity_Defect_Pontryagin_and_Annihilating_Cascades.pdf" \
  -H "Authorization: Bearer ${ZENODO_TOKEN}" \
  --upload-file Agourakis_2026_Alternativity_Defect_Pontryagin_and_Annihilating_Cascades.pdf

curl -X PUT "${BUCKET_URL}/Agourakis_2026_Zenodo_Deposit_Alternativity_Pontryagin_ZeroAxiom.zip" \
  -H "Authorization: Bearer ${ZENODO_TOKEN}" \
  --upload-file Agourakis_2026_Zenodo_Deposit_Alternativity_Pontryagin_ZeroAxiom.zip

echo "Draft ready for inspection & DOI minting at: https://zenodo.org/uploads/${DEPOSIT_ID}"
```
