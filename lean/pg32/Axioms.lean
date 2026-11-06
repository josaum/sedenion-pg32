import SedenionPG32
open SedenionPG32

/-!
# Exhaustive Kernel Axiom Audit (`#print axioms`) for `SedenionPG32`

Verifies that **all 167 / 167 theorems** across `Core.lean`, `Geometry.lean`,
`Census.lean`, and `TriangularRootFreeCells.lean` depend on **zero axioms**
(`does not depend on any axioms` — 0 `propext`, 0 `Classical.choice`, 0 `Quot.sound`,
0 `Lean.ofReduceBool`, 0 `sorryAx`).
-/

-- Core.lean (13 theorems)
#print axioms eps_unit_left
#print axioms eps_unit_right
#print axioms eps_square
#print axioms eps_values
#print axioms adjoint_left_identity
#print axioms adjoint_right_identity
#print axioms epsT_eq_eps
#print axioms int_mul_zero
#print axioms mulR_eq_mul
#print axioms basis_mul
#print axioms zero_divisor_1_10
#print axioms zero_divisor_in_A9
#print axioms kernel_witness_4

-- Geometry.lean (95 theorems)
#print axioms H_card
#print axioms H_closed
#print axioms apex_in_alt_planes
#print axioms H8_is_octonion_core
#print axioms apex_not_in_zd_planes
#print axioms zd_plane_octonion_part_is_polar
#print axioms polar_line_contains_p_iff_even
#print axioms zd_plane_twist_part
#print axioms lines_card
#print axioms line_counts
#print axioms seven_lines_per_plane
#print axioms mixed_lines_per_plane
#print axioms hinge_card
#print axioms falt_cyclic_assoc_invariant
#print axioms fzd_stratum_bcj_pole_cancellation
#print axioms hinge_is_projective_line
#print axioms gnss_pll_hinge_associative
#print axioms crpa_null_and_antipartner_4
#print axioms crpa_multi_jammer_simultaneous_null
#print axioms pre_adc_crossbar_eq_full_sedenion_mul
#print axioms hardware_crossbar_84_blades_336_partners
#print axioms cascaded_peeler_two_stage_kernel
#print axioms cascaded_peeler_two_stage_kernel_physical
#print axioms native_associator_moreno_curvature_dichotomy
#print axioms associator_quaternion_triad_orthogonality
#print axioms morenoPartner_eq_basis
#print axioms svd_free_moreno_nuller_168_identity
#print axioms quaternion_hinge_associativity_all_7_lines
#print axioms Z.S_succ
#print axioms Z.S_add
#print axioms Z.S_zero_right
#print axioms Z.S_zero_left
#print axioms Z.repr
#print axioms Z.canon
#print axioms Z.S_canon
#print axioms Z.zero_add_negOfNat
#print axioms Z.negOfNat_add_negOfNat
#print axioms Z.ofNat_add_negOfNat
#print axioms Z.negOfNat_add_ofNat
#print axioms Z.add_canon
#print axioms Z.S_add_S
#print axioms Z.add_comm
#print axioms Z.add_assoc
#print axioms Z.add_zero
#print axioms Z.zero_add
#print axioms Z.neg_canon
#print axioms Z.neg_S
#print axioms Z.add_right_neg
#print axioms Z.neg_add
#print axioms Z.sub_self
#print axioms Z.sub_zero
#print axioms Z.add_sub_cancel
#print axioms Z.nat_add_mul
#print axioms Z.nat_mul_assoc
#print axioms Z.negOfNat_mul_ofNat
#print axioms Z.negOfNat_mul_negOfNat
#print axioms Z.ofNat_mul_negOfNat'
#print axioms Z.mul_canon
#print axioms Z.shift_left
#print axioms Z.shift_right
#print axioms Z.S_mul_S
#print axioms Z.mul_comm
#print axioms Z.mul_add
#print axioms Z.add_mul
#print axioms Z.negOfNat_mul_negSucc
#print axioms Z.mul_assoc
#print axioms Z.neg_mul
#print axioms Z.mul_neg
#print axioms Z.neg_mul_neg
#print axioms Z.nat_add_sub_cancel
#print axioms Z.zero_div
#print axioms Z.zero_mod
#print axioms Z.mul_div_cancel
#print axioms Z.mul_mod_left
#print axioms hinge_snap_zero_te_residual_and_exact_recovery
#print axioms hinge_snap_idempotent
#print axioms glitch_coalesce_preserves_total_duration
#print axioms multi_glitch_cascade_duration_conservation
#print axioms lfrfid_stock_dma_underflow_on_glitch
#print axioms lfrfid_pg32_dma_underflow_immune
#print axioms term_scale_left
#print axioms term_scale_right
#print axioms sq_scale
#print axioms sum16_scale_left
#print axioms sum16_scale_right
#print axioms mul_scale_left
#print axioms mul_scale_right
#print axioms mul_scale_left_right
#print axioms sum16_sq_scale
#print axioms vnormSq_mul_scale_left
#print axioms q14_fixed_point_moreno_zero_lsb_identity
#print axioms cross_band_inductive_associator_dichotomy
#print axioms nfc_1356mhz_subcarrier_and_uid_bcc_invariants
#print axioms kronecker_multitile_16x16_two_stage_gain
#print axioms wifi_multipath_peeler_step_energy_reduction

-- Census.lean (38 theorems)
#print axioms internal_pair_9
#print axioms internal_pair_10
#print axioms internal_pair_11
#print axioms internal_pair_12
#print axioms internal_pair_13
#print axioms internal_pair_14
#print axioms internal_pair_15
#print axioms collinear_row_1
#print axioms collinear_row_2
#print axioms collinear_row_3
#print axioms collinear_row_4
#print axioms collinear_row_5
#print axioms collinear_row_6
#print axioms collinear_row_7
#print axioms collinear_row_8
#print axioms collinear_row_9
#print axioms collinear_row_10
#print axioms collinear_row_11
#print axioms collinear_row_12
#print axioms collinear_row_13
#print axioms collinear_row_14
#print axioms collinear_row_15
#print axioms census_row_1
#print axioms census_row_2
#print axioms census_row_3
#print axioms census_row_4
#print axioms census_row_5
#print axioms census_row_6
#print axioms census_row_7
#print axioms census_row_8
#print axioms census_row_9
#print axioms census_row_10
#print axioms census_row_11
#print axioms census_row_12
#print axioms census_row_13
#print axioms census_row_14
#print axioms census_row_15
#print axioms associator_census

-- TriangularRootFreeCells.lean (21 theorems)
#print axioms pedro_toy_triangular_root_identities
#print axioms geometric_root_domain_66_and_ranks
#print axioms geometric_root_left_inverse_all_66
#print axioms support_zd_criterion_equivalence_all_66
#print axioms zd_filtered_domain_30_plus_36_by_rank
#print axioms bitmask_universe_trichotomy_32767
#print axioms paper1_examples_10_1_to_10_5
#print axioms order_ideal_and_ghost_filter_duality_all_66
#print axioms ghost_line_annihilator_multiples_of_four
#print axioms cayley_dickson_tower_census_n4_n5_n6
#print axioms doubling_generator_degree_separation
#print axioms tower_absoluteness_n4_in_n5_all_66
#print axioms sequential_prefix_chunk_root_left_inverse_and_62_domain
#print axioms gl32_six_free_orbits_decomposition_a4
#print axioms pathion_a5_pg42_155_lines_29_free_126_ghost_and_23_blocks
#print axioms two_pow_succ_add
#print axioms generator_degree_step_difference_all_n
#print axioms cayley_dickson_tower_step_recurrences_n4_to_n12
#print axioms chingon_a6_pg52_651_lines_60_free_591_ghost_and_54_blocks
#print axioms gl32_orbit_decomposition_a4_a5_a6
#print axioms all_8_maximal_free_blocks_algebraic_and_ghost_dichotomy
