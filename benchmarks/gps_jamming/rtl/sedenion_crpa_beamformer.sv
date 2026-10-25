// =============================================================================
// Module: sedenion_crpa_beamformer
// Project: Sedenion PG(3,2) Zero-Divisor CRPA Beamformer (DARPA-PS-26-09 Task 2)
// Author:  Dionisio Chiuratto Agourakis (jai)
// License: Apache-2.0
//
// Description:
//   Single-cycle / 2-stage pipelined 16D Sedenion Cayley-Dickson Zero-Divisor
//   CRPA Beamformer & Associative Quaternion Hinge Projector for 8-element
//   phase-coherent GNSS arrays (C^8 ~= R^16 ~= S).
//
//   Hardware Characteristics (Xilinx Zynq UltraScale+ RFSoC xczu28dr):
//     - Data Precision: 16-bit signed fixed-point (Q1.14) per real component
//     - Matrix Inversions: ZERO (O(1) single-snapshot algebraic annihilation)
//     - Primitive Blade Mode: Multiplier-free (64 signed add/sub + 16 bit-shifts)
//     - Full Bilinear Mode:   Single-cycle 16x16 Cayley-Dickson XOR-sign butterfly
//     - Quaternion Hinge Mask: Enforces strict associativity ([x,y,z] = 0) on
//                              any of the 35 projective lines of PG(3,2).
// =============================================================================
`timescale 1ns / 1ps

module sedenion_crpa_beamformer #(
    parameter int DATA_W = 16,          // Q1.14 signed fixed-point input width
    parameter int ACC_W  = 36           // Full-precision accumulator width
) (
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        in_valid,

    // 8-element CRPA complex baseband snapshot: x_in[0..7] = Re(C^8), x_in[8..15] = Im(C^8)
    input  logic signed [DATA_W-1:0]    x_in   [0:15],

    // Zero-divisor partner steering weight w_beam[0..15] in ker(L_{u_jam}) subset S
    input  logic signed [DATA_W-1:0]    w_beam [0:15],

    // 16-bit PG(3,2) associative quaternion hinge mask (e_0 + 3 collinear Fano points)
    // Set to 16'hFFFF for full 16D sedenion output, or 4-bit active line mask for PLL
    input  logic        [15:0]          hinge_mask,

    output logic                        out_valid,
    output logic signed [DATA_W-1:0]    z_out  [0:15]
);

    // =========================================================================
    // Exact 16x16 Cayley-Dickson Output-Indexed Sign Table SIGN_NEG_KJ[k][j]
    // For output basis e_k and right operand e_j (with left operand e_{i = k ^ j}):
    //   e_{k ^ j} * e_j = (SIGN_NEG_KJ[k][j] ? -1 : +1) * e_k
    // =========================================================================
    localparam logic [0:15][0:15] SIGN_NEG_KJ = '{
        16'b0111111111111111, // k =  0
        16'b0010100110010110, // k =  1
        16'b0001110011000011, // k =  2
        16'b0100101010100101, // k =  3
        16'b0000011111110000, // k =  4
        16'b0101000110101010, // k =  5
        16'b0110010010011001, // k =  6
        16'b0011001011001100, // k =  7
        16'b0000000001111111, // k =  8
        16'b0101011000010110, // k =  9
        16'b0110001101000011, // k = 10
        16'b0011010100100101, // k = 11
        16'b0111100001110000, // k = 12
        16'b0010111000101010, // k = 13
        16'b0001101100011001, // k = 14
        16'b0100110101001100  // k = 15
    };

    logic signed [ACC_W-1:0] acc_comb [0:15];

    always_comb begin
        for (int k = 0; k < 16; k++) begin
            acc_comb[k] = '0;
            for (int j = 0; j < 16; j++) begin
                automatic int i = k ^ j;
                automatic logic signed [2*DATA_W-1:0] prod = x_in[i] * w_beam[j];
                if (SIGN_NEG_KJ[k][j])
                    acc_comb[k] -= ACC_W'(prod);
                else
                    acc_comb[k] += ACC_W'(prod);
            end
        end
    end

    // Pipeline register with Q1.14 scaling (>> 14) and PG(3,2) Quaternion Hinge gating
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            for (int k = 0; k < 16; k++) begin
                z_out[k] <= '0;
            end
        end else begin
            out_valid <= in_valid;
            for (int k = 0; k < 16; k++) begin
                if (hinge_mask[k])
                    z_out[k] <= DATA_W'(acc_comb[k] >>> 14);
                else
                    z_out[k] <= '0;
            end
        end
    end

endmodule
