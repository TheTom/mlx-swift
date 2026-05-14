// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLX
@testable import MLXFast

/// Tests for `MLXFast.flashQuantizedSDPA` — the spec 041 phase 1.1 fused
/// flash SDPA that consumes affine-quantized K/V triples and dequantises
/// inline inside the tiled online-softmax loop.
///
/// The "matches dequant+SDPA" tests build a reference path by calling
/// `dequantized(...)` on the same K/V triples and feeding the result to
/// `MLXFast.scaledDotProductAttention`. Both paths run through softmax in
/// float32, so the only divergence is the inline dequant order in the fused
/// kernel — modest `atol/rtol` covers the bit-for-bit differences.
class MLXFastFlashQuantizedSDPATests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// Generate a Q + affine-quantized K/V triple at a given `bits` / `groupSize`.
    /// Returned shapes:
    ///   q:        [B, n_q_heads, T_q, headDim]
    ///   kPacked:  [B, n_kv_heads, T_kv, headDim / (32/bits)]
    ///   kScales:  [B, n_kv_heads, T_kv, headDim / groupSize]
    ///   kBiases:  [B, n_kv_heads, T_kv, headDim / groupSize]
    ///   v* triples have the same shape rule (V dim == K dim here).
    ///   kDequant / vDequant: the floating-point round-trip of the triples,
    ///     for the reference path. dtype matches `q.dtype`.
    private func makeQKV(
        B: Int, nQ: Int, nKV: Int, Tq: Int, Tkv: Int, headDim: Int,
        bits: Int, groupSize: Int, dtype: DType = .float32
    ) -> (
        q: MLXArray,
        kPacked: MLXArray, kScales: MLXArray, kBiases: MLXArray,
        vPacked: MLXArray, vScales: MLXArray, vBiases: MLXArray,
        kDequant: MLXArray, vDequant: MLXArray
    ) {
        MLXRandom.seed(0x041)
        let q = MLXRandom.normal([B, nQ, Tq, headDim]).asType(dtype)

        // Flatten the leading dims for quantize() — affine quantize operates
        // on the trailing axis, so we just stack rows.
        let kFloat = MLXRandom.normal([B, nKV, Tkv, headDim]).asType(dtype)
        let vFloat = MLXRandom.normal([B, nKV, Tkv, headDim]).asType(dtype)

        let (kPacked, kScales, kBiasOpt) = quantized(
            kFloat, groupSize: groupSize, bits: bits, mode: .affine)
        let (vPacked, vScales, vBiasOpt) = quantized(
            vFloat, groupSize: groupSize, bits: bits, mode: .affine)
        let kBiases = kBiasOpt!
        let vBiases = vBiasOpt!

        let kDequant = dequantized(
            kPacked, scales: kScales, biases: kBiases,
            groupSize: groupSize, bits: bits, mode: .affine, dtype: dtype)
        let vDequant = dequantized(
            vPacked, scales: vScales, biases: vBiases,
            groupSize: groupSize, bits: bits, mode: .affine, dtype: dtype)

        return (q, kPacked, kScales, kBiases, vPacked, vScales, vBiases, kDequant, vDequant)
    }

    private func runMatchesDequant(
        Tq: Int, Tkv: Int, bits: Int, groupSize: Int = 64,
        causal: Bool = false, headDim: Int = 128,
        rtol: Double = 1e-2, atol: Double = 1e-3
    ) {
        let B = 1
        let nQ = 4
        let nKV = 4
        let (q, kP, kS, kB, vP, vS, vB, kDQ, vDQ) = makeQKV(
            B: B, nQ: nQ, nKV: nKV, Tq: Tq, Tkv: Tkv, headDim: headDim,
            bits: bits, groupSize: groupSize)
        let scale = 1.0 / sqrt(Float(headDim))

        let outFlash = MLXFast.flashQuantizedSDPA(
            queries: q,
            kPacked: kP, kScales: kS, kBiases: kB,
            vPacked: vP, vScales: vS, vBiases: vB,
            scale: scale, bits: bits, groupSize: groupSize,
            causal: causal)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
            causal ? .causal : .none
        let outRef = MLXFast.scaledDotProductAttention(
            queries: q, keys: kDQ, values: vDQ, scale: scale, mask: maskMode)

        eval(outFlash, outRef)
        XCTAssertEqual(outFlash.shape, outRef.shape)
        XCTAssertTrue(
            allClose(outFlash, outRef, rtol: rtol, atol: atol).item(Bool.self),
            "flashQuantizedSDPA diverges from dequant+SDPA reference"
                + " (Tq=\(Tq) Tkv=\(Tkv) bits=\(bits) causal=\(causal))")
    }

    /// `flashQuantizedSDPA` output must agree with the unfused
    /// `dequantized → MLXFast.scaledDotProductAttention` path at the same
    /// (bits, groupSize) — they differ only in numeric order.
    func testFlashQuantizedSDPAMatchesDequantThenSDPA() {
        runMatchesDequant(Tq: 1, Tkv: 256, bits: 4)
        runMatchesDequant(Tq: 1, Tkv: 1024, bits: 4)
    }

    /// Same path, but with the causal mask mode (and a square Tq=Tkv shape so
    /// the diagonal is meaningful).
    func testFlashQuantizedSDPACausal() {
        runMatchesDequant(Tq: 8, Tkv: 8, bits: 4, causal: true)
    }

    /// 8-bit affine triples — the same equivalence should hold at a tighter
    /// quantization grid.
    func testFlashQuantizedSDPABits8() {
        runMatchesDequant(Tq: 1, Tkv: 256, bits: 8)
    }

    /// Build a `[1, 1, qL, kL]` additive mask that combines causal +
    /// sliding-window: keys are allowed iff `q_idx >= k_idx` AND
    /// `k_idx > q_idx - window`. Mirrors the predicate used inside the C++
    /// `scaled_dot_product_attention` sliding-window fallback (see fast.cpp,
    /// the `within_window` block).
    private func causalSlidingMask(
        qL: Int, kL: Int, window: Int, dtype: DType
    ) -> MLXArray {
        let offset = kL - qL
        let q = MLXArray(Int32(offset) ..< Int32(offset + qL))[0..., .newAxis]
        let k = MLXArray(Int32(0) ..< Int32(kL))[.newAxis]
        var ok = q .>= k
        ok = ok & (k .> (q - MLXArray(Int32(window))))
        let neg: Float = -Float.greatestFiniteMagnitude
        var mask = MLX.where(ok, MLXArray(Float(0)), MLXArray(neg))
        mask = mask.asType(dtype).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        return mask
    }

    /// Spec 041 phase 1.2: `windowSize > 0` plumbing on
    /// `flashQuantizedSDPA`. Reference path uses a materialized causal +
    /// sliding-window mask through the unquantized SDPA — same predicate the
    /// fused kernel applies internally.
    func testFlashQuantizedSDPAWithSlidingWindow() {
        let B = 1, nQ = 4, nKV = 4
        let Tq = 8, Tkv = 8, headDim = 128, window = 4
        let bits = 4, groupSize = 64
        let (q, kP, kS, kB, vP, vS, vB, kDQ, vDQ) = makeQKV(
            B: B, nQ: nQ, nKV: nKV, Tq: Tq, Tkv: Tkv, headDim: headDim,
            bits: bits, groupSize: groupSize)
        let scale = 1.0 / sqrt(Float(headDim))

        let outFlash = MLXFast.flashQuantizedSDPA(
            queries: q,
            kPacked: kP, kScales: kS, kBiases: kB,
            vPacked: vP, vScales: vS, vBiases: vB,
            scale: scale, bits: bits, groupSize: groupSize,
            causal: true, windowSize: window)

        let mask = causalSlidingMask(qL: Tq, kL: Tkv, window: window, dtype: q.dtype)
        let outRef = MLXFast.scaledDotProductAttention(
            queries: q, keys: kDQ, values: vDQ, scale: scale, mask: .array(mask))

        eval(outFlash, outRef)
        XCTAssertEqual(outFlash.shape, outRef.shape)
        XCTAssertTrue(
            allClose(outFlash, outRef, rtol: 1e-2, atol: 1e-3).item(Bool.self),
            "flashQuantizedSDPA(windowSize=\(window)) diverges from causal+sliding mask reference")
    }
}
