// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLX
@testable import MLXFast

/// Tests for `MLXFast.turboFlashSDPAv` — the spec 041 phase 1.1 follow-up
/// TurboQuant fused single-pass SDPA with optional sinks fold.
///
/// The kernel consumes the rotated codec-space tensors directly (caller is
/// responsible for the Hadamard rotation Π and the Π_v^T inverse), so the
/// inputs here are synthetic: the kernel itself is exercised, not the
/// codec. Output dtype is bfloat16, shape `[total_q, dim]`.
///
/// Layout (from `turbo_flash_sdpa.h` / `turbo_flash_sdpa.cpp`):
///   - queries:    [B * nQ, dim]            (float32 caller-side)
///   - k_packed:   [B * nKV, tokenCount, KEY_PACKED_WIDTH]   uint32
///   - k_norms:    [B * nKV, tokenCount]    float32
///   - k_codebook: [2^keyBits]              float32
///   - v_packed:   [B * nKV, tokenCount, VAL_PACKED_WIDTH]   uint32
///   - v_norms:    [B * nKV, tokenCount]    float32
///   - v_codebook: [2^valueBits]            float32
///   - sinks?:     [nQ]                     bfloat16 (the wrapper casts)
///   - out:        [B * nQ, dim]            bfloat16
class MLXFastTurboFlashSDPATests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// Bit-contiguous packed-width matching the kernel's
    /// `(Dim * Bits + 31) / 32` rule (see `turbo_flash_sdpa.h`).
    private func packedWidth(dim: Int, bits: Int) -> Int {
        (dim * bits + 31) / 32
    }

    /// Build a turbo4v2-style input pack. nQ/nKV/repeat satisfy
    /// `repeatCount = nQ / nKV`. Codebook is filled with `2^bits` evenly-
    /// spaced values in `[-1, 1]`. Norms are all ones. Packed indices are
    /// small random uint32 values clamped to the codebook range. The kernel
    /// will read these and produce a finite output — realism of the codec
    /// itself isn't being tested here.
    private struct Inputs {
        let queries: MLXArray
        let kPacked: MLXArray
        let kNorms: MLXArray
        let kCodebook: MLXArray
        let vPacked: MLXArray
        let vNorms: MLXArray
        let vCodebook: MLXArray
        let dim: Int
        let nQ: Int
        let nKV: Int
        let repeatCount: Int
    }

    private func makeInputs(
        B: Int, nQ: Int, nKV: Int, Tkv: Int, dim: Int,
        keyBits: Int, valueBits: Int
    ) -> Inputs {
        precondition(nQ % nKV == 0, "GQA: nQ must be a multiple of nKV")
        MLXRandom.seed(0xC0DE)
        let repeatCount = nQ / nKV
        let totalQ = B * nQ
        let kpw = packedWidth(dim: dim, bits: keyBits)
        let vpw = packedWidth(dim: dim, bits: valueBits)

        // Caller-side scaled queries (the codec rotation + scale would be
        // applied in real use; here the kernel just needs finite floats).
        let queries = MLXRandom.normal([totalQ, dim]).asType(.float32)

        // Packed indices: random uint32 patterns. The kernel masks off
        // `(1 << bits) - 1` per index so any 32-bit value is legal — only
        // the low `bits` bits matter per slot.
        let kPacked = MLXRandom.randInt(
            low: MLXArray(Int32(0)),
            high: MLXArray(Int32(1_000_000)),
            [B * nKV, Tkv, kpw]
        ).asType(.uint32)
        let vPacked = MLXRandom.randInt(
            low: MLXArray(Int32(0)),
            high: MLXArray(Int32(1_000_000)),
            [B * nKV, Tkv, vpw]
        ).asType(.uint32)

        let kNorms = MLX.ones([B * nKV, Tkv], dtype: .float32)
        let vNorms = MLX.ones([B * nKV, Tkv], dtype: .float32)

        // Codebooks: 2^bits evenly-spaced points in [-1, 1].
        let kLevels = 1 << keyBits
        let vLevels = 1 << valueBits
        let kCodebook = MLX.linspace(
            Float(-1.0), Float(1.0), count: kLevels).asType(.float32)
        let vCodebook = MLX.linspace(
            Float(-1.0), Float(1.0), count: vLevels).asType(.float32)

        return Inputs(
            queries: queries,
            kPacked: kPacked, kNorms: kNorms, kCodebook: kCodebook,
            vPacked: vPacked, vNorms: vNorms, vCodebook: vCodebook,
            dim: dim, nQ: nQ, nKV: nKV, repeatCount: repeatCount)
    }

    /// Output shape must match `[B * nQ, dim]` and contain no NaN/Inf.
    /// turbo4v2 = keyBits=4, valueBits=2 is the bench's headline scheme.
    func testTurboFlashSDPAvShapeAndFiniteness() {
        let B = 1, nQ = 4, nKV = 2, Tkv = 128, dim = 128
        let inputs = makeInputs(
            B: B, nQ: nQ, nKV: nKV, Tkv: Tkv, dim: dim,
            keyBits: 4, valueBits: 2)

        let out = MLXFast.turboFlashSDPAv(
            queries: inputs.queries,
            kPacked: inputs.kPacked, kNorms: inputs.kNorms, kCodebook: inputs.kCodebook,
            vPacked: inputs.vPacked, vNorms: inputs.vNorms, vCodebook: inputs.vCodebook,
            keyBits: 4, valueBits: 2, dim: dim, repeatCount: inputs.repeatCount)

        eval(out)
        XCTAssertEqual(out.shape, [B * nQ, dim])
        XCTAssertEqual(out.dtype, .bfloat16)
        let outF = out.asType(.float32)
        XCTAssertFalse(
            isNaN(outF).any().item(Bool.self),
            "turboFlashSDPAv output contains NaN")
        XCTAssertFalse(
            isInf(outF).any().item(Bool.self),
            "turboFlashSDPAv output contains Inf")
    }

    /// With sinks zero, the softmax denominator becomes
    /// `1 + Σ exp(scores)` instead of `Σ exp(scores)` — the output must
    /// shrink toward the per-head value-mean by the sink fold. So the
    /// sinks-on path must differ from the sinks-off path.
    func testTurboFlashSDPAvWithSinks() {
        let B = 1, nQ = 4, nKV = 2, Tkv = 128, dim = 128
        let inputs = makeInputs(
            B: B, nQ: nQ, nKV: nKV, Tkv: Tkv, dim: dim,
            keyBits: 4, valueBits: 2)

        let sinks = MLX.zeros([nQ], dtype: .float32)

        let outWithSinks = MLXFast.turboFlashSDPAv(
            queries: inputs.queries,
            kPacked: inputs.kPacked, kNorms: inputs.kNorms, kCodebook: inputs.kCodebook,
            vPacked: inputs.vPacked, vNorms: inputs.vNorms, vCodebook: inputs.vCodebook,
            keyBits: 4, valueBits: 2, dim: dim, repeatCount: inputs.repeatCount,
            sinks: sinks)
        let outNoSinks = MLXFast.turboFlashSDPAv(
            queries: inputs.queries,
            kPacked: inputs.kPacked, kNorms: inputs.kNorms, kCodebook: inputs.kCodebook,
            vPacked: inputs.vPacked, vNorms: inputs.vNorms, vCodebook: inputs.vCodebook,
            keyBits: 4, valueBits: 2, dim: dim, repeatCount: inputs.repeatCount)

        eval(outWithSinks, outNoSinks)
        XCTAssertEqual(outWithSinks.shape, [B * nQ, dim])
        let outF = outWithSinks.asType(.float32)
        XCTAssertFalse(isNaN(outF).any().item(Bool.self))
        XCTAssertFalse(isInf(outF).any().item(Bool.self))

        // sinks-on vs sinks-off must differ — softmax denominator changes.
        let diff =
            (outWithSinks.asType(.float32) - outNoSinks.asType(.float32)).abs()
        let maxAbs = diff.max().item(Float.self)
        XCTAssertGreaterThan(
            maxAbs, 1e-4,
            "sinks=0 should shift the softmax denominator, but turboFlashSDPAv"
                + " gave identical outputs (maxAbs=\(maxAbs))")
    }

    /// Spec 041 phase 1.2: sliding-window plumbing on `turboFlashSDPAv`.
    /// The kernel currently dispatches the same fused path with the window
    /// constraint applied in the causal mask. We verify only finiteness +
    /// that the window changes the output (i.e. the parameter actually
    /// reaches the kernel) — no scalar reference is available for the
    /// codec-space output.
    func testTurboFlashSDPAvWithSlidingWindow() {
        let B = 1, nQ = 4, nKV = 2, Tkv = 128, dim = 128
        let window = 64
        let inputs = makeInputs(
            B: B, nQ: nQ, nKV: nKV, Tkv: Tkv, dim: dim,
            keyBits: 4, valueBits: 2)

        let outWindow = MLXFast.turboFlashSDPAv(
            queries: inputs.queries,
            kPacked: inputs.kPacked, kNorms: inputs.kNorms, kCodebook: inputs.kCodebook,
            vPacked: inputs.vPacked, vNorms: inputs.vNorms, vCodebook: inputs.vCodebook,
            keyBits: 4, valueBits: 2, dim: dim, repeatCount: inputs.repeatCount,
            causal: true, windowSize: window)
        let outCausal = MLXFast.turboFlashSDPAv(
            queries: inputs.queries,
            kPacked: inputs.kPacked, kNorms: inputs.kNorms, kCodebook: inputs.kCodebook,
            vPacked: inputs.vPacked, vNorms: inputs.vNorms, vCodebook: inputs.vCodebook,
            keyBits: 4, valueBits: 2, dim: dim, repeatCount: inputs.repeatCount,
            causal: true)

        eval(outWindow, outCausal)
        XCTAssertEqual(outWindow.shape, [B * nQ, dim])
        let outF = outWindow.asType(.float32)
        XCTAssertFalse(isNaN(outF).any().item(Bool.self))
        XCTAssertFalse(isInf(outF).any().item(Bool.self))

        // window=64 excludes older-than-64 keys; pure causal includes all
        // past keys. With Tkv=128 some queries get a strict subset of K
        // positions, so outputs must differ.
        let diff =
            (outWindow.asType(.float32) - outCausal.asType(.float32)).abs()
        let maxAbs = diff.max().item(Float.self)
        XCTAssertGreaterThan(
            maxAbs, 1e-4,
            "windowSize=\(window) should narrow the causal mask, but"
                + " turboFlashSDPAv produced identical output (maxAbs=\(maxAbs))")
    }
}
