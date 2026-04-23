// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLX
@testable import MLXFast

/// Tests for the symbolic `.slidingWindow(size:)` SDPA mask mode.
///
/// Each test compares output against the materialized-mask reference built by
/// hand from the same (q_idx >= k_idx) && (k_idx > q_idx - window) predicate.
/// Both modes run through `MLXFast.scaledDotProductAttention`; the only
/// difference is where the mask is constructed (user-side tensor vs inside the
/// op's fallback path).
class SlidingWindowSDPATests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    /// Build `[1, 1, qL, kL]` additive mask with the same semantics as the
    /// symbolic `.slidingWindow(size:)` constraint.
    private func referenceMask(qL: Int, kL: Int, window: Int, dtype: DType) -> MLXArray {
        let offset = kL - qL
        let q = MLXArray(Int32(offset) ..< Int32(offset + qL))[0..., .newAxis]
        let k = MLXArray(Int32(0) ..< Int32(kL))[.newAxis]
        var ok = q .>= k
        ok = ok & (k .> (q - MLXArray(Int32(window))))
        // 0 where allowed, -inf where blocked.
        let neg: Float = -Float.greatestFiniteMagnitude
        var mask = MLX.where(ok, MLXArray(Float(0)), MLXArray(neg))
        mask = mask.asType(dtype).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        return mask
    }

    private func randomQKV(
        batch: Int, heads: Int, qL: Int, kL: Int, headDim: Int, dtype: DType
    ) -> (MLXArray, MLXArray, MLXArray) {
        let q = MLXRandom.normal([batch, heads, qL, headDim]).asType(dtype)
        let k = MLXRandom.normal([batch, heads, kL, headDim]).asType(dtype)
        let v = MLXRandom.normal([batch, heads, kL, headDim]).asType(dtype)
        return (q, k, v)
    }

    private func runCase(qL: Int, kL: Int, window: Int, dtype: DType = .float32) {
        let batch = 1
        let heads = 2
        let headDim = 64
        let (q, k, v) = randomQKV(
            batch: batch, heads: heads, qL: qL, kL: kL, headDim: headDim, dtype: dtype)
        let scale = 1.0 / sqrt(Float(headDim))

        let mat = referenceMask(qL: qL, kL: kL, window: window, dtype: dtype)

        let refOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(mat))
        let symOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale,
            mask: .slidingWindow(size: window))

        eval(refOut, symOut)
        // SDPA goes through softmax; use a permissive atol for bf16/f16.
        let atol: Double = (dtype == .float32) ? 1e-4 : 5e-3
        XCTAssertTrue(
            allClose(refOut, symOut, atol: atol).item(Bool.self),
            "qL=\(qL) kL=\(kL) window=\(window) dtype=\(dtype)")
    }

    func testDecodeStepShortCache() {
        // Cache still within the window: sliding constraint is a no-op
        // vs. pure causal. Output must match a causal-masked reference.
        runCase(qL: 1, kL: 64, window: 128)
    }

    func testDecodeStepAtWindowBoundary() {
        runCase(qL: 1, kL: 128, window: 128)
    }

    func testDecodeStepPastWindow() {
        // Most interesting: cache wider than window, so some older keys get
        // excluded. Divergence from pure causal is visible here.
        runCase(qL: 1, kL: 512, window: 128)
    }

    func testPrefillChunkAtWindowSize() {
        // Prefill in one chunk at the window boundary.
        runCase(qL: 128, kL: 128, window: 128)
    }

    func testPrefillChunkLargerThanWindow() {
        runCase(qL: 256, kL: 256, window: 128)
    }

    func testLargeWindow() {
        runCase(qL: 64, kL: 2048, window: 1024)
    }

    func testFloat16() {
        runCase(qL: 1, kL: 512, window: 128, dtype: .float16)
    }

    func testBfloat16() {
        runCase(qL: 1, kL: 512, window: 128, dtype: .bfloat16)
    }
}
