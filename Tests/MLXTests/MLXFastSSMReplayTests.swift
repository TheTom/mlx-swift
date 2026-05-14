// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLX
@testable import MLXFast

/// Tests for the spec 040 Mamba state-replay primitives —
/// `MLXFast.ssmStepRecord` and `MLXFast.ssmReplay`.
///
/// The kernels are only instantiated for `Dh ∈ {64, 128}`, `Ds ∈ {64, 128}`,
/// `H ∈ {16, 32, 48}`, `G ∈ {1, 2, 4, 8}`, dtype ∈ {float16, bfloat16}, so
/// these tests use the smallest valid combo: `H=16, G=1, Dh=64, Ds=64,
/// float16`.
///
/// Input/output layouts (from `ssm_replay.metal`):
///   x:        [B, T, H, Dh]
///   A_log:    [H]
///   B:        [B, T, G, Ds]
///   C:        [B, T, G, Ds]
///   D:        [H]
///   dt:       [B, T, H]
///   state:    [B, H, Dh, Ds]
///   ssm_step_record returns:
///     y         [B, T, H, Dh]
///     state_out [B, H, Dh, Ds]
///     dA_log    [B, T, H, Ds]
///     dBx_log   [B, T, H, Dh, Ds]
///   ssm_replay returns [B, H, Dh, Ds].
class MLXFastSSMReplayTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    private struct SSMShapes {
        let B: Int
        let T: Int
        let H: Int
        let G: Int
        let Dh: Int
        let Ds: Int
        let dtype: DType
    }

    private func makeInputs(_ s: SSMShapes) -> (
        x: MLXArray, ALog: MLXArray, Bm: MLXArray, Cm: MLXArray,
        Dm: MLXArray, dt: MLXArray, state: MLXArray
    ) {
        MLXRandom.seed(0x040)
        let x = MLXRandom.normal([s.B, s.T, s.H, s.Dh]).asType(s.dtype)
        // A_log is typically log(-A) of a negative state matrix — use a small
        // negative tensor so `dA = exp(A · dt)` stays bounded.
        let ALog = (MLXRandom.uniform(0.0 ..< 1.0, [s.H]) * Float(-2.0)).asType(s.dtype)
        let Bm = MLXRandom.normal([s.B, s.T, s.G, s.Ds]).asType(s.dtype)
        let Cm = MLXRandom.normal([s.B, s.T, s.G, s.Ds]).asType(s.dtype)
        let Dm = MLXRandom.normal([s.H]).asType(s.dtype)
        // `dt` is typically softplus-positive; uniform[0, 0.1] keeps the
        // recurrence numerically tame across T steps.
        let dt = (MLXRandom.uniform(0.0 ..< 1.0, [s.B, s.T, s.H]) * Float(0.1))
            .asType(s.dtype)
        let state = MLXRandom.normal([s.B, s.H, s.Dh, s.Ds]).asType(s.dtype)
            * Float(0.1)
        return (x, ALog, Bm, Cm, Dm, dt, state)
    }

    /// `ssmStepRecord` must return exactly four arrays with the shapes
    /// documented in `fast.h` (`y`, `state_out`, `dA_log`, `dBx_log`).
    func testSSMStepRecordReturnsFourArrays() {
        let s = SSMShapes(B: 1, T: 4, H: 16, G: 1, Dh: 64, Ds: 64, dtype: .float16)
        let (x, ALog, Bm, Cm, Dm, dt, state) = makeInputs(s)
        let outs = MLXFast.ssmStepRecord(
            x: x, ALog: ALog, B: Bm, C: Cm, D: Dm, dt: dt, state: state)
        XCTAssertEqual(outs.count, 4, "ssmStepRecord must return [y, state_out, dA_log, dBx_log]")
        eval(outs)
        XCTAssertEqual(outs[0].shape, [s.B, s.T, s.H, s.Dh], "y shape")
        XCTAssertEqual(outs[1].shape, [s.B, s.H, s.Dh, s.Ds], "state_out shape")
        XCTAssertEqual(outs[2].shape, [s.B, s.T, s.H, s.Ds], "dA_log shape")
        XCTAssertEqual(outs[3].shape, [s.B, s.T, s.H, s.Dh, s.Ds], "dBx_log shape")
    }

    /// Replaying the entire log onto the initial state must reproduce the
    /// `state_out` that `ssmStepRecord` returned — the log is, by
    /// construction, the full recurrence applied to that snapshot.
    func testSSMReplayRoundTrip() {
        let s = SSMShapes(B: 1, T: 4, H: 16, G: 1, Dh: 64, Ds: 64, dtype: .float16)
        let (x, ALog, Bm, Cm, Dm, dt, state) = makeInputs(s)
        let outs = MLXFast.ssmStepRecord(
            x: x, ALog: ALog, B: Bm, C: Cm, D: Dm, dt: dt, state: state)
        eval(outs)
        let stateOut = outs[1]
        let dALog = outs[2]
        let dBxLog = outs[3]

        let replayed = MLXFast.ssmReplay(
            stateSnapshot: state, dALog: dALog, dBxLog: dBxLog,
            acceptedPrefix: s.T)
        eval(replayed)
        XCTAssertEqual(replayed.shape, stateOut.shape)
        // Record's `local_state` runs in fp32 across T steps, but writes
        // dA/dBx out as the kernel dtype (fp16). Replay reads those back in
        // fp32 — so the two paths agree up to fp16 storage rounding, not
        // bit-for-bit. Tolerance scaled accordingly.
        let diff = (replayed.asType(.float32) - stateOut.asType(.float32)).abs()
        let maxAbs = diff.max().item(Float.self)
        XCTAssertLessThan(
            maxAbs, 1e-2,
            "Replay over the full T-step log should match state_out (maxAbs=\(maxAbs))")
    }

    /// A partial-accept replay must land on a state that is neither the
    /// snapshot (k=0 case) nor the full final state (k=T case) — partial
    /// rollback is the actual reason this primitive exists.
    func testSSMReplayPartialAccept() {
        let s = SSMShapes(B: 1, T: 8, H: 16, G: 1, Dh: 64, Ds: 64, dtype: .float16)
        let (x, ALog, Bm, Cm, Dm, dt, state) = makeInputs(s)
        let outs = MLXFast.ssmStepRecord(
            x: x, ALog: ALog, B: Bm, C: Cm, D: Dm, dt: dt, state: state)
        eval(outs)
        let stateOut = outs[1]
        let dALog = outs[2]
        let dBxLog = outs[3]

        let replayed = MLXFast.ssmReplay(
            stateSnapshot: state, dALog: dALog, dBxLog: dBxLog,
            acceptedPrefix: 4)
        eval(replayed)
        XCTAssertEqual(replayed.shape, state.shape)
        // Partial replay must move the state away from the snapshot…
        XCTAssertFalse(
            allClose(replayed, state, rtol: 1e-3, atol: 1e-4).item(Bool.self),
            "Partial-accept replay (k=4) should differ from the snapshot")
        // …and must also not match the full-T final state.
        XCTAssertFalse(
            allClose(replayed, stateOut, rtol: 1e-3, atol: 1e-4).item(Bool.self),
            "Partial-accept replay (k=4) should differ from state_out at k=T")
    }
}
