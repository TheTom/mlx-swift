// Copyright © 2026 Eric Kryski. GatedDeltaNet tape-replay rollback kernel.
//
// Companion kernel to `gated_delta.metal`. While the forward kernel
// (`gated_delta_step`) advances the recurrent state token-by-token under
// `S_t = g_t * S_{t-1} + k_t * (v_t - k_t^T * S_{t-1})^T * beta_t`, this
// kernel fuses a *re-fold* of an accepted prefix of an innovation tape
// into a single dispatch.
//
// Use case (spec 020 phase 2): a speculative decoder runs a verify
// forward over T draft tokens, accepts the first k ≤ T, and needs the
// recurrent state to be `s_{t_0 + k}` for the next iteration.
// `SSMStateCache.beginRecord()` snapshots `s_{t_0}` and tapes the
// per-step `(delta_t, k_t, g_t)` triples during the verify forward.
// `SSMStateCache.rollback(acceptedPrefix: k)` then restores the
// snapshot and re-folds the first k entries through the recurrence.
//
// dflash-mlx upstream: `bstnxbt/dflash-mlx@main`'s
// `_make_tape_replay_kernel(...)` is the reference. This port adopts:
//   - **Masked-timestep correctness fix** (commit `3217e15`): save
//     `old_state` before each step and restore via `metal::select`
//     on masked positions, so masked timesteps don't silently corrupt
//     state. Required from day one.
//   - **Branchless pattern** (commit `c9f992e`): `mask_gate =
//     float(mask)` multiply instead of `if(mask)` guards. Better SIMD
//     occupancy when timestep masks are non-uniform within a SIMD
//     group (the verify-window mask shape on partial-accept rounds).
//
// The recurrence inside the loop body matches `gated_delta_step` in
// `gated_delta.metal` — same fp32-accumulator pattern, same SIMD
// shuffle reduction across Dk lanes — but reads `delta_t` from the
// tape rather than computing `(v_t - kv_mem) * beta_t` afresh.

#include <metal_common>
#include <metal_simdgroup>
#include <metal_math>

#include "utils.h"

using namespace metal;

// Function constant — single source for masked vs non-masked variants.
// Mirrors the pattern in gated_delta.metal so the bench harness can
// flip masking on/off via the same dispatch surface.
constant bool tape_replay_has_mask [[function_constant(20)]];

// ============================================================================
// Tape-replay kernel
// ============================================================================
//
// Inputs:
//   - delta_tape:   [B, T_tape, Hv, Dv]      per-step innovation
//                                            (`(v_t - kv_mem_t) * beta_t`)
//   - k_tape:       [B, T_tape, Hk, Dk]      per-step keys
//   - g_tape:       [B, T_tape, Hv]          per-step decay scalar
//   - state_in:     [B, Hv, Dv, Dk]          pre-record snapshot
//   - mask:         [B, T_tape]              per-timestep accept gate
//                                            (true = re-fold, false =
//                                             keep `old_state`). When
//                                            `tape_replay_has_mask =
//                                             false`, this buffer is
//                                            ignored and ALL `accepted`
//                                            steps re-fold.
//   - accepted:     int                      number of tape entries to
//                                            re-fold (0 ≤ accepted ≤
//                                            T_tape). Steps t ≥
//                                            `accepted` are skipped.
//
// Output:
//   - state_out:    [B, Hv, Dv, Dk]          post-rollback recurrent
//                                            state
//
// Threadgroup geometry mirrors `gated_delta_step`: thread_pos.z = (b,hv)
// flattened, thread_pos.y = dv lane, tg_pos.x = dk lane (1 per
// SIMD-group thread under `n_per_t = Dk / 32`).
template <typename T, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void tape_replay(
    const device T*    delta_tape [[buffer(0)]],
    const device T*    k_tape     [[buffer(1)]],
    const device T*    g_tape     [[buffer(2)]],
    const device T*    state_in   [[buffer(3)]],
    const device bool* mask       [[buffer(4)]],
    device T*          state_out  [[buffer(5)]],
    constant int&      T_tape     [[buffer(6)]],
    constant int&      accepted   [[buffer(7)]],
    uint3              thread_pos [[thread_position_in_grid]],
    uint3              tg_pos     [[thread_position_in_threadgroup]],
    uint               simd_lane  [[thread_index_in_simdgroup]]) {

  constexpr int n_per_t = Dk / 32;

  auto n      = thread_pos.z;
  auto b_idx  = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto dk_idx = tg_pos.x;
  auto dv_idx = thread_pos.y;

  // Per-step input pointers — advanced by one timestep stride at the
  // bottom of the loop, like `gated_delta_step`.
  auto delta_ = delta_tape + b_idx * T_tape * Hv * Dv + hv_idx * Dv;
  auto k_     = k_tape     + b_idx * T_tape * Hk * Dk + hk_idx * Dk;
  auto g_     = g_tape     + b_idx * T_tape * Hv;

  // State buffers — read once into a per-thread fp32 register file,
  // process all accepted timesteps, write back at the end. Same pattern
  // as `gated_delta_step` so register pressure / occupancy remain
  // identical to the forward path.
  auto i_state = state_in  + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
  }

  // Re-fold loop. We iterate up to `T_tape` so the kernel surface is
  // shape-stable, and gate work via `do_step = (t < accepted) &&
  // (mask[t] || !has_mask)`. Steps with `do_step = false` save
  // `old_state` first and `metal::select(new, old, do_step)` it back,
  // so masked or post-accepted timesteps are exact no-ops on the
  // running state — the masked-timestep correctness fix.
  for (int t = 0; t < T_tape; ++t) {
    bool within_accepted = (t < accepted);
    bool mask_passes     = tape_replay_has_mask
                              ? mask[b_idx * T_tape + t]
                              : true;
    bool do_step         = within_accepted && mask_passes;

    // Save old state for the conditional restore. Stack-allocated;
    // the compiler holds these in registers alongside `state[i]`.
    float old_state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      old_state[i] = state[i];
    }

    // GDN recurrence step — same body as `gated_delta_step` minus the
    // delta computation (delta is already on the tape):
    //   state[i] *= g
    //   state[i] += k[s_idx] * delta_at_dv
    //
    // Branchless pattern: we always do the math, then `metal::select`
    // back to old_state on masked timesteps. This avoids divergent
    // execution within a SIMD group when the mask is non-uniform.
    float g_val   = static_cast<float>(g_[hv_idx]);
    float d_val   = static_cast<float>(delta_[dv_idx]);

    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      float new_val = state[i] * g_val
                    + static_cast<float>(k_[s_idx]) * d_val;
      state[i] = metal::select(old_state[i], new_val, do_step);
    }

    // Advance per-step pointers regardless of `do_step` so the index
    // math stays in lockstep with the tape layout.
    delta_ += Hv * Dv;
    k_     += Hk * Dk;
    g_     += Hv;
  }

  // Write final state back to global memory.
  for (int i = 0; i < n_per_t; ++i) {
    o_state[n_per_t * dk_idx + i] = static_cast<T>(state[i]);
  }
}

// ============================================================================
// Instantiation — same (Dk, Dv, Hk, Hv) cells as `gated_delta_step` so
// every model that already uses the forward kernel can also use the
// replay kernel.
// ============================================================================
#define instantiate_gdn_replay(type, tname, dk, dv, hk, hv) \
  template [[host_name("tape_replay_" #tname "_" #dk "_" #dv "_" #hk "_" #hv)]] \
  [[kernel]] void tape_replay<type, dk, dv, hk, hv>( \
    const device type*, const device type*, const device type*, \
    const device type*, const device bool*, device type*, \
    constant int&, constant int&, uint3, uint3, uint);

// Qwen3.5-A3B: Dk=192, Dv=128, Hk=4, Hv=4
instantiate_gdn_replay(half,        float16,  192, 128, 4, 4)
instantiate_gdn_replay(bfloat16_t,  bfloat16, 192, 128, 4, 4)

// Qwen3.5 larger variants
instantiate_gdn_replay(half,        float16,  128, 128, 8, 8)
instantiate_gdn_replay(bfloat16_t,  bfloat16, 128, 128, 8, 8)
instantiate_gdn_replay(half,        float16,  64,  64,  8, 8)
instantiate_gdn_replay(bfloat16_t,  bfloat16, 64,  64,  8, 8)

// Qwen3.5-35B: Dk=128, Dv=128, numHeads=16, numKVHeads=2 → Hk=16 Hv=32
instantiate_gdn_replay(half,        float16,  128, 128, 16, 32)
instantiate_gdn_replay(bfloat16_t,  bfloat16, 128, 128, 16, 32)

// Qwen3.5 dense models: Dk=128, Dv=128, Hk=16, Hv=16 (0.8B–9B)
instantiate_gdn_replay(half,        float16,  128, 128, 16, 16)
instantiate_gdn_replay(bfloat16_t,  bfloat16, 128, 128, 16, 16)

// Qwen3.5 dense 27B: Hv=48
instantiate_gdn_replay(half,        float16,  128, 128, 16, 48)
instantiate_gdn_replay(bfloat16_t,  bfloat16, 128, 128, 16, 48)
