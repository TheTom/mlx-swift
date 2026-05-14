// Copyright © 2024 Apple Inc.

import Cmlx

public enum MLXFast {

    /// Optimized implementation of `NN.RoPE`.
    ///
    /// Used like this:
    ///
    /// ```swift
    /// let x: MLXArray
    /// let dimensions: Int
    /// let traditional: Bool
    /// let base: Float
    /// let scale: Float
    /// let offset: Int
    ///
    /// let shape = x.shape
    /// var x = x.reshaped(-1, x.dim(-2), x.dim(-1))
    /// x = MLXFast.RoPE(x, dimensions: dimensions, traditional: traditional, base: base, scale: scale, offset: offset)
    /// return x.reshaped(shape)
    /// ```
    ///
    /// > Note: `MLXNN.RoPE` uses this implementation internally.
    public static func RoPE(
        _ array: MLXArray, dimensions: Int, traditional: Bool, base: Float?, scale: Float,
        offset: Int,
        freqs: MLXArray? = nil, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        let base = mlx_optional_float(value: base ?? 0, has_value: base != nil)
        mlx_fast_rope(
            &result,
            array.ctx, Int32(dimensions), traditional, base, scale, Int32(offset),
            (freqs ?? .mlxNone).ctx, stream.ctx)
        return MLXArray(result)
    }

    /// Optimized implementation of `NN.RoPE` with array offset for batched inference.
    ///
    /// This overload accepts an array offset, allowing different position offsets for each
    /// sequence in a batch. The offset can be a scalar array or a vector with length
    /// matching the batch size.
    ///
    /// - Parameters:
    ///   - array: input array
    ///   - dimensions: The feature dimensions to be rotated. If the input feature is larger
    ///     than dims then the rest is left unchanged.
    ///   - traditional: If `true` choose the traditional implementation which is slightly less efficient.
    ///   - base: The base used to compute angular frequency for each dimension in the positional encodings.
    ///   - scale: The scale used to scale the positions.
    ///   - offset: The position offset as an array. Can be a scalar or a vector of offsets for each batch element.
    ///   - freqs: Optional frequencies to use with RoPE.
    ///   - stream: stream or device to evaluate on
    /// - Returns: The input with rotary positional encoding applied.
    public static func RoPE(
        _ array: MLXArray,
        dimensions: Int,
        traditional: Bool,
        base: Float?,
        scale: Float,
        offset: MLXArray,
        freqs: MLXArray? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        let base = mlx_optional_float(value: base ?? 0, has_value: base != nil)
        let offset = offset
        mlx_fast_rope_dynamic(
            &result,
            array.ctx, Int32(dimensions), traditional, base, scale, offset.ctx,
            (freqs ?? .mlxNone).ctx, stream.ctx)
        return MLXArray(result)
    }

    /// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
    ///
    /// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
    ///
    /// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
    ///
    /// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
    ///
    /// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
    ///
    /// Specifically this implements:
    ///
    /// ```swift
    /// var scores = (queries * self.scale).matmul(keys.transposed(0, 1, 3, 2))
    /// if let mask {
    ///     scores = scores + mask
    /// }
    ///
    /// scores = softMax(scores.asType(.float32), axis: -1).asType(scores.dtype)
    ///
    /// return matmul(scores, values).transposed(0, 2, 1, 3)
    /// ```
    ///
    /// In the following the dimensions are given by:
    ///
    /// * `B`: The batch size.
    /// * `N_q`: The number of query heads.
    /// * `N_kv`: The number of key and value heads.
    /// * `T_q`: The number of queries per example.
    /// * `T_kv`: The number of keys and values per example.
    /// * `D`: The per-head dimension.
    ///
    /// - Parameters:
    ///   - queries: queries with shape `[B, N_q, T_q, D]`
    ///   - keys: keys with shape `[B, N_kv, T_kv, D]`
    ///   - values: values with shape `[B, N_kv, T_kv, D]`
    ///   - scale: scale for queries, typically `1 / sqrt(q.dim(-1))`
    ///   - mask: mask array
    ///   - sinks: optional array of attention sinks
    ///   - memoryEfficientThreshold: unused
    ///   - stream: stream to evaluate on
    public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?,
        sinks: MLXArray? = nil,
        memoryEfficientThreshold: Int? = nil, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()

        mlx_fast_scaled_dot_product_attention(
            &result,
            queries.ctx, keys.ctx, values.ctx, scale,
            "", mask?.ctx ?? MLXArray.mlxNone.ctx,
            (sinks ?? .mlxNone).ctx,
            stream.ctx)
        return MLXArray(result)
    }

    public enum ScaledDotProductAttentionMaskMode {
        case none
        case array(MLXArray)

        @available(*, deprecated, message: "Use .array instead")
        case arrays([MLXArray])
        case causal

        /// Causal mask restricted to a diagonal band of `size` keys per query.
        /// Equivalent to `.causal` with an additional constraint
        /// `k_idx > q_idx - size`. Used by Gemma-family "sliding_attention"
        /// layers. Currently routes through the composed fallback path on
        /// every backend — the causal constraint is synthesized on GPU via
        /// arange + compare, no user-side mask materialization.
        case slidingWindow(size: Int)

        public var mask: MLXArray? {
            switch self {
            case .none: return nil
            case .array(let array): return array
            case .arrays(let arrays):
                precondition(arrays.count <= 1, "Only a single array is allowed")
                return arrays.first
            case .causal: return nil
            case .slidingWindow: return nil
            }
        }

        public var mode: String {
            switch self {
            case .none: ""
            case .array: ""
            case .arrays: ""
            case .causal: "causal"
            case .slidingWindow: "causal"
            }
        }

        /// Positive window size for `.slidingWindow`; `-1` otherwise.
        public var windowSize: Int32 {
            switch self {
            case .slidingWindow(let size):
                precondition(size > 0, "slidingWindow size must be positive")
                return Int32(size)
            default: return -1
            }
        }
    }

    /// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
    ///
    /// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
    ///
    /// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
    ///
    /// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
    ///
    /// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
    ///
    /// Specifically this implements:
    ///
    /// ```swift
    /// var scores = (queries * self.scale).matmul(keys.transposed(0, 1, 3, 2))
    /// if let mask {
    ///     scores = scores + mask
    /// }
    ///
    /// scores = softMax(scores.asType(.float32), axis: -1).asType(scores.dtype)
    ///
    /// return matmul(scores, values).transposed(0, 2, 1, 3)
    /// ```
    ///
    /// In the following the dimensions are given by:
    ///
    /// * `B`: The batch size.
    /// * `N_q`: The number of query heads.
    /// * `N_kv`: The number of key and value heads.
    /// * `T_q`: The number of queries per example.
    /// * `T_kv`: The number of keys and values per example.
    /// * `D`: The per-head dimension.
    ///
    /// - Parameters:
    ///   - queries: queries with shape `[B, N_q, T_q, D]`
    ///   - keys: keys with shape `[B, N_kv, T_kv, D]`
    ///   - values: values with shape `[B, N_kv, T_kv, D]`
    ///   - scale: scale for queries, typically `1 / sqrt(q.dim(-1))`
    ///   - mask: a ``ScaledDotProductAttentionMaskMode``
    ///   - sinks: optional array of attention sinks
    ///   - stream: stream to evaluate on
    public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: ScaledDotProductAttentionMaskMode,
        sinks: MLXArray? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        let window = mask.windowSize
        if window > 0 {
            mlx_fast_scaled_dot_product_attention_sliding(
                &result,
                queries.ctx, keys.ctx, values.ctx, scale,
                mask.mode, mask.mask?.ctx ?? MLXArray.mlxNone.ctx,
                (sinks ?? .mlxNone).ctx,
                Int32(window),
                stream.ctx)
        } else {
            mlx_fast_scaled_dot_product_attention(
                &result,
                queries.ctx, keys.ctx, values.ctx, scale,
                mask.mode, mask.mask?.ctx ?? MLXArray.mlxNone.ctx,
                (sinks ?? .mlxNone).ctx,
                stream.ctx)
        }
        return MLXArray(result)
    }

    /// Root Mean Square normalization (RMS norm).
    ///
    /// The normalization is with respect to the last axis of the input `x`.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
    ///     with the same size as the last axis of `x`.
    ///   - eps: A small additive constant for numerical stability
    ///   - stream: stream or device to evaluate on
    public static func rmsNorm(
        _ x: MLXArray, weight: MLXArray, eps: Float, stream: StreamOrDevice = .default
    )
        -> MLXArray
    {
        var result = mlx_array_new()
        mlx_fast_rms_norm(&result, x.ctx, weight.ctx, eps, stream.ctx)
        return MLXArray(result)
    }

    /// Fused RMSNorm + Residual Add operation.
    ///
    /// Computes `residual + rmsNorm(x, weight, eps)` in a single Metal dispatch.
    /// Saves one kernel launch per call vs separate rmsNorm + add.
    ///
    /// - Parameters:
    ///   - x: input array to normalize
    ///   - residual: skip connection array (same shape as x)
    ///   - weight: RMSNorm weight (1D, same size as last axis of x)
    ///   - eps: normalization epsilon
    ///   - stream: stream or device to evaluate on
    public static func rmsNormResidual(
        _ x: MLXArray,
        residual: MLXArray,
        weight: MLXArray,
        eps: Float,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_rms_norm_residual(
            &result, x.ctx, residual.ctx, weight.ctx, eps, stream.ctx)
        return MLXArray(result)
    }

    /// Activation variants supported by the fused dense gate+activation kernel.
    public enum DenseGateActivation: Int32, Sendable {
        case silu = 0
        case geluApprox = 1
        /// Clipped SwiGLU (GPT-OSS). Both halves clamped to [-7, 7];
        /// `out = gate·sigmoid(1.702·gate)·(up + 1)`.
        case clippedSwiglu = 2
    }

    /// Fused dense gate+activation (inline SwiGLU/GeGLU) kernel.
    ///
    /// Takes a pre-concatenated `gateUp` tensor of shape `[..., 2 * hiddenDims]`
    /// and computes `activation(gate) * up` of shape `[..., hiddenDims]` in a
    /// single Metal dispatch. Replaces the
    /// Split + activation + Multiply chain (4 dispatches) commonly found in
    /// gated-MLP forward passes.
    ///
    /// - Parameters:
    ///   - gateUp: input array `[..., 2 * hiddenDims]`
    ///   - hiddenDims: post-split feature dim
    ///   - activation: `.silu` (Qwen / GLU) or `.geluApprox` (Gemma GEGLU)
    ///   - stream: stream or device to evaluate on
    public static func fusedGateActivation(
        _ gateUp: MLXArray,
        hiddenDims: Int,
        activation: DenseGateActivation,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_fused_gate_activation(
            &result,
            gateUp.ctx,
            Int32(hiddenDims),
            activation.rawValue,
            stream.ctx)
        return MLXArray(result)
    }

    /// Fused RMSNorm + RoPE operation.
    ///
    /// Combines RMS normalization and rotary position embedding in a single dispatch.
    /// Input is in `[B, L, nHeads, headDim]` layout (pre-transpose). The operation applies
    /// RMSNorm weight scaling then RoPE rotation for each (batch, position, head) row.
    ///
    /// - Parameters:
    ///   - x: input array `[B, L, nHeads, headDim]`
    ///   - weight: RMSNorm weight `[headDim]`
    ///   - invFreqs: inverse frequencies `[headDim/2]`. Use 0 for unrotated dimensions.
    ///   - eps: normalization epsilon
    ///   - offset: RoPE position offset (cache.offset)
    ///   - nHeads: number of attention heads
    ///   - seqLen: sequence length (L dimension)
    ///   - stream: stream or device to evaluate on
    public static func rmsNormRoPE(
        _ x: MLXArray,
        weight: MLXArray,
        invFreqs: MLXArray,
        eps: Float,
        offset: Int,
        nHeads: Int,
        seqLen: Int,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_rms_norm_rope(
            &result, x.ctx, weight.ctx, invFreqs.ctx,
            eps, Int32(offset), Int32(nHeads), Int32(seqLen),
            stream.ctx)
        return MLXArray(result)
    }

    /// Fused RMSNorm + Quantized GEMV (matrix-vector multiply) for decode inference.
    ///
    /// Combines RMS normalization with 4-bit quantized matrix-vector multiply in a
    /// single kernel dispatch. Eliminates the global memory round-trip between
    /// separate RMSNorm and quantized matmul operations.
    ///
    /// - Parameters:
    ///   - x: input vector `[..., K]`
    ///   - normWeight: RMSNorm weight `[K]`
    ///   - w: quantized weights `[N, K_packed]` (4-bit packed)
    ///   - scales: per-group scales `[N, K/groupSize]`
    ///   - biases: per-group biases `[N, K/groupSize]`
    ///   - eps: normalization epsilon
    ///   - groupSize: quantization group size (typically 64)
    ///   - stream: stream or device to evaluate on
    public static func rmsNormQuantizedGEMV(
        _ x: MLXArray,
        normWeight: MLXArray,
        w: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        eps: Float,
        groupSize: Int = 64,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_rms_norm_qgemv(
            &result, x.ctx, normWeight.ctx, w.ctx, scales.ctx, biases.ctx,
            eps, Int32(groupSize), stream.ctx)
        return MLXArray(result)
    }

    /// Batched QKV quantized GEMV: 3 projections in a single Metal dispatch.
    ///
    /// Loads input x to shared memory once, then computes Q, K, V GEMVs sequentially.
    /// Returns concatenated [N_q + N_k + N_v] output. Caller splits.
    /// Saves 2 Metal dispatches per layer vs 3 separate `quantizedMM` calls.
    /// Decode-only (T=1).
    public static func batchedQKVQuantizedGEMV(
        _ x: MLXArray,
        wQ: MLXArray, scalesQ: MLXArray, biasesQ: MLXArray,
        wK: MLXArray, scalesK: MLXArray, biasesK: MLXArray,
        wV: MLXArray, scalesV: MLXArray, biasesV: MLXArray,
        groupSize: Int = 64,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_batched_qkv_qgemv(
            &result,
            x.ctx,
            wQ.ctx, scalesQ.ctx, biasesQ.ctx,
            wK.ctx, scalesK.ctx, biasesK.ctx,
            wV.ctx, scalesV.ctx, biasesV.ctx,
            Int32(groupSize), stream.ctx)
        return MLXArray(result)
    }

    /// Warp MoE Gate+Up: fused gate+up projection with activation for decode.
    ///
    /// Each SIMD group computes one activated neuron for one expert.
    /// Replaces gatherQuantizedMM(gate_up) + split + activation.
    ///
    /// - Parameters:
    ///   - x: input vector [inputDims] (flattened)
    ///   - w: gate_up weights [numExperts, 2*hiddenDims, inputDims_packed]
    ///   - indices: expert indices [topK]
    ///   - activationType: 0=silu, 1=gelu_approx, 2=swiglu
    /// - Returns: activated [topK, hiddenDims]
    public static func warpMoeGateUp(
        _ x: MLXArray,
        w: MLXArray, scales: MLXArray, biases: MLXArray,
        indices: MLXArray,
        groupSize: Int = 64,
        hiddenDims: Int,
        activationType: Int = 0,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_warp_moe_gate_up(
            &result, x.ctx,
            w.ctx, scales.ctx, biases.ctx,
            indices.ctx,
            Int32(groupSize), Int32(hiddenDims), Int32(activationType),
            stream.ctx)
        return MLXArray(result)
    }

    /// Warp MoE Down: fused down projection with routing weight folding.
    ///
    /// Each SIMD group computes one final output neuron, looping over all
    /// topK experts and folding routing scores into the accumulator.
    /// Replaces gatherQuantizedMM(down) + weighted sum.
    ///
    /// - Parameters:
    ///   - activated: per-expert intermediates [topK, hiddenDims]
    ///   - w: down projection weights [numExperts, outputDims, hiddenDims_packed]
    ///   - indices: expert indices [topK]
    ///   - scores: routing weights [topK]
    /// - Returns: final MoE output [outputDims]
    public static func warpMoeDown(
        _ activated: MLXArray,
        w: MLXArray, scales: MLXArray, biases: MLXArray,
        indices: MLXArray, scores: MLXArray,
        groupSize: Int = 64,
        hiddenDims: Int,
        outDims: Int,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_warp_moe_down(
            &result, activated.ctx,
            w.ctx, scales.ctx, biases.ctx,
            indices.ctx, scores.ctx,
            Int32(groupSize), Int32(hiddenDims), Int32(outDims),
            stream.ctx)
        return MLXArray(result)
    }

    /// Layer normalization.
    ///
    /// The normalization is with respect to the last axis of the input `x`.
    ///
    /// - Parameters:
    ///   - x: input array
    ///   - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
    ///     with the same size as the last axis of `x`.  If not given no scaling will occur.
    ///   - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
    ///     with the same size as the last axis of `x`.  It not given no offset will occur.
    ///   - eps: A small additive constant for numerical stability
    ///   - stream: stream or device to evaluate on
    public static func layerNorm(
        _ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil, eps: Float,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_layer_norm(
            &result, x.ctx, (weight ?? .mlxNone).ctx, (bias ?? .mlxNone).ctx, eps, stream.ctx)
        return MLXArray(result)
    }

}

/// Optimized implementation of `NN.RoPE`.
///
/// Used like this:
///
/// ```swift
/// let x: MLXArray
/// let dimensions: Int
/// let traditional: Bool
/// let base: Float
/// let scale: Float
/// let offset: Int
///
/// let shape = x.shape
/// var x = x.reshaped(-1, x.dim(-2), x.dim(-1))
/// x = MLXFast.RoPE(x, dimensions: dimensions, traditional: traditional, base: base, scale: scale, offset: offset)
/// return x.reshaped(shape)
/// ```
///
/// > Note: `MLXNN.RoPE` uses this implementation internally.
public func RoPE(
    _ array: MLXArray, dimensions: Int, traditional: Bool, base: Float?, scale: Float, offset: Int,
    freqs: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXFast.RoPE(
        array, dimensions: dimensions, traditional: traditional, base: base, scale: scale,
        offset: offset, freqs: freqs, stream: stream)
}

/// Optimized implementation of `NN.RoPE` with array offset for batched inference.
///
/// > Note: `MLXNN.RoPE` uses this implementation internally.
public func RoPE(
    _ array: MLXArray, dimensions: Int, traditional: Bool, base: Float?, scale: Float,
    offset: MLXArray,
    freqs: MLXArray? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXFast.RoPE(
        array, dimensions: dimensions, traditional: traditional, base: base, scale: scale,
        offset: offset, freqs: freqs, stream: stream)
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
///
/// Specifically this implements:
///
/// ```swift
/// var scores = (queries * self.scale).matmul(keys.transposed(0, 1, 3, 2))
/// if let mask {
///     scores = scores + mask
/// }
///
/// scores = softMax(scores.asType(.float32), axis: -1).asType(scores.dtype)
///
/// return matmul(scores, values).transposed(0, 2, 1, 3)
/// ```
public func scaledDotProductAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?,
    memoryEfficientThreshold: Int? = nil, stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values, scale: scale, mask: mask,
        memoryEfficientThreshold: memoryEfficientThreshold, stream: stream)
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// - Parameters:
///   - x: input array
///   - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///     with the same size as the last axis of `x`.
///   - eps: A small additive constant for numerical stability
///   - stream: stream or device to evaluate on
public func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float, stream: StreamOrDevice = .default)
    -> MLXArray
{
    return MLXFast.rmsNorm(x, weight: weight, eps: eps, stream: stream)
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// - Parameters:
///   - x: input array
///   - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///     with the same size as the last axis of `x`.  If not given no scaling will occur.
///   - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///     with the same size as the last axis of `x`.  It not given no offset will occur.
///   - eps: A small additive constant for numerical stability
///   - stream: stream or device to evaluate on
public func layerNorm(
    _ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil, eps: Float,
    stream: StreamOrDevice = .default
) -> MLXArray {
    return MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps, stream: stream)
}

// MARK: - TurboQuant Framework Kernels

extension MLXFast {

    /// Compute Q*K attention scores from packed codebook-quantized keys.
    public static func turboScore(
        _ qRot: MLXArray, packed: MLXArray, norms: MLXArray, codebook: MLXArray,
        tokenCount: Int, repeatCount: Int, bits: Int, dim: Int,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_turbo_score(&result, qRot.ctx, packed.ctx, norms.ctx, codebook.ctx,
            Int32(tokenCount), Int32(repeatCount), Int32(bits), Int32(dim), stream.ctx)
        return MLXArray(result)
    }

    /// Fused norm+rotate+quantize+pack (dense rotation). Returns (packed, norms).
    public static func turboEncode(
        _ input: MLXArray, rotation: MLXArray, boundaries: MLXArray, codebook: MLXArray,
        bits: Int, dim: Int, stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_turbo_encode(&result, input.ctx, rotation.ctx, boundaries.ctx, codebook.ctx,
            Int32(bits), Int32(dim), stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// Fused norm+WHT+quantize+pack (Walsh-Hadamard). Returns (packed, norms).
    public static func turboEncodeWHT(
        _ input: MLXArray, whtSigns: MLXArray, boundaries: MLXArray,
        bits: Int, dim: Int, stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_turbo_encode_wht(&result, input.ctx, whtSigns.ctx, boundaries.ctx,
            Int32(bits), Int32(dim), stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// TurboFlash pass 1 (non-causal). Returns (o_partials, m_partials, l_partials).
    public static func turboFlashPass1(
        _ qRot: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int, numBlocks: Int, blockSize: Int,
        keyBits: Int, valueBits: Int, dim: Int,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_turbo_flash_pass1(&result, qRot.ctx,
            keyPacked.ctx, keyNorms.ctx, keyCodebook.ctx,
            valPacked.ctx, valNorms.ctx, valCodebook.ctx,
            Int32(tokenCount), Int32(repeatCount), Int32(numBlocks), Int32(blockSize),
            Int32(keyBits), Int32(valueBits), Int32(dim), stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// TurboFlash pass 1 (causal). Returns (o_partials, m_partials, l_partials).
    public static func turboFlashPass1Causal(
        _ qRot: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int, numBlocks: Int, blockSize: Int,
        L: Int, qOffset: Int,
        keyBits: Int, valueBits: Int, dim: Int,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_turbo_flash_pass1_causal(&result, qRot.ctx,
            keyPacked.ctx, keyNorms.ctx, keyCodebook.ctx,
            valPacked.ctx, valNorms.ctx, valCodebook.ctx,
            Int32(tokenCount), Int32(repeatCount), Int32(numBlocks), Int32(blockSize),
            Int32(L), Int32(qOffset),
            Int32(keyBits), Int32(valueBits), Int32(dim), stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// TurboFlash pass 1 NR0 (non-causal, multi-row). Returns (o_partials, m_partials, l_partials).
    public static func turboFlashPass1NR0(
        _ qRot: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int, numBlocks: Int, blockSize: Int,
        keyBits: Int, valueBits: Int, dim: Int, nr0: Int,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_turbo_flash_pass1_nr0(&result, qRot.ctx,
            keyPacked.ctx, keyNorms.ctx, keyCodebook.ctx,
            valPacked.ctx, valNorms.ctx, valCodebook.ctx,
            Int32(tokenCount), Int32(repeatCount), Int32(numBlocks), Int32(blockSize),
            Int32(keyBits), Int32(valueBits), Int32(dim), Int32(nr0), stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// TurboFlash pass 1 NR0 (causal, multi-row). Returns (o_partials, m_partials, l_partials).
    public static func turboFlashPass1NR0Causal(
        _ qRot: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int, numBlocks: Int, blockSize: Int,
        L: Int, qOffset: Int,
        keyBits: Int, valueBits: Int, dim: Int, nr0: Int,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_turbo_flash_pass1_nr0_causal(&result, qRot.ctx,
            keyPacked.ctx, keyNorms.ctx, keyCodebook.ctx,
            valPacked.ctx, valNorms.ctx, valCodebook.ctx,
            Int32(tokenCount), Int32(repeatCount), Int32(numBlocks), Int32(blockSize),
            Int32(L), Int32(qOffset),
            Int32(keyBits), Int32(valueBits), Int32(dim), Int32(nr0), stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// TurboFlash pass 2: cross-block online softmax reduction.
    public static func turboFlashPass2(
        oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray,
        numBlocks: Int, dim: Int, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_turbo_flash_pass2(&result, oPartials.ctx, mPartials.ctx, lPartials.ctx,
            Int32(numBlocks), Int32(dim), stream.ctx)
        return MLXArray(result)
    }

    /// TurboFlash pass 2 with fused output rotation.
    public static func turboFlashPass2Fused(
        oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray,
        valRotation: MLXArray,
        numBlocks: Int, dim: Int, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_turbo_flash_pass2_fused(&result, oPartials.ctx, mPartials.ctx, lPartials.ctx,
            valRotation.ctx, Int32(numBlocks), Int32(dim), stream.ctx)
        return MLXArray(result)
    }

    /// Weighted sum of codebook-quantized values (V aggregation).
    public static func turboValue(
        _ weights: MLXArray, packed: MLXArray, norms: MLXArray, codebook: MLXArray,
        tokenCount: Int, repeatCount: Int, sparseThreshold: Float,
        bits: Int, dim: Int, stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_turbo_value(&result, weights.ctx, packed.ctx, norms.ctx, codebook.ctx,
            Int32(tokenCount), Int32(repeatCount), sparseThreshold,
            Int32(bits), Int32(dim), stream.ctx)
        return MLXArray(result)
    }

    /// Bulk-dequantize a packed `[B, H, T, PackedWidth]` codec buffer back to
    /// BF16/FP16 `[B, H, T, dim]` in rotated codec space — one Metal dispatch.
    ///
    /// Output dtype must be `.bfloat16` or `.float16`.
    public static func turboBulkDequantRotated(
        _ packed: MLXArray, norms: MLXArray, codebook: MLXArray,
        bits: Int, dim: Int, outputDType: DType,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_turbo_bulk_dequant_rotated(&result, packed.ctx, norms.ctx, codebook.ctx,
            Int32(bits), Int32(dim), outputDType.cmlxDtype, stream.ctx)
        return MLXArray(result)
    }
}

// MARK: - GatedDelta Framework Kernel

extension MLXFast {

    /// GatedDeltaNet recurrent step (standard or fused).
    /// Returns (y, state_out).
    public static func gatedDeltaStep(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray, state: MLXArray,
        mask: MLXArray? = nil,
        T: Int, fused: Bool,
        Dk: Int, Dv: Int, Hk: Int, Hv: Int,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_gated_delta_step(&result,
            q.ctx, k.ctx, v.ctx, g.ctx, beta.ctx, state.ctx,
            mask?.ctx ?? mlx_array_new(),
            Int32(T), fused, Int32(Dk), Int32(Dv), Int32(Hk), Int32(Hv), stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// GatedDeltaNet fused recurrence step (norm+gate+beta fused inside kernel).
    /// Returns (y, state_out).
    public static func gatedDeltaStepFused(
        qRaw: MLXArray, kRaw: MLXArray, v: MLXArray,
        a: MLXArray, bInput: MLXArray,
        aLog: MLXArray, dtBias: MLXArray,
        state: MLXArray, mask: MLXArray? = nil,
        T: Int, Dk: Int, Dv: Int, Hk: Int, Hv: Int,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_gated_delta_step_fused(&result,
            qRaw.ctx, kRaw.ctx, v.ctx, a.ctx, bInput.ctx,
            aLog.ctx, dtBias.ctx, state.ctx,
            mask?.ctx ?? mlx_array_new(),
            Int32(T), Int32(Dk), Int32(Dv), Int32(Hk), Int32(Hv), stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// GatedDeltaNet forward step with per-step `delta_t` tape capture.
    /// Companion to `gatedDeltaStep` — same forward computation but also
    /// writes per-step delta to a tape output buffer. Used by speculative-
    /// decoder verify forwards on hybrid GDN+Attention models (Qwen 3.5 / 3.6)
    /// to record innovations for possible partial-accept rollback via
    /// `stateReplay`.
    /// Returns (y [B, T, Hv, Dv], state_out [B, Hv, Dv, Dk], delta_log [B, T, Hv, Dv]).
    public static func gatedDeltaStepRecord(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray, state: MLXArray,
        mask: MLXArray? = nil,
        T: Int,
        Dk: Int, Dv: Int, Hk: Int, Hv: Int,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_gated_delta_step_record(&result,
            q.ctx, k.ctx, v.ctx, g.ctx, beta.ctx, state.ctx,
            mask?.ctx ?? mlx_array_new(),
            Int32(T), Int32(Dk), Int32(Dv), Int32(Hk), Int32(Hv), stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// Tape-replay rollback. Re-folds the accepted prefix `[0, accepted)` of
    /// an innovation tape (per-step `(delta_t, k_t, g_t)` triples) onto a
    /// pre-record state snapshot. k_log carries GQA-expanded keys so the
    /// kernel stride is `Hv * Dk` (not `Hk * Dk`). Adopts upstream dflash-mlx
    /// correctness patterns from day 1 (masked-timestep fix + branchless
    /// `metal::select`).
    /// Returns (state_out [B, Hv, Dv, Dk]).
    public static func stateReplay(
        deltaLog: MLXArray, kLog: MLXArray, gLog: MLXArray,
        state: MLXArray, mask: MLXArray? = nil,
        T_log: Int, accepted: Int,
        Dk: Int, Dv: Int, Hk: Int, Hv: Int,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_state_replay(&result,
            deltaLog.ctx, kLog.ctx, gLog.ctx, state.ctx,
            mask?.ctx ?? mlx_array_new(),
            Int32(T_log), Int32(accepted),
            Int32(Dk), Int32(Dv), Int32(Hk), Int32(Hv), stream.ctx)
        return mlx_vector_array_values(result)
    }
}

// MARK: - SSM Framework Kernel

extension MLXFast {

    /// Mamba2 SSM recurrent step (single timestep).
    /// Returns (out, state_out).
    public static func ssmStep(
        X: MLXArray, ALog: MLXArray, B: MLXArray, C: MLXArray,
        D: MLXArray, dt: MLXArray, state: MLXArray,
        Dh: Int, Ds: Int, H: Int, G: Int,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_ssm_step(&result,
            X.ctx, ALog.ctx, B.ctx, C.ctx, D.ctx, dt.ctx, state.ctx,
            Int32(Dh), Int32(Ds), Int32(H), Int32(G), stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// Spec 040: Mamba sequential step with per-step delta-log capture for
    /// n-gram speculative-decode state replay.
    ///
    /// Returns `[y, state_out, dA_log, dBx_log]` (4 arrays). Drop-in for the
    /// L>1 Mamba forward path during a `cache.isRecording` window — the delta
    /// log is consumed by `ssmReplay` on rollback.
    ///
    /// `mask` is an optional `[B, T]` bool array; masked timesteps fold
    /// `dA=1, dBx=0` so any rollback past a masked t is identity-preserving.
    public static func ssmStepRecord(
        x: MLXArray, ALog: MLXArray, B: MLXArray, C: MLXArray,
        D: MLXArray, dt: MLXArray, state: MLXArray,
        mask: MLXArray? = nil,
        stream: StreamOrDevice = .default
    ) -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }
        mlx_fast_ssm_step_record(
            &result,
            x.ctx, ALog.ctx, B.ctx, C.ctx, D.ctx, dt.ctx, state.ctx,
            (mask ?? .mlxNone).ctx,
            stream.ctx)
        return mlx_vector_array_values(result)
    }

    /// Spec 040: rollback step — re-folds the first `acceptedPrefix` entries
    /// of a delta log produced by `ssmStepRecord` onto a recurrent state
    /// snapshot. Returns `state_after_k` of the same shape as `stateSnapshot`.
    public static func ssmReplay(
        stateSnapshot: MLXArray,
        dALog: MLXArray,
        dBxLog: MLXArray,
        acceptedPrefix: Int,
        mask: MLXArray? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_ssm_replay(
            &result,
            stateSnapshot.ctx, dALog.ctx, dBxLog.ctx,
            Int32(acceptedPrefix),
            (mask ?? .mlxNone).ctx,
            stream.ctx)
        return MLXArray(result)
    }

    /// Spec 041 phase 1.1: fused flash quantized SDPA.
    ///
    /// Same shape and semantics as `scaledDotProductAttention(... sinks:)` but
    /// consumes affine-quantized K/V triples and dequantises inline inside the
    /// tiled online-softmax loop. Avoids materialising the
    /// `[B, H, T_q, T_kv]` score matrix that the discrete
    /// `quantizedMM → softmax → quantizedMM` path produces.
    ///
    /// - Parameters:
    ///   - queries: `[B, n_q_heads, T_q, D]`
    ///   - kPacked / kScales / kBiases: affine quantization triple
    ///     `[B, n_kv_heads, T_kv, ...]`. K head dim derived from queries.
    ///   - vPacked / vScales / vBiases: same shape rule. V dim may differ
    ///     from K dim (rare; defaults equal).
    ///   - bits / groupSize: affine codec params (currently `{2,3,4,6,8}` ×
    ///     `64` instantiated).
    ///   - mask: optional bool or float mask. `.causal` mode passes
    ///     `mask: nil` and `causal: true`.
    ///   - sinks: optional `[n_q_heads]` per-Q-head sink logits (GPT-OSS).
    /// TurboQuant fused single-pass SDPA with sinks — spec 041 phase 1.1
    /// follow-up. MSE-codec equivalent of `flashQuantizedSDPA(...)`. Single
    /// kernel dispatch, online softmax inline with sinks fold — no
    /// pass1/pass2 split (sidesteps the graph-fusion incoherence that the
    /// previous β-with-sinks drafts hit on GPT-OSS-20B).
    ///
    /// Output is in rotated V space; caller applies the inverse codec
    /// rotation Π_v^T afterward.
    public static func turboFlashSDPAv(
        queries: MLXArray,
        kPacked: MLXArray, kNorms: MLXArray, kCodebook: MLXArray,
        vPacked: MLXArray, vNorms: MLXArray, vCodebook: MLXArray,
        keyBits: Int, valueBits: Int, dim: Int, repeatCount: Int,
        sinks: MLXArray? = nil,
        causal: Bool = false,
        windowSize: Int = -1,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        mlx_fast_turbo_flash_sdpa_v(
            &result,
            queries.ctx,
            kPacked.ctx, kNorms.ctx, kCodebook.ctx,
            vPacked.ctx, vNorms.ctx, vCodebook.ctx,
            Int32(keyBits), Int32(valueBits), Int32(dim), Int32(repeatCount),
            (sinks ?? .mlxNone).ctx,
            causal,
            Int32(windowSize),
            stream.ctx)
        return MLXArray(result)
    }

    public static func flashQuantizedSDPA(
        queries: MLXArray,
        kPacked: MLXArray, kScales: MLXArray, kBiases: MLXArray,
        vPacked: MLXArray, vScales: MLXArray, vBiases: MLXArray,
        scale: Float,
        bits: Int,
        groupSize: Int,
        causal: Bool = false,
        windowSize: Int = -1,
        mask: MLXArray? = nil,
        sinks: MLXArray? = nil,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        var result = mlx_array_new()
        // Sliding window requires causal (matches the C++ entry-point check).
        let maskMode = (causal || windowSize > 0) ? "causal" : ""
        mlx_fast_flash_quantized_sdpa(
            &result,
            queries.ctx,
            kPacked.ctx, kScales.ctx, kBiases.ctx,
            vPacked.ctx, vScales.ctx, vBiases.ctx,
            scale,
            Int32(bits), Int32(groupSize),
            maskMode,
            (mask ?? .mlxNone).ctx,
            (sinks ?? .mlxNone).ctx,
            Int32(windowSize),
            stream.ctx)
        return MLXArray(result)
    }
}
