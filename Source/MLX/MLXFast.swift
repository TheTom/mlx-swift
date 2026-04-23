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
}
