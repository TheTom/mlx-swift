/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_FAST_H
#define MLX_FAST_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup fast Fast custom operations
 */
/**@{*/

typedef struct mlx_fast_cuda_kernel_config_ {
  void* ctx;
} mlx_fast_cuda_kernel_config;
mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void);
void mlx_fast_cuda_kernel_config_free(mlx_fast_cuda_kernel_config cls);

int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value);
int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose);
int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_cuda_kernel_ {
  void* ctx;
} mlx_fast_cuda_kernel;

mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory);

void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls);

int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream);

int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s);

typedef struct mlx_fast_metal_kernel_config_ {
  void* ctx;
} mlx_fast_metal_kernel_config;
mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void);
void mlx_fast_metal_kernel_config_free(mlx_fast_metal_kernel_config cls);

int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value);
int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose);
int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_metal_kernel_ {
  void* ctx;
} mlx_fast_metal_kernel;

mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs);

void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls);

int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream);

int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s);
int mlx_fast_rms_norm_residual(
    mlx_array* res,
    const mlx_array x,
    const mlx_array residual,
    const mlx_array weight,
    float eps,
    const mlx_stream s);
int mlx_fast_rms_norm_rope(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight,
    const mlx_array inv_freqs,
    float eps,
    int offset,
    int n_heads,
    int seq_len,
    const mlx_stream s);
int mlx_fast_rms_norm_qgemv(
    mlx_array* res,
    const mlx_array x,
    const mlx_array norm_weight,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases,
    float eps,
    int group_size,
    const mlx_stream s);
int mlx_fast_batched_qkv_qgemv(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w_q, const mlx_array scales_q, const mlx_array biases_q,
    const mlx_array w_k, const mlx_array scales_k, const mlx_array biases_k,
    const mlx_array w_v, const mlx_array scales_v, const mlx_array biases_v,
    int group_size,
    const mlx_stream s);
int mlx_fast_warp_moe_gate_up(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w, const mlx_array scales, const mlx_array biases,
    const mlx_array indices,
    int group_size, int hidden_dims, int activation_type,
    const mlx_stream s);
int mlx_fast_warp_moe_down(
    mlx_array* res,
    const mlx_array activated,
    const mlx_array w, const mlx_array scales, const mlx_array biases,
    const mlx_array indices, const mlx_array scores,
    int group_size, int hidden_dims, int out_dims,
    const mlx_stream s);
int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_rope_dynamic(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    const mlx_array offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s);

int mlx_fast_scaled_dot_product_attention_sliding(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    int window_size,
    const mlx_stream s);

// TurboQuant
int mlx_fast_turbo_score(mlx_array* res, const mlx_array q_rot, const mlx_array packed, const mlx_array norms, const mlx_array codebook, int token_count, int repeat_count, int bits, int dim, const mlx_stream s);
int mlx_fast_turbo_encode(mlx_vector_array* res, const mlx_array input, const mlx_array rotation, const mlx_array boundaries, const mlx_array codebook, int bits, int dim, const mlx_stream s);
int mlx_fast_turbo_encode_wht(mlx_vector_array* res, const mlx_array input, const mlx_array wht_signs, const mlx_array boundaries, int bits, int dim, const mlx_stream s);
int mlx_fast_turbo_flash_pass1(mlx_vector_array* res, const mlx_array q_rot, const mlx_array key_packed, const mlx_array key_norms, const mlx_array key_codebook, const mlx_array val_packed, const mlx_array val_norms, const mlx_array val_codebook, int token_count, int repeat_count, int num_blocks, int block_size, int key_bits, int value_bits, int dim, const mlx_stream s);
int mlx_fast_turbo_flash_pass1_causal(mlx_vector_array* res, const mlx_array q_rot, const mlx_array key_packed, const mlx_array key_norms, const mlx_array key_codebook, const mlx_array val_packed, const mlx_array val_norms, const mlx_array val_codebook, int token_count, int repeat_count, int num_blocks, int block_size, int L, int q_offset, int key_bits, int value_bits, int dim, const mlx_stream s);
int mlx_fast_turbo_flash_pass1_nr0(mlx_vector_array* res, const mlx_array q_rot, const mlx_array key_packed, const mlx_array key_norms, const mlx_array key_codebook, const mlx_array val_packed, const mlx_array val_norms, const mlx_array val_codebook, int token_count, int repeat_count, int num_blocks, int block_size, int key_bits, int value_bits, int dim, int nr0, const mlx_stream s);
int mlx_fast_turbo_flash_pass1_nr0_causal(mlx_vector_array* res, const mlx_array q_rot, const mlx_array key_packed, const mlx_array key_norms, const mlx_array key_codebook, const mlx_array val_packed, const mlx_array val_norms, const mlx_array val_codebook, int token_count, int repeat_count, int num_blocks, int block_size, int L, int q_offset, int key_bits, int value_bits, int dim, int nr0, const mlx_stream s);
int mlx_fast_turbo_flash_pass2(mlx_array* res, const mlx_array o_partials, const mlx_array m_partials, const mlx_array l_partials, int num_blocks, int dim, const mlx_stream s);
int mlx_fast_turbo_flash_pass2_fused(mlx_array* res, const mlx_array o_partials, const mlx_array m_partials, const mlx_array l_partials, const mlx_array val_rotation, int num_blocks, int dim, const mlx_stream s);
int mlx_fast_turbo_value(mlx_array* res, const mlx_array weights, const mlx_array packed, const mlx_array norms, const mlx_array codebook, int token_count, int repeat_count, float sparse_threshold, int bits, int dim, const mlx_stream s);
// GatedDelta
int mlx_fast_gated_delta_step(mlx_vector_array* res, const mlx_array q, const mlx_array k, const mlx_array v, const mlx_array g, const mlx_array beta, const mlx_array state, const mlx_array mask, int T, bool fused, int Dk, int Dv, int Hk, int Hv, const mlx_stream s);
int mlx_fast_gated_delta_step_fused(mlx_vector_array* res, const mlx_array q_raw, const mlx_array k_raw, const mlx_array v, const mlx_array a, const mlx_array b_input, const mlx_array a_log, const mlx_array dt_bias, const mlx_array state, const mlx_array mask, int T, int Dk, int Dv, int Hk, int Hv, const mlx_stream s);
// SSM
int mlx_fast_ssm_step(mlx_vector_array* res, const mlx_array X, const mlx_array A_log, const mlx_array B, const mlx_array C, const mlx_array D, const mlx_array dt, const mlx_array state, int Dh, int Ds, int H, int G, const mlx_stream s);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif
