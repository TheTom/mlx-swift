// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "../../../utils.h"

#include "../../../steel/attn/kernels/steel_attention.h"

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1, mname, mtype)

// BD=256: BQ=32, BK=16, WM=4 — matches BD≤128 ratios. 28.5KB tgp mem fits 32KB.
// float16/bfloat16 only (fp32 would need 57KB, exceeding 32KB limit).
#define instantiate_attn_shapes_helper_bd256(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 256, 4, 1, mname, mtype)

#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

#define instantiate_attn_mask_helper_bd256(iname, itype) \
    instantiate_attn_shapes_helper_bd256(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper_bd256(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper_bd256(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat16_t);
instantiate_attn_mask_helper_bd256(bfloat16, bfloat16_t);

instantiate_attn_mask_helper(float32, float);
// BD=256 is NOT instantiated for float32 — threadgroup memory exceeds 32KB
// limit (41KB at 4 bytes/element). Only float16/bfloat16 fit. Dispatch gates on dtype.
// clang-format on
