#!/bin/zsh

# See MAINTENANCE.md : Updating `mlx` and `mlx-c`

set -e

if [[ ! -d Source ]]
then
    echo "Please run from the root of the repository, e.g. ./tools/update-mlx.sh"
    exit 1
fi

# copy mlx-c headers to build area
rm -f Source/Cmlx/include/mlx/c/*
cp Source/Cmlx/mlx-c/mlx/c/*.h Source/Cmlx/include/mlx/c

# run the command to do the build-time code generation

mkdir build
cd build
cmake ../Source/Cmlx/mlx -DMLX_METAL_JIT=ON -DMACOS_VERSION=14.0

# run the cmake build to generate the source files
cd mlx/backend/metal
make \
    arange \
    binary \
    binary_ops \
    binary_two \
    conv \
    copy \
    fft \
    fp_quantized \
    fp_quantized_nax \
    gather \
    gather_axis \
    gather_front \
    gemm \
    gemm_nax \
    gemv_masked \
    hadamard \
    logsumexp \
    masked_scatter \
    quantized \
    quantized_nax \
    quantized_utils \
    reduce \
    reduce_utils \
    scan \
    scatter \
    scatter_axis \
    softmax \
    sort \
    steel_attention \
    steel_attention_nax \
    steel_conv \
    steel_conv_3d \
    steel_conv_general \
    steel_gemm_fused \
    steel_gemm_fused_nax \
    steel_gemm_gather \
    steel_gemm_gather_nax \
    steel_gemm_masked \
    steel_gemm_segmented \
    steel_gemm_segmented_nax \
    steel_gemm_splitk \
    steel_gemm_splitk_nax \
    ternary \
    ternary_ops \
    unary \
    unary_ops \
    utils

cd ../../..
make cpu_compiled_preamble

cd ..

# Fork preservation (pre-destructive-rm): capture the names of every
# fork .metal kernel currently in the committed mirror at depth 1
# (Source/Cmlx/mlx-generated/metal/*.metal). The upstream cmake regen
# + fix-metal-includes.sh only know about the canonical KERNEL_LIST
# from `mlx/backend/metal/kernels/CMakeLists.txt`, so anything outside
# that list would be wiped by the `rm -rf` below and never restored.
#
# Two classes of fork kernels need different handling post-regen:
#  - Class A: kernel source LIVES IN the submodule under
#    `mlx/backend/metal/kernels/` but isn't in upstream KERNEL_LIST
#    (turbo_*, gated_delta_*, ssm, rms_norm_qgemv, rms_norm_residual,
#    rms_norm_rope, fused_gate_activation). Post-regen: re-copy from
#    the submodule with includes rewritten.
#  - Class B: kernel exists ONLY in the mlx-swift mirror, no submodule
#    counterpart (warp_moe_*, batched_qkv_qgemv until promoted to
#    canonical). Post-regen: restore from the tmpdir backup.
#
# Importantly, we ONLY auto-detect fork kernels FROM THE EXISTING
# MIRROR — we don't pull arbitrary new .metal files out of the
# submodule. New fork kernels require deliberate action (extending
# the metallib build list, registering wrappers, etc.); the auto-sync
# is just for keeping the existing set in step with submodule changes.
FORK_KERNEL_BACKUP=$(mktemp -d)
FORK_CLASS_A_LIST=""
SUBMODULE_KERNELS_DIR=Source/Cmlx/mlx/mlx/backend/metal/kernels
if [[ -d Source/Cmlx/mlx-generated/metal ]]; then
    find Source/Cmlx/mlx-generated/metal -maxdepth 1 -name "*.metal" -type f | while read -r f; do
        name=$(basename "$f")
        if [[ -f "${SUBMODULE_KERNELS_DIR}/${name}" ]]; then
            # Class A — record name in a sidecar file (`while ... done`
            # in a pipeline runs in a subshell so a plain variable
            # wouldn't survive; the sidecar pattern is robust).
            echo "$name" >> "$FORK_KERNEL_BACKUP/.class_a"
        else
            # Class B — back up the file content itself.
            cp "$f" "$FORK_KERNEL_BACKUP/$name"
        fi
    done
fi

rm -rf Source/Cmlx/mlx-generated/metal
rm -f Source/Cmlx/mlx-generated/*
cp build/mlx/backend/metal/jit/* Source/Cmlx/mlx-generated
cp build/mlx/backend/cpu/compiled_preamble.cpp Source/Cmlx/mlx-generated

# we don't need the cmake build directory any more
rm -rf build

# remove any absolute paths and make them relative to the package root
for x in Source/Cmlx/mlx-generated/*.cpp ; do \
    sed -i .tmp -e "s:`pwd`/::g" $x
done;
rm Source/Cmlx/mlx-generated/*.tmp

# Update the headers (upstream KERNEL_LIST only). This also creates the
# mlx-generated/metal/ directory and copies the upstream-built .metal
# files into it with includes rewritten to the short form.
./tools/fix-metal-includes.sh

# Fork sync (post-fix-metal-includes): re-copy Class A fork kernels
# from the submodule's kernels/ dir into mlx-generated/metal/. Only
# kernels that were ALREADY in the mirror — auto-detected pre-rm into
# `.class_a` — get synced; arbitrary new submodule kernels (like
# fence.metal which needs Metal 3.2) are intentionally NOT picked up.
# Rewrite include directives the same way fix-metal-includes does so
# the short form (`#include "utils.h"`) replaces the canonical full
# path.
if [[ -f "$FORK_KERNEL_BACKUP/.class_a" ]]; then
    while IFS= read -r name; do
        src="${SUBMODULE_KERNELS_DIR}/${name}"
        dest="Source/Cmlx/mlx-generated/metal/${name}"
        if [[ -f "$src" && ! -f "$dest" ]]; then
            cp "$src" "$dest"
            sed -i '' -E "s|#include \"mlx/backend/metal/kernels/([^\"]*)\"|#include \"\\1\"|g" "$dest"
        fi
    done < "$FORK_KERNEL_BACKUP/.class_a"
fi

# Restore Class B (mlx-swift-only) kernels — those have no submodule
# counterpart (warp_moe_*, batched_qkv_qgemv until promoted to
# canonical). Committed already in short-path-include form, no
# rewrite needed.
# `find -exec` instead of glob expansion because the script's shebang
# is /bin/zsh and `compgen` is bash-only.
find "$FORK_KERNEL_BACKUP" -maxdepth 1 -name "*.metal" -type f -exec cp {} Source/Cmlx/mlx-generated/metal/ \;
rm -rf "$FORK_KERNEL_BACKUP"

# prepare xcodeproj files
./tools/update-mlx-xcodeproj.sh
