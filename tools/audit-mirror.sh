#!/bin/bash
# audit-mirror.sh — drift detector for Source/Cmlx/mlx-generated/metal/
#
# Why this exists:
#   The `mlx-generated/metal/` directory is a hand-maintained mirror of
#   the canonical Metal kernel sources at
#   `Source/Cmlx/mlx/mlx/backend/metal/kernels/` (mlx submodule). SwiftPM
#   can't compile `.metal` files, so we pre-bake them into the mirror with
#   include paths rewritten to the short form (`#include "utils.h"`
#   instead of `#include "mlx/backend/metal/kernels/utils.h"`). The mirror
#   is the actual input to `scripts/build-metallib.sh` in downstream
#   consumers — what gets compiled into `mlx.metallib`.
#
# Failure mode this catches:
#   When the submodule's kernel body changes (bug fix, perf improvement,
#   new template instantiation) but the mirror isn't re-synced via
#   `tools/update-mlx.sh`, the metallib silently compiles the stale mirror
#   body. The function signatures remain compatible so nothing fails at
#   link time — only the numerical output diverges at runtime. This is
#   the failure pattern that broke 3-bit turbo flash SDPA on
#   ekryski/mlx-swift-lm PR #215.
#
# What this audit does NOT catch:
#   - Top-level `mlx-generated/*.cpp` JIT preambles. Those are generated
#     by cmake's `make <kernel>` step and embed kernel source as raw
#     strings; we can't cheaply diff them without re-running cmake.
#     Mitigation: signature drift in those is caught by the existing
#     `make doctor` symbol check in downstream `mlx-swift-lm`.
#   - mlx-swift-only fork kernels (Class B) like `warp_moe_*`. Those
#     have no submodule counterpart by design and are skipped.
#   - New kernel files added to the submodule but not yet mirrored
#     (coverage gap, not body drift). The deliberate `update-mlx.sh`
#     flow is where that gets handled.
#
# Exit codes:
#   0  — mirror is in sync with submodule
#   1  — drift detected (non-include-path differences)
#   2  — usage error / missing inputs

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
MIRROR="$ROOT_DIR/Source/Cmlx/mlx-generated/metal"
SUB="$ROOT_DIR/Source/Cmlx/mlx/mlx/backend/metal/kernels"

# Submodule include prefix that gets rewritten to flat form in the mirror.
INCLUDE_PREFIX="mlx/backend/metal/kernels/"

if [ ! -d "$MIRROR" ]; then
    echo "Error: mirror directory not found at $MIRROR" >&2
    echo "       Run from the mlx-swift repo root or pass the right path." >&2
    exit 2
fi

if [ ! -d "$SUB" ]; then
    echo "Error: submodule kernel source not found at $SUB" >&2
    echo "       Run: git submodule update --init --recursive" >&2
    exit 2
fi

drift_count=0
drift_files=()
checked=0

# Iterate every file at the top level of the mirror (depth 1). The mirror
# also has subdirectories (`steel/`, `fft/`, `indexing/`, `reduction/`)
# but those aren't currently fork-touched, so the top-level set is what
# matters for the bug pattern this audit is targeting. Expand the find
# scope here if a fork kernel ever shows up in a subdir.
while IFS= read -r f; do
    name=$(basename "$f")
    sub_path="$SUB/$name"

    # Skip files that don't exist in the submodule — those are Class B
    # (mlx-swift-only) kernels by design (e.g. `warp_moe_*`). There's no
    # canonical source to compare against.
    [ -f "$sub_path" ] || continue

    checked=$((checked + 1))

    # Normalize: rewrite the submodule's full include paths to the
    # mirror's short form before diffing.
    if ! diff_out=$(diff "$f" <(sed -E "s|#include \"${INCLUDE_PREFIX}([^\"]*)\"|#include \"\\1\"|g" "$sub_path") 2>&1); then
        drift_count=$((drift_count + 1))
        drift_files+=("$name")
        echo "DRIFT: $name"
        # Show first 20 lines of the diff for triage. Full diff is one
        # `git diff` away once the developer knows what file to look at.
        echo "$diff_out" | head -20 | sed 's/^/  /'
        echo "  (...)"
        echo ""
    fi
done < <(find "$MIRROR" -maxdepth 1 \( -name '*.metal' -o -name '*.h' \) -type f)

if [ "$drift_count" -gt 0 ]; then
    echo "============================================================"
    echo "Mirror drift detected in $drift_count file(s):"
    for f in "${drift_files[@]}"; do
        echo "  - $f"
    done
    echo ""
    echo "The hand-maintained mirror at \`Source/Cmlx/mlx-generated/metal/\`"
    echo "has fallen out of sync with the canonical submodule sources at"
    echo "\`Source/Cmlx/mlx/mlx/backend/metal/kernels/\`. The metallib that"
    echo "downstream consumers compile will use the STALE mirror, not the"
    echo "fixed submodule source — silently producing wrong numerics."
    echo ""
    echo "Fix: ./tools/update-mlx.sh"
    echo "     (requires cmake + a working mlx build)"
    echo "============================================================"
    exit 1
fi

echo "Mirror in sync — $checked file(s) checked."
exit 0
