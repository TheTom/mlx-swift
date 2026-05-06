#ifdef __cplusplus
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include <Cmlx/mlx-api.h>

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/**
 * Reset the cumulative dispatch counter on the Metal command encoder of the
 * default device. Used to audit per-region dispatch counts — e.g. bracket a
 * decode-token forward pass with reset + total_dispatches to measure the
 * dispatch count that an ICB prototype would need to pre-encode.
 */
MLX_API void reset_dispatch_counter();

/**
 * Read the cumulative dispatch counter since the last reset. A dispatch is
 * either `dispatchThreadgroups` or `dispatchThreads` — i.e. one GPU kernel
 * launch.
 */
MLX_API uint64_t total_dispatches();

} // namespace mlx::core::metal
#endif
