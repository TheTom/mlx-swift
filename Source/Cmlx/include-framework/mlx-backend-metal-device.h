#ifdef __cplusplus
// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <Cmlx/Metal.hpp>
#include <os/signpost.h>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <Cmlx/mlx-array.h>
#include <Cmlx/mlx-backend-metal-resident.h>
#include <Cmlx/mlx-device.h>

namespace mlx::core::metal {

// Per-kernel-dispatch os_signpost emitter. Opt-in via
// `MLX_METAL_PROFILE=1` — when unset, `signpost_log()` returns
// `OS_LOG_DISABLED` and all `os_signpost_*` calls short-circuit on
// an atomic-load + branch, costing a handful of nanoseconds.
//
// Subsystem `ai.mlx.metal`, category `PointsOfInterest`. Instruments
// renders each kernel dispatch as an interval labelled with the
// pipeline state's name (e.g. `sdpa_unified_vector_bf16_128_128`),
// overlaid on Metal System Trace's GPU kernel timeline so you can
// see both the CPU encoding window and the subsequent GPU execution
// window side by side.
//
// Cost measurement: ~40 ns per begin/end pair on M-series silicon.
// For a decode step with ~500 kernel dispatches, that's ~40 µs of
// overhead per token — well under 0.2% on GPT-OSS 20B's ~22 ms
// steady-state decode budget.
MLX_API os_log_t signpost_log();
MLX_API bool signposts_enabled();

using MTLFCList =
    std::vector<std::tuple<const void*, MTL::DataType, NS::UInteger>>;

class Device;

class MLX_API CommandEncoder {
 public:
  CommandEncoder(Device& d, int index, ResidencySet& residency_set);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  struct ConcurrentContext {
    ConcurrentContext(CommandEncoder& enc) : enc(enc) {
      enc.concurrent_ = true;
    }
    ~ConcurrentContext() {
      enc.concurrent_ = false;
      enc.prev_outputs_.insert(
          enc.concurrent_outputs_.begin(), enc.concurrent_outputs_.end());
      enc.concurrent_outputs_.clear();
    }

   private:
    CommandEncoder& enc;
  };

  void set_buffer(const MTL::Buffer* buf, int idx, int64_t offset = 0);
  void set_input_array(const array& a, int idx, int64_t offset = 0);
  void set_output_array(array& a, int idx, int64_t offset = 0);
  void register_output_array(const array& a);

  void add_temporary(array arr);
  void add_temporaries(std::vector<array> arrays);

  void dispatch_threadgroups(MTL::Size grid_dims, MTL::Size group_dims);
  void dispatch_threads(MTL::Size grid_dims, MTL::Size group_dims);
  void maybeInsertBarrier();

  // Sets the active compute pipeline state. Out-of-line because it
  // also opens an `os_signpost` interval when `MLX_METAL_PROFILE=1`;
  // the corresponding close happens inside `dispatch_threadgroups` /
  // `dispatch_threads`.
  void set_compute_pipeline_state(MTL::ComputePipelineState* kernel);

  template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
  void set_vector_bytes(const Vec& vec, size_t nelems, int idx) {
    get_command_encoder()->setBytes(
        vec.data(), nelems * sizeof(typename Vec::value_type), idx);
  }
  template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
  void set_vector_bytes(const Vec& vec, int idx) {
    return set_vector_bytes(vec, vec.size(), idx);
  }

  template <typename T>
  void set_bytes(const T* v, int n, int idx) {
    return get_command_encoder()->setBytes(v, n * sizeof(T), idx);
  }

  template <typename T>
  void set_bytes(const T& v, int idx) {
    return get_command_encoder()->setBytes(&v, sizeof(T), idx);
  }

  void set_threadgroup_memory_length(size_t length, int idx) {
    get_command_encoder()->setThreadgroupMemoryLength(length, idx);
  }

  ConcurrentContext start_concurrent() {
    return ConcurrentContext(*this);
  }

  void barrier();
  void end_encoding();
  bool needs_commit() const;
  void commit();
  void synchronize();

  // Dispatch counter — cumulative count of dispatchThreadgroups /
  // dispatchThreads calls since the last reset_dispatch_counter(). Used to
  // audit per-token dispatch counts for ICB feasibility studies and to
  // quantify the CPU-encoding optimization target.
  void reset_dispatch_counter();
  uint64_t total_dispatches() const;

  MTL::CommandQueue* get_command_queue() const {
    return queue_.get();
  }
  MTL::CommandBuffer* get_command_buffer() const {
    return buffer_.get();
  }

  // Transfer ownership of MTL::Buffer pointers retained at bind for the
  // current command buffer. Called from eval()'s completion-handler
  // closure setup; the closure is responsible for releasing each pointer
  // when the command buffer completes. See set_buffer / set_input_array
  // for the matching retain.
  std::vector<MTL::Buffer*> take_retained_buffers();

 private:
  MTL::ComputeCommandEncoder* get_command_encoder();

  Device& device_;
  bool exiting_{false};

  // Buffer that stores encoded commands.
  NS::SharedPtr<MTL::CommandQueue> queue_;
  NS::SharedPtr<MTL::CommandBuffer> buffer_;
  int buffer_ops_{0};
  // Cumulative bytes of unique input buffers referenced in the current
  // command buffer. Incremented in `set_input_array` on first sighting.
  size_t buffer_input_sizes_{0};
  // Cumulative bytes of unique output buffers materialized in the current
  // command buffer. Tracked separately from inputs so the commit heuristic
  // can see the full memory footprint (allocators dedupe input-side
  // re-uses of output pointers, which would otherwise hide output pressure).
  size_t buffer_output_sizes_{0};
  uint64_t total_dispatches_{0};

  // Encoder for issuing GPU commands.
  // The members are used within a single ComputeCommandEncoder and will be
  // reset after calling end_encoding().
  NS::SharedPtr<MTL::ComputeCommandEncoder> encoder_;
  NS::SharedPtr<MTL::Fence> fence_;
  bool needs_barrier_{false};
  bool concurrent_{false};
  std::vector<array> temporaries_;
  std::unordered_set<MTL::Resource*> prev_outputs_;
  std::unordered_set<MTL::Resource*> next_outputs_;
  std::unordered_set<MTL::Resource*> concurrent_outputs_;
  std::unordered_set<const void*> all_inputs_;
  std::unordered_set<const void*> all_outputs_;
  // MTL::Buffer* pointers retained at bind for the current command buffer.
  // Allocator buffers use MTLResourceHazardTrackingModeUntracked and command
  // buffers use commandBufferWithUnretainedReferences(); both APIs require
  // the application to keep bound buffers alive until CB completion. We
  // accumulate retains here per-CB and transfer them to the eval-side
  // completion handler via take_retained_buffers().
  std::vector<MTL::Buffer*> retained_buffers_;

  // A map of prior command encoder outputs to their corresponding fence.
  std::unordered_map<const void*, NS::SharedPtr<MTL::Fence>> prev_ce_outputs_;
  std::mutex outputs_mtx_;

  // Signpost state for MLX_METAL_PROFILE=1 per-kernel tracing. The
  // ID is generated in `set_compute_pipeline_state` and consumed by
  // the next `dispatch_*` call on this encoder. `OS_SIGNPOST_ID_NULL`
  // means "no interval in flight" (the normal state when signposts
  // are disabled, or between dispatches).
  os_signpost_id_t cur_dispatch_signpost_id_{OS_SIGNPOST_ID_NULL};
};

class MLX_API Device {
 public:
  Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device();

  MTL::Device* mtl_device() {
    return device_.get();
  };

  const std::string& get_architecture() const {
    return arch_;
  }
  int get_architecture_gen() const {
    return arch_gen_;
  }
  std::tuple<int, int> get_max_ops_mb_per_buffer() const {
    return std::make_tuple(max_ops_per_buffer_, max_mb_per_buffer_);
  }

  MTL::Library* get_library(
      const std::string& name,
      const std::string& path = "");

  MTL::Library* get_library(
      const std::string& name,
      const std::function<std::string(void)>& builder);

  void clear_library(const std::string& name);

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  MTL::ComputePipelineState* get_kernel(
      const std::string& base_name,
      const std::string& hash_name = "",
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  ResidencySet& residency_set() {
    return residency_set_;
  }

  // Debug-name lookup for a compute pipeline state. Populated by
  // `get_kernel_` at PSO creation time — keyed by the PSO pointer
  // so callers holding a raw `MTL::ComputePipelineState*` (e.g.
  // the signpost emitter in `set_compute_pipeline_state`) can
  // recover the kernel name without relying on the PSO's built-in
  // `label()` property, which metal-cpp does not expose a setter
  // for. Returns a pointer into the map's stored string if found,
  // or `"?"` otherwise. The returned pointer remains valid for
  // the Device's lifetime — kernels are never evicted once cached.
  const char* pso_name(const MTL::ComputePipelineState* pso) const;

 private:
  NS::SharedPtr<MTL::Library> build_library_(const std::string& source_string);

  NS::SharedPtr<MTL::Function> get_function_(
      const std::string& name,
      MTL::Library* mtl_lib);
  NS::SharedPtr<MTL::Function> get_function_(
      const std::string& name,
      const std::string& specialized_name,
      const MTLFCList& func_consts,
      MTL::Library* mtl_lib);

  NS::SharedPtr<MTL::LinkedFunctions> get_linked_functions_(
      const std::vector<MTL::Function*>& funcs);

  NS::SharedPtr<MTL::ComputePipelineState> get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function);
  NS::SharedPtr<MTL::ComputePipelineState> get_kernel_(
      const std::string& name,
      const MTL::Function* mtl_function,
      const MTL::LinkedFunctions* linked_functions);

  MTL::ComputePipelineState* get_kernel_(
      const std::string& base_name,
      MTL::Library* mtl_lib,
      const std::string& hash_name,
      const MTLFCList& func_consts = {},
      const std::vector<MTL::Function*>& linked_functions = {});

  NS::SharedPtr<MTL::Device> device_;
  ResidencySet residency_set_;

  std::shared_mutex kernel_mtx_;
  std::shared_mutex library_mtx_;
  std::unordered_map<std::string, NS::SharedPtr<MTL::Library>> library_map_;
  NS::SharedPtr<MTL::Library> default_library_;
  std::unordered_map<
      MTL::Library*,
      std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>>
      library_kernels_;
  // PSO → kernel-name index for debug/signpost labelling. Populated
  // under `kernel_mtx_`, read under shared lock. See `pso_name()`.
  std::unordered_map<const MTL::ComputePipelineState*, std::string> pso_names_;
  std::string arch_;
  int arch_gen_;
  int max_ops_per_buffer_;
  int max_mb_per_buffer_;
};

MLX_API Device& device(mlx::core::Device);
MLX_API CommandEncoder& get_command_encoder(Stream s);

std::unordered_map<int, CommandEncoder>& get_command_encoders();
NS::SharedPtr<NS::AutoreleasePool> new_scoped_memory_pool();

bool is_nax_available();

} // namespace mlx::core::metal
#endif
