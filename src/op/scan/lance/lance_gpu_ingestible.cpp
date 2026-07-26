/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/interop.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <op/scan/lance/lance_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>

#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

/// Runs an action on every exit path, including an unwinding one.
template <typename Action>
class scope_exit {
 public:
  explicit scope_exit(Action action) : _action(std::move(action)) {}
  ~scope_exit() { _action(); }

  scope_exit(scope_exit const&)            = delete;
  scope_exit& operator=(scope_exit const&) = delete;
  scope_exit(scope_exit&&)                 = delete;
  scope_exit& operator=(scope_exit&&)      = delete;

 private:
  Action _action;
};

/// One Arrow batch becomes exactly one split, so nothing is merged. The only
/// real work is the two contract obligations the scan driver imposes: tolerate
/// a null push, and guarantee at least one split ever reaches the pipeline.
class lance_batch_coalescer final : public batch_coalescer {
 public:
  using sentinel_factory = std::function<std::unique_ptr<scan_info>()>;

  explicit lance_batch_coalescer(sentinel_factory make_sentinel)
    : _make_sentinel(std::move(make_sentinel))
  {
  }

  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info> info) override
  {
    // A null split reaches the coalescer unfiltered — the driver forwards
    // whatever the metadata task produced.
    if (!info) { return {}; }
    _emitted = true;
    std::vector<std::unique_ptr<scan_info>> out;
    out.push_back(std::move(info));
    return out;
  }

  std::vector<std::unique_ptr<scan_info>> flush() override
  {
    // Zero splits would mean zero GPU tasks, and the pipeline-completion signal
    // would never fire. A stream that yielded nothing still owes one typed
    // zero-row batch.
    if (_emitted) { return {}; }
    _emitted = true;
    std::vector<std::unique_ptr<scan_info>> out;
    out.push_back(_make_sentinel());
    return out;
  }

 private:
  sentinel_factory _make_sentinel;
  bool _emitted{false};
};

}  // namespace

struct lance_gpu_ingestible::impl {
  explicit impl(std::unique_ptr<lance_ingestible_table_info> table_info)
    : info(std::move(table_info))
  {
    if (!info) { throw std::invalid_argument("lance_gpu_ingestible requires table info"); }
    if (!info->api) {
      throw std::invalid_argument("lance_gpu_ingestible requires a Lance FFI API");
    }
    if (info->names.size() != info->output_types.size()) {
      throw std::invalid_argument("lance_gpu_ingestible: names and output types disagree");
    }

    retirer =
      info->retirer_factory ? info->retirer_factory() : sirius::lance::make_synchronizing_retirer();
    if (!retirer) {
      throw std::invalid_argument("lance_gpu_ingestible: retirer factory returned null");
    }
  }

  /// Creates the producer — and with it the dataset handle, the stream, and the
  /// reader thread — the first time work is actually claimed.
  ///
  /// A Sirius plan can be generated and thrown away without ever executing:
  /// prepare-time validation builds one purely to prove the plan is
  /// GPU-translatable. Opening a Lance dataset for a plan nobody runs would be
  /// visible I/O, so nothing happens until the scan driver claims a split.
  sirius::lance::lance_split_producer& ensure_producer()
  {
    std::call_once(producer_once, [this] {
      producer = std::make_unique<sirius::lance::lance_split_producer>(
        info->api, info->spec, info->expect, info->producer_config);
    });
    return *producer;
  }

  std::unique_ptr<lance_ingestible_table_info> info;
  std::unique_ptr<sirius::lance::arrow_batch_retirer> retirer;
  std::once_flag producer_once;
  std::unique_ptr<sirius::lance::lance_split_producer> producer;
};

lance_gpu_ingestible::lance_gpu_ingestible(std::unique_ptr<lance_ingestible_table_info> info)
  : _impl(std::make_unique<impl>(std::move(info)))
{
}

lance_gpu_ingestible::~lance_gpu_ingestible() = default;

std::unique_ptr<batch_coalescer> lance_gpu_ingestible::create_batch_coalescer() const
{
  return std::make_unique<lance_batch_coalescer>([] {
    auto sentinel               = std::make_unique<lance_scan_info>();
    sentinel->is_empty_sentinel = true;
    return sentinel;
  });
}

bool lance_gpu_ingestible::has_processed_all_metadata() const
{
  // Polled by the driver before every claim, so it must never start the scan:
  // a producer that does not exist yet has not consumed its metadata.
  return _impl->producer != nullptr && _impl->producer->eof_claimed();
}

gpu_ingestible::metadata_scan_task_t lance_gpu_ingestible::next_split_provider(
  io::ioctx_resolver /*resolve*/)
{
  // A Lance scan owns no sirius_datasource, so the resolver is unused and the
  // split exposes no fadvise entries — every prefetch site is a no-op.
  //
  // The claim blocks until a batch, an error, or EOF is available. Returning a
  // null callable for "nothing yet" would busy-spin the driver's claim loop.
  auto task = _impl->ensure_producer().claim();
  if (!task) { return task; }

  // A producer failure is repackaged as a poisoned split rather than allowed to
  // fail the metadata task. See lance_scan_info::poison for why: the error has
  // to travel as data so that a scan task exists to rethrow it.
  return [task = std::move(task)]() -> std::unique_ptr<scan_info> {
    try {
      return task();
    } catch (...) {
      auto poisoned    = std::make_unique<lance_scan_info>();
      poisoned->poison = std::current_exception();
      return poisoned;
    }
  };
}

filtered_table lance_gpu_ingestible::materialize_metadata_to_table(
  const scan_info& info,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream)
{
  auto const* split = dynamic_cast<lance_scan_info const*>(&info);
  if (split == nullptr) {
    throw std::invalid_argument("lance_gpu_ingestible: split is not a lance_scan_info");
  }

  // Producer-side failure, delivered as data: rethrow inside the scan task, the
  // one place the driver is built to fail from.
  if (split->poison) { std::rethrow_exception(split->poison); }

  rmm::device_async_resource_ref mr(mem_space.get_default_allocator());
  auto const& expect = _impl->info->expect;

  if (split->is_empty_sentinel) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(_impl->info->output_types.size());
    for (auto const& type : _impl->info->output_types) {
      columns.push_back(cudf::make_empty_column(sirius::get_cudf_type(type)));
    }
    return filtered_table{
      .table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))},
      .state = filter_state::UNFILTERED};
  }

  if (!split->batch.valid()) {
    throw std::invalid_argument("lance_gpu_ingestible: split carries no Arrow batch");
  }

  auto const& schema = split->batch.schema();
  auto const& array  = split->batch.array();
  if (array.offset != 0) {
    throw std::runtime_error("lance_gpu_ingestible: sliced Arrow batches are unsupported");
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  {
    // cudf issues its host-to-device copies on this stream and returns without
    // synchronizing, so the host buffers must outlive them. Synchronizing here
    // covers every exit path, an unwinding one included: whoever frees the
    // batch afterwards — the retirer below, or the split's destructor on the
    // failure path — is then guaranteed to be safe.
    scope_exit sync_guard([stream] { stream.synchronize(); });

    columns.reserve(expect.kept_child_indices.size());
    for (auto const child_index : expect.kept_child_indices) {
      if (static_cast<std::int64_t>(child_index) >= schema.n_children ||
          static_cast<std::int64_t>(child_index) >= array.n_children) {
        throw std::runtime_error("lance_gpu_ingestible: Arrow batch is missing child index " +
                                 std::to_string(child_index));
      }
      // Importing by index is what keeps the vector column off the device: it
      // is present in the batch (the Lance KNN surface has no projection list)
      // but absent from kept_child_indices, so it is never touched here.
      columns.push_back(cudf::from_arrow_column(
        schema.children[child_index], array.children[child_index], stream, mr));
    }
  }

  _impl->retirer->retire(std::move(split->batch), stream);

  return filtered_table{
    .table = owning_table_view{std::make_unique<cudf::table>(std::move(columns))},
    .state = filter_state::UNFILTERED};
}

std::unique_ptr<cudf::table> lance_gpu_ingestible::post_filter_and_project(
  filtered_table&& input,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream)
{
  // Materialization already emitted exactly the projected columns in output
  // order, and v1 registers the table function without filter pushdown, so
  // there is nothing left to apply.
  return std::move(input.table).release(stream, mem_space.get_default_allocator());
}

const ingestible_table_info& lance_gpu_ingestible::table_info() const noexcept
{
  return *_impl->info;
}

std::vector<std::size_t> lance_gpu_ingestible::materialized_column_order() const
{
  return _impl->info->materialized_order;
}

std::shared_ptr<gpu_ingestible> make_ingestible(std::unique_ptr<lance_ingestible_table_info> info)
{
  return std::make_shared<lance_gpu_ingestible>(std::move(info));
}

}  // namespace sirius::op::scan
