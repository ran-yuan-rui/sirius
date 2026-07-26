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

#include <op/scan/lance/lance_arrow_batch.hpp>

#include <utility>

namespace sirius::lance {

namespace {

/// Steals the callbacks and payload, leaving the source inert so a subsequent
/// release on it is a no-op. Arrow's C ABI has no move concept, so ownership is
/// expressed by who still holds a non-null release callback.
template <typename ArrowStruct>
void steal(ArrowStruct& dst, ArrowStruct& src) noexcept
{
  dst = src;
  src = ArrowStruct{};
}

}  // namespace

lance_owning_arrow_batch::lance_owning_arrow_batch(ArrowArray&& array,
                                                   ArrowSchema&& schema) noexcept
{
  steal(_array, array);
  steal(_schema, schema);
}

lance_owning_arrow_batch::~lance_owning_arrow_batch() { release(); }

lance_owning_arrow_batch::lance_owning_arrow_batch(lance_owning_arrow_batch&& other) noexcept
{
  steal(_array, other._array);
  steal(_schema, other._schema);
}

lance_owning_arrow_batch& lance_owning_arrow_batch::operator=(
  lance_owning_arrow_batch&& other) noexcept
{
  if (this != &other) {
    release();
    steal(_array, other._array);
    steal(_schema, other._schema);
  }
  return *this;
}

bool lance_owning_arrow_batch::valid() const noexcept { return _array.release != nullptr; }

std::int64_t lance_owning_arrow_batch::num_rows() const noexcept
{
  return valid() ? _array.length : 0;
}

void lance_owning_arrow_batch::release() noexcept
{
  // Schema first: it describes the array, so releasing it last would briefly
  // leave a described-by-nothing array visible to anything inspecting the pair.
  //
  // The callback is invoked with `release` still set: the Arrow C data
  // interface makes the callee responsible for clearing it, and producers rely
  // on seeing a live pointer to recognise a genuine release. Zeroing the struct
  // afterwards is what makes a second call a no-op.
  if (_schema.release != nullptr) {
    _schema.release(&_schema);
    _schema = ArrowSchema{};
  }
  if (_array.release != nullptr) {
    _array.release(&_array);
    _array = ArrowArray{};
  }
}

namespace {

/// v1 retirement policy: block until the importing stream has run every copy
/// that reads these host buffers, then free them.
///
/// cudf's host-to-device copies are issued on the caller's stream and are not
/// synchronized before `from_arrow_*` returns, so releasing earlier would free
/// memory the device may still be reading. Synchronizing is heavy-handed but
/// unconditionally correct; the seam exists so an event-based policy can take
/// over without touching any caller.
class synchronizing_retirer final : public arrow_batch_retirer {
 public:
  void retire(lance_owning_arrow_batch&& batch, rmm::cuda_stream_view stream) override
  {
    // Take ownership first so the batch is released even if the sync throws.
    lance_owning_arrow_batch owned(std::move(batch));
    stream.synchronize();
    owned.release();
  }
};

}  // namespace

std::unique_ptr<arrow_batch_retirer> make_synchronizing_retirer()
{
  return std::make_unique<synchronizing_retirer>();
}

}  // namespace sirius::lance
