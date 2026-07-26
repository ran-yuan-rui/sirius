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

#include <lance_shim/lance_ffi_api.hpp>

#include <mutex>
#include <stdexcept>
#include <utility>

namespace sirius::lance {

namespace {

std::mutex& factory_mutex()
{
  static std::mutex mutex;
  return mutex;
}

/// The active factory. Empty means "no implementation": either the build has no
/// embedded Lance FFI, or no test override is installed.
lance_ffi_api_factory& active_factory()
{
  static lance_ffi_api_factory factory;
  return factory;
}

#ifdef SIRIUS_HAVE_LANCE_KNN
/// Provided by the gated translation unit that wraps the pinned Rust staticlib.
std::shared_ptr<lance_ffi_api> make_embedded_lance_ffi_api();
#endif

std::shared_ptr<lance_ffi_api> make_default_api()
{
#ifdef SIRIUS_HAVE_LANCE_KNN
  return make_embedded_lance_ffi_api();
#else
  return nullptr;
#endif
}

}  // namespace

scoped_ffi_api_override::scoped_ffi_api_override(lance_ffi_api_factory factory)
{
  std::lock_guard lock(factory_mutex());
  _previous        = std::move(active_factory());
  active_factory() = std::move(factory);
}

scoped_ffi_api_override::~scoped_ffi_api_override()
{
  std::lock_guard lock(factory_mutex());
  active_factory() = std::move(_previous);
}

std::shared_ptr<lance_ffi_api> acquire_lance_ffi_api()
{
  lance_ffi_api_factory factory;
  {
    std::lock_guard lock(factory_mutex());
    factory = active_factory();
  }
  // The factory runs outside the lock: it may block (the embedded
  // implementation starts its control thread on first use) and must not be able
  // to deadlock against another thread installing an override.
  auto api = factory ? factory() : make_default_api();
  if (!api) {
    throw std::runtime_error(
      "Lance vector search is unavailable: this build was configured without "
      "SIRIUS_ENABLE_LANCE_KNN and no Lance FFI override is installed");
  }
  return api;
}

bool lance_ffi_api_available() noexcept
{
  {
    std::lock_guard lock(factory_mutex());
    if (active_factory()) { return true; }
  }
#ifdef SIRIUS_HAVE_LANCE_KNN
  return true;
#else
  return false;
#endif
}

}  // namespace sirius::lance
