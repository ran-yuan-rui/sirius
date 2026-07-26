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

#pragma once

// sirius — pulls in the Arrow C data interface (see the include-order note there)
#include <lance_shim/lance_ffi_api.hpp>
#include <op/scan/lance/lance_split_producer.hpp>

// duckdb
#include <duckdb/function/function.hpp>
#include <duckdb/function/table_function.hpp>

// standard library
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace sirius::lance {

/// SQL-visible name of the Lance vector-search table function. The planner
/// matches on this string, and the no-fallback plan walk keys off it.
inline constexpr char const* kLanceVectorSearchName = "sirius_lance_vector_search";

//===----------------------------------------------------------------------===//
// lance_vector_search_bind_data
//===----------------------------------------------------------------------===//
/**
 * @brief Everything bind resolved, carried to the planner and then the scan.
 *
 * Bind opens the dataset only long enough to read the KNN schema, validates it,
 * and closes it again; the scan reopens at execution time so a re-executed
 * prepared statement sees the dataset's then-current version.
 *
 * The FFI handle is captured here as a shared_ptr, which is what lets a plan
 * bound under a test override keep using that override after the guard is gone.
 */
struct lance_vector_search_bind_data : public duckdb::TableFunctionData {
  std::shared_ptr<lance_ffi_api> api;
  lance_open_spec spec;
  schema_expectation expect;

  /// Output columns in emission order, `_distance` included, vector excluded.
  std::vector<std::string> names;
  std::vector<duckdb::LogicalType> return_types;

  [[nodiscard]] duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    auto copy          = duckdb::make_uniq<lance_vector_search_bind_data>();
    copy->api          = api;
    copy->spec         = spec;
    copy->expect       = expect;
    copy->names        = names;
    copy->return_types = return_types;
    return copy;
  }

  [[nodiscard]] bool Equals(duckdb::FunctionData const& other_p) const override
  {
    auto const& other = other_p.Cast<lance_vector_search_bind_data>();
    return api == other.api && spec.uri == other.spec.uri &&
           spec.knn.vector_column == other.spec.knn.vector_column &&
           spec.knn.query == other.spec.knn.query && spec.knn.k == other.spec.knn.k &&
           names == other.names;
  }
};

}  // namespace sirius::lance
