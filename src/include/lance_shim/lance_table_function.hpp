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

// duckdb
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_transaction.hpp>
#include <duckdb/function/table_function.hpp>

namespace sirius::lance {

/// Bind callback: validates options, reads the KNN schema, and produces the
/// output layout. Throws @c duckdb::BinderException for anything it refuses.
duckdb::unique_ptr<duckdb::FunctionData> lance_vector_search_bind(
  duckdb::ClientContext& context,
  duckdb::TableFunctionBindInput& input,
  duckdb::vector<duckdb::LogicalType>& return_types,
  duckdb::vector<std::string>& names);

/// Registers the table function. Purely additive to the catalog.
void register_lance_vector_search(duckdb::Catalog& catalog, duckdb::CatalogTransaction transaction);

}  // namespace sirius::lance
