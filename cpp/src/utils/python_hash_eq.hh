/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_HASH_EQ_HH
#define SKDECIDE_PYTHON_HASH_EQ_HH

#include <cstddef>

namespace pybind11 {
class object;
}

namespace py = pybind11;

namespace skdecide {

template <typename Texecution> struct PythonEqual {
  bool operator()(const py::object &o1, const py::object &o2) const;
};

template <typename Texecution> struct PythonHash {
  std::size_t operator()(const py::object &o) const;

  struct ItemHasher {
    const py::object &_pyobj;

    ItemHasher(const py::object &o);
    std::size_t hash() const;
  };
};

struct SequentialExecution;
struct ParallelExecution;
std::size_t hash_value(const PythonHash<SequentialExecution>::ItemHasher &ih);
std::size_t hash_value(const PythonHash<ParallelExecution>::ItemHasher &ih);

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/python_hash_eq_impl.hh"
#endif

#endif // SKDECIDE_PYTHON_HASH_EQ_HH
