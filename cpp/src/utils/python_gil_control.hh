/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_GIL_CONTROL_HH
#define SKDECIDE_PYTHON_GIL_CONTROL_HH

#include <pybind11/pybind11.h>

#include "utils/execution.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution> struct GilControl;

template <>
struct GilControl<skdecide::SequentialExecution> {
    struct Acquire { Acquire() {} };
    struct Release { Release() {} };
};

template <>
struct GilControl<skdecide::ParallelExecution> {
    typedef py::gil_scoped_acquire Acquire;
    typedef py::gil_scoped_release Release;
};

} // namespace skdecide

#endif // SKDECIDE_PYTHON_GIL_CONTROL_HH
