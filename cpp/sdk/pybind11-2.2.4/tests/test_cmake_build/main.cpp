/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(test_cmake_build, m) {
    m.def("add", [](int i, int j) { return i + j; });
}
