/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>

#include "utils/python_globals.hh"

namespace py = pybind11;

void init_pyaostar(py::module &m);
void init_pyastar(py::module &m);
void init_pybfws(py::module &m);
void init_pyilaostar(py::module &m);
void init_pyiw(py::module &m);
void init_pylrtdp(py::module &m);
void init_pymartdp(py::module &m);
void init_pymcts(py::module &m);
void init_pyriw(py::module &m);

PYBIND11_MODULE(__skdecide_hub_cpp, m) {
  skdecide::Globals::init();
  init_pyaostar(m);
  init_pyastar(m);
  init_pybfws(m);
  init_pyilaostar(m);
  init_pyiw(m);
  init_pylrtdp(m);
  init_pymartdp(m);
  init_pymcts(m);
  init_pyriw(m);
}
