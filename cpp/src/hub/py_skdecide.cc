/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pybind11/pybind11.h>

#include "config.h"
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
void init_pyvi(py::module &m);
void init_pypi(py::module &m);
void init_pyldfs(py::module &m);
void init_pyrtdp_bel(py::module &m);
void init_pyssipp(py::module &m);
void init_pyfret(py::module &m);
void init_pygpci(py::module &m);
void init_pysarsop(py::module &m);
void init_pydespot(py::module &m);
void init_pyehc(py::module &m);
void init_pyff(py::module &m);
void init_pyppddlreplan(py::module &m);
void init_pyppddldethindsight(py::module &m);
void init_pysspreplan(py::module &m);
void init_pysspdethindsight(py::module &m);
void init_pysspplanmerger(py::module &m);
void init_pyppddlplanmerger(py::module &m);
void init_pyhsvi(py::module &m);
void init_pypomcp(py::module &m);
#ifdef HAS_HIGHS
void init_pymdplp(py::module &m);
void init_pyidual(py::module &m);
void init_pywitness(py::module &m);
#endif
void init_pypddl(py::module &m);

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
  init_pyvi(m);
  init_pypi(m);
  init_pyldfs(m);
  init_pyrtdp_bel(m);
  init_pyssipp(m);
  init_pyfret(m);
  init_pygpci(m);
  init_pysarsop(m);
  init_pydespot(m);
  init_pyehc(m);
  init_pyff(m);
  init_pyppddlreplan(m);
  init_pyppddldethindsight(m);
  init_pysspreplan(m);
  init_pysspdethindsight(m);
  init_pysspplanmerger(m);
  init_pyppddlplanmerger(m);
  init_pyhsvi(m);
  init_pypomcp(m);
#ifdef HAS_HIGHS
  init_pymdplp(m);
  init_pyidual(m);
  init_pywitness(m);
#endif
  init_pypddl(m);
}
