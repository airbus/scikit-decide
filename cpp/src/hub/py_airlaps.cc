#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_pyastar(py::module& m);
void init_pyaostar(py::module& m);
void init_pyiw(py::module& m);
void init_pyriw(py::module& m);
void init_pybfws(py::module& m);
void init_pypddl(py::module& m);

PYBIND11_MODULE(__airlaps_hub_cpp, m) {
    init_pyastar(m);
    init_pyaostar(m);
    init_pyiw(m);
    init_pyriw(m);
    init_pybfws(m);
    init_pypddl(m);
}
