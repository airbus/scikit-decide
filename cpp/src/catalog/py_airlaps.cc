#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_pyaostar(py::module& m);
void init_pyiw(py::module& m);
void init_pypddl(py::module& m);

PYBIND11_MODULE(__airlaps_catalog_cpp, m) {
    init_pyaostar(m);
    init_pyiw(m);
    init_pypddl(m);
}
