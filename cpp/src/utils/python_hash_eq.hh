/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PYTHON_HASH_EQ_HH
#define AIRLAPS_PYTHON_HASH_EQ_HH

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/python_gil_control.hh"
#include "utils/python_container_adapter.hh"

namespace py = pybind11;

namespace airlaps {

template <typename Texecution> class PythonContainerAdapter;

template <typename Texecution>
struct PythonHash {
    std::size_t operator()(const py::object& o) const {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            std::size_t python_sys_maxsize = py::module::import("sys").attr("maxsize").template cast<std::size_t>();
            std::function<std::size_t (const py::object&)> compute_hash = [&python_sys_maxsize](const py::object& ho) {
                // python __hash__ can return negative integers but c++ expects positive integers only
                // return  (ho.attr("__hash__")().template cast<std::size_t>()) % ((python_sys_maxsize + 1) * 2);
                std::size_t res = (ho.attr("__hash__")().template cast<long long>()) + python_sys_maxsize;
                return res;
            };
            if (py::isinstance<py::array>(o)) {
                return PythonContainerAdapter<Texecution>(o).hash();
            } else if (!py::hasattr(o, "__hash__") || o.attr("__hash__").is_none()) {
                // Try to hash using __repr__
                py::object r = o.attr("__repr__")();
                if (!py::hasattr(r, "__hash__") || r.attr("__hash__").is_none()) {
                    // Try to hash using __str__
                    py::object s = o.attr("__str__")();
                    if (!py::hasattr(s, "__hash__") || s.attr("__hash__").is_none()) {
                        // Desperate case...
                        throw std::invalid_argument("AIRLAPS exception: python object does not provide usable __hash__ nor hashable __repr__ or __str__");
                    } else {
                        return compute_hash(s);
                    }
                } else {
                    return compute_hash(r);
                }
            } else {
                return compute_hash(o);
            }
        } catch(const py::error_already_set* e) {
            spdlog::error(std::string("AIRLAPS exception when hashing python object: ") + e->what());
            std::runtime_error err(e->what());
            delete e;
            throw err;
        }
    }
};


template <typename Texecution>
struct PythonEqual {
    bool operator()(const py::object& o1, const py::object& o2) const {
        typename GilControl<Texecution>::Acquire acquire;
        try {
            std::function<bool (const py::object&, const py::object&)> compute_equal = [](const py::object& eo1, const py::object& eo2) {
                return eo1.attr("__eq__")(eo2).template cast<bool>();
            };
            if (py::isinstance<py::array>(o1) && py::isinstance<py::array>(o2)) {
                return py::module::import("numpy").attr("array_equal")(o1, o2).template cast<bool>();
            } else if (!py::hasattr(o1, "__eq__") || o1.attr("__eq__").is_none() ||
                    !py::hasattr(o2, "__eq__") || o2.attr("__eq__").is_none()) {
                // Try to equalize using __repr__
                py::object r1 = o1.attr("__repr__")();
                py::object r2 = o2.attr("__repr__")();
                if (!py::hasattr(r1, "__eq__") || r1.attr("__eq__").is_none() ||
                    !py::hasattr(r2, "__eq__") || r2.attr("__eq__").is_none()) {
                    // Try to equalize using __str__
                    py::object s1 = o1.attr("__str__")();
                    py::object s2 = o2.attr("__str__")();
                    if (!py::hasattr(s1, "__eq__") || s1.attr("__eq__").is_none() ||
                        !py::hasattr(s2, "__eq__") || s2.attr("__eq__").is_none()) {
                        // Desperate case...
                        throw std::invalid_argument("AIRLAPS exception: python objects do not provide usable __eq__ nor equal tests using __repr__ or __str__");
                    } else {
                        return compute_equal(s1, s2);
                    }
                } else {
                    return compute_equal(r1, r2);
                }
            } else {
                return compute_equal(o1, o2);
            }
        } catch(const py::error_already_set* e) {
            spdlog::error(std::string("AIRLAPS exception when testing equality of python objects: ") + e->what());
            std::runtime_error err(e->what());
            delete e;
            throw err;
        }
    }
};

} // namespace airlaps

#endif // AIRLAPS_PYTHON_HASH_EQ_HH
