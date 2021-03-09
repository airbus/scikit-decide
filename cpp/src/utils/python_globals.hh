/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_GLOBALS_HH
#define SKDECIDE_PYTHON_GLOBALS_HH

#include <pybind11/pybind11.h>

#include "utils/logging.hh"

namespace py = pybind11;

namespace skdecide {

// This class enforces that python global data are read just after importing the module
// using them (i.e. in PYBIND11_MODULE). This is to prevent reading them on the fly
// within a parallel thread, which is not supported by pybind11 (acquiring the GIL before
// reading global python objects has no effect and results in crashing the interpreter).
class Globals {
public :
    static void init() {
        if (!_initialized) {
            Logger::set_level(logging::debug);
            _not_implemented_object = py::globals()["__builtins__"]["NotImplemented"];
            _sorted = py::globals()["__builtins__"]["sorted"];
            _python_sys_maxsize = py::module::import("sys").attr("maxsize").template cast<std::size_t>();
            _skdecide = py::module::import("skdecide");
            _initialized = true;
        }
    }

    static py::object not_implemented_object() {
        check_initialized();
        return py::reinterpret_borrow<py::object>(_not_implemented_object);
    }

    static py::object sorted() {
        check_initialized();
        return py::reinterpret_borrow<py::object>(_sorted);
    }

    static const std::size_t& python_sys_maxsize() {
        check_initialized();
        return _python_sys_maxsize;
    }

    static py::object skdecide() {
        check_initialized();
        return py::reinterpret_borrow<py::object>(_skdecide);
    }

private :
    // Initializing python objects here (in their declaration) by calling
    // python functions crashes the python interpreter (unable to load the
    // C++ extension library) on some platforms and some version of Python.
    // Thus we enforce calling the Globals::init() method in PYBIND11_MODULE.
    // Also, we don't declare python objects as py::object because their
    // reference count is managed outside pybind11 so that we don't know
    // when those python object pointers are no more valid.
    inline static py::handle _not_implemented_object = py::handle();
    inline static py::handle _sorted = py::handle();
    inline static std::size_t _python_sys_maxsize = 0;
    inline static py::handle _skdecide = py::handle();
    inline static bool _initialized = false;

    static void check_initialized() {
        if (!_initialized) {
            throw std::runtime_error("Python globals not properly initialized. Call skdecide::Globals::init() in the PYBIND11_MODULE() macro.");
        }
    }
};

} // namespace skdecide

#endif // SKDECIDE_PYTHON_GLOBALS_HH
