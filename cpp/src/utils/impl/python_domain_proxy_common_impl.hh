/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_DOMAIN_PROXY_PYOBJ_IMPL_HH
#define SKDECIDE_PYTHON_DOMAIN_PROXY_PYOBJ_IMPL_HH

#include <pybind11/pybind11.h>

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === PyObj implementation ===

template <typename Texecution>
template <typename Derived, typename Tpyobj>
struct PythonDomainProxyBase<Texecution>::PyObj<Derived, Tpyobj>::Implementation {
    
    typedef typename PythonDomainProxyBase<Texecution>::template PyObj<Derived, Tpyobj> PyObj;

    template <typename TTpyobj = Tpyobj,
              std::enable_if_t<std::is_same<TTpyobj, py::object>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po) {
        typename GilControl<Texecution>::Acquire acquire;
        po = std::make_unique<Tpyobj>(py::none());
    }

    template <typename TTpyobj = Tpyobj,
              std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                               std::is_base_of<py::object, TTpyobj>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po) {
        typename GilControl<Texecution>::Acquire acquire;
        po = std::make_unique<Tpyobj>(Tpyobj());
    }

    template <typename TTpyobj = Tpyobj,
              std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                               !std::is_base_of<py::object, TTpyobj>::value &&
                               std::is_base_of<py::detail::pyobject_tag, TTpyobj>::value &&
                               !std::is_default_constructible<TTpyobj>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po) {}

    template <typename TTpyobj = Tpyobj,
              std::enable_if_t<std::is_same<TTpyobj, py::object>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po, std::unique_ptr<py::object>&& o, bool check = true) {
        typename GilControl<Texecution>::Acquire acquire; // !(*po) class python if *po is of type py::bool_
        po = std::move(o);
        if (check && (!po || !(*po))) {
            throw std::runtime_error(std::string("Unitialized python ") + Derived::class_name + " object!");
        }
    }

    template <typename TTpyobj = Tpyobj,
              std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                               std::is_base_of<py::object, TTpyobj>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po, std::unique_ptr<py::object>&& o, bool check = true) {
        typename GilControl<Texecution>::Acquire acquire;
        if (check && (!o || !(*o) || !py::isinstance<Tpyobj>(*o))) {
            throw std::runtime_error(std::string("Python ") + Derived::class_name + " object not initialized as a " +
                                     std::string(py::str(Tpyobj().attr("__class__").attr("__name__"))));
        }
        po = std::make_unique<Tpyobj>(o->template cast<Tpyobj>());
        o.reset();
    }

    template <typename TTpyobj = Tpyobj,
              std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                               std::is_base_of<py::object, TTpyobj>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po, std::unique_ptr<TTpyobj>&& o, bool check = true) {
        typename GilControl<Texecution>::Acquire acquire; // !(*po) class python if *po is of type py::bool_
        po = std::move(o);
        if (check && (!po || !(*po))) {
            throw std::runtime_error(std::string("Unitialized python ") + Derived::class_name + " object!");
        }
    }

    template <typename TTpyobj = Tpyobj,
              std::enable_if_t<!std::is_base_of<py::object, TTpyobj>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po, std::unique_ptr<TTpyobj>&& o, bool check = true) {
        po = std::move(o);
        if (check && !po) {
            throw std::runtime_error(std::string("Unitialized python ") + Derived::class_name + " object!");
        }
    }

    template <typename TTpyobj = Tpyobj,
              std::enable_if_t<std::is_same<TTpyobj, py::object>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po, const py::object& o, bool check = true) {
        typename GilControl<Texecution>::Acquire acquire;
        po = std::make_unique<py::object>(o);
        if (check && !(*po)) {
            throw std::runtime_error(std::string("Unitialized python ") + Derived::class_name + " object!");
        }
    }

    template <typename TTpyobj = Tpyobj,
              std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                               std::is_base_of<py::object, TTpyobj>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po, const py::object& o, bool check = true) {
        if (check && (!o || !py::isinstance<Tpyobj>(o))) {
            throw std::runtime_error(std::string("Python ") + Derived::class_name + " object not initialized as a " +
                                     std::string(py::str(Tpyobj().attr("__class__").attr("__name__"))));
        }
        po = std::make_unique<Tpyobj>(o.template cast<Tpyobj>());
    }

    template <typename TTpyobj,
              std::enable_if_t<!std::is_same<TTpyobj, py::object>::value &&
                               std::is_base_of<py::object, TTpyobj>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po, const TTpyobj& o, bool check = true) {
        typename GilControl<Texecution>::Acquire acquire;
        po = std::make_unique<Tpyobj>(o);
        if (check && !(*po)) {
            throw std::runtime_error(std::string("Unitialized python ") + Derived::class_name + " object!");
        }
    }

    template <typename TTpyobj,
              std::enable_if_t<!std::is_base_of<py::object, TTpyobj>::value &&
                               !std::is_base_of<PyObj, TTpyobj>::value, int> = 0>
    static void construct(std::unique_ptr<Tpyobj>& po, const TTpyobj& o, [[maybe_unused]] bool check = true) {
        typename GilControl<Texecution>::Acquire acquire;
        po = std::make_unique<Tpyobj>(o);
    }
};

#define SK_PYOBJ_TEMPLATE_DECL \
template <typename Texecution> \
template <typename Derived, typename Tpyobj>

#define SK_PY_OBJ_CLASS \
PythonDomainProxyBase<Texecution>::PyObj<Derived, Tpyobj>

#define SK_PY_OBJ_TYPE \
typename PythonDomainProxyBase<Texecution>::template PyObj<Derived, Tpyobj>

SK_PYOBJ_TEMPLATE_DECL
SK_PY_OBJ_CLASS::PyObj() {
    Implementation::construct(_pyobj);
}

SK_PYOBJ_TEMPLATE_DECL
template <typename TTpyobj>
void SK_PY_OBJ_CLASS::move_construct(std::unique_ptr<TTpyobj>&& o, bool check) {
    Implementation::construct(_pyobj, std::move(o), check);
}

SK_PYOBJ_TEMPLATE_DECL
template <typename TTpyobj>
void SK_PY_OBJ_CLASS::conv_construct(const TTpyobj& o, bool check) {
    Implementation::construct(_pyobj, o, check);
}

SK_PYOBJ_TEMPLATE_DECL
SK_PY_OBJ_CLASS::PyObj(const PyObj& other) {
    typename GilControl<Texecution>::Acquire acquire;
    this->_pyobj = std::make_unique<Tpyobj>(*other._pyobj);
}

SK_PYOBJ_TEMPLATE_DECL
SK_PY_OBJ_TYPE& SK_PY_OBJ_CLASS::operator=(const PyObj& other) {
    typename GilControl<Texecution>::Acquire acquire;
    *(this->_pyobj) = *other._pyobj;
    return *this;
}

SK_PYOBJ_TEMPLATE_DECL
SK_PY_OBJ_CLASS::~PyObj() {
    typename GilControl<Texecution>::Acquire acquire;
    _pyobj.reset();
}

SK_PYOBJ_TEMPLATE_DECL
const Tpyobj& SK_PY_OBJ_CLASS::pyobj() const {
    return *_pyobj;
}

SK_PYOBJ_TEMPLATE_DECL
std::string SK_PY_OBJ_CLASS::print() const {
    typename GilControl<Texecution>::Acquire acquire;
    return py::str(*_pyobj);
}

SK_PYOBJ_TEMPLATE_DECL
std::size_t SK_PY_OBJ_CLASS::Hash::operator()(const PyObj<Derived, Tpyobj>& o) const {
    try {
        return skdecide::PythonHash<Texecution>()(*o._pyobj);
    } catch(const std::exception& e) {
        Logger::error(std::string("SKDECIDE exception when hashing ") +
                      Derived::class_name + "s: " + std::string(e.what()));
        throw;
    }
}

SK_PYOBJ_TEMPLATE_DECL
bool SK_PY_OBJ_CLASS::Equal::operator()(const PyObj<Derived, Tpyobj>& o1, const PyObj<Derived, Tpyobj>& o2) const {
    try {
        return skdecide::PythonEqual<Texecution>()(*o1._pyobj, *o2._pyobj);
    } catch(const std::exception& e) {
        Logger::error(std::string("SKDECIDE exception when testing ") +
                      Derived::class_name + "s equality: " + std::string(e.what()));
        throw;
    }
}

// === PyIter implementation ===

template <typename Texecution>
template<typename T, typename Titerator>
struct PythonDomainProxyBase<Texecution>::PyIter<T, Titerator>::Implementation {
    template <typename TTiterator = Titerator,
              std::enable_if_t<!std::is_same<TTiterator, py::detail::dict_iterator>::value, int> = 0>
    static T dereference_object(Titerator& pit) {
        typename GilControl<Texecution>::Acquire acquire;
        return T(py::reinterpret_borrow<py::object>(*pit));
    }

    template <typename TTiterator = Titerator,
              std::enable_if_t<std::is_same<TTiterator, py::detail::dict_iterator>::value, int> = 0>
    static T dereference_object(Titerator& pit) {
        typename GilControl<Texecution>::Acquire acquire;
        return T(py::make_tuple(py::reinterpret_borrow<py::object>(pit->first),
                                py::reinterpret_borrow<py::object>(pit->second)));
    }

    template <typename TTiterator = Titerator,
              std::enable_if_t<!std::is_same<TTiterator, py::detail::dict_iterator>::value, int> = 0>
    static std::unique_ptr<T> dereference_pointer(Titerator& pit) {
        typename GilControl<Texecution>::Acquire acquire;
        return std::make_unique<T>(py::reinterpret_borrow<py::object>(*pit));
    }

    template <typename TTiterator = Titerator,
              std::enable_if_t<std::is_same<TTiterator, py::detail::dict_iterator>::value, int> = 0>
    static std::unique_ptr<T> dereference_pointer(Titerator& pit) {
        typename GilControl<Texecution>::Acquire acquire;
        return std::make_unique<T>(py::make_tuple(py::reinterpret_borrow<py::object>(pit->first),
                                                  py::reinterpret_borrow<py::object>(pit->second)));
    }
};

#define SK_PY_ITER_TEMPLATE_DECL \
template <typename Texecution> \
template <typename T, typename Titerator>

#define SK_PY_ITER_CLASS \
PythonDomainProxyBase<Texecution>::PyIter<T, Titerator>

#define SK_PY_ITER_TYPE \
typename PythonDomainProxyBase<Texecution>::template PyIter<T, Titerator>

SK_PY_ITER_TEMPLATE_DECL
SK_PY_ITER_CLASS::PyIter(const Titerator& iterator)
: PyObj<PyIter<T, Titerator>, Titerator>(iterator, false) {}

SK_PY_ITER_TEMPLATE_DECL
SK_PY_ITER_CLASS::PyIter(const PyIter<T, Titerator>& other)
: PyObj<PyIter<T, Titerator>, Titerator>(other) {}

SK_PY_ITER_TEMPLATE_DECL
SK_PY_ITER_TYPE& SK_PY_ITER_CLASS::operator=(const PyIter<T, Titerator>& other) {
    static_cast<PyObj<PyIter<T, Titerator>, Titerator>&>(*this) = other;
    return *this;
}

SK_PY_ITER_TEMPLATE_DECL
SK_PY_ITER_CLASS::~PyIter() {}

SK_PY_ITER_TEMPLATE_DECL
SK_PY_ITER_TYPE& SK_PY_ITER_CLASS::operator++() {
    typename GilControl<Texecution>::Acquire acquire;
    ++(*(this->_pyobj));
    return *this;
}

SK_PY_ITER_TEMPLATE_DECL
SK_PY_ITER_TYPE SK_PY_ITER_CLASS::operator++(int) {
    typename GilControl<Texecution>::Acquire acquire;
    Titerator rv = (*(this->_pyobj))++;
    return PyIter<T, Titerator>(rv);
}

SK_PY_ITER_TEMPLATE_DECL
T SK_PY_ITER_CLASS::operator*() const {
    return SK_PY_ITER_CLASS::Implementation::dereference_object(*(this->_pyobj));
}

SK_PY_ITER_TEMPLATE_DECL
std::unique_ptr<T> SK_PY_ITER_CLASS::operator->() const {
    return SK_PY_ITER_CLASS::Implementation::dereference_pointer(*(this->_pyobj));
}

SK_PY_ITER_TEMPLATE_DECL
bool SK_PY_ITER_CLASS::operator==(const PyIter<T, Titerator>& other) const {
    typename GilControl<Texecution>::Acquire acquire;
    return *(this->_pyobj) == *(other._pyobj);
}

SK_PY_ITER_TEMPLATE_DECL
bool SK_PY_ITER_CLASS::operator!=(const PyIter<T, Titerator>& other) const {
    typename GilControl<Texecution>::Acquire acquire;
    return *(this->_pyobj) != *(other._pyobj);
}

} // namespace skdecide

#endif // PYTHON_DOMAIN_PROXY_PYOBJ_IMPL_HH
