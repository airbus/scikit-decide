/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_CONTAINER_PROXY_IMPL_HH
#define SKDECIDE_PYTHON_CONTAINER_PROXY_IMPL_HH

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <boost/container_hash/hash.hpp>

#include "utils/python_gil_control.hh"
#include "utils/python_hash_eq.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace py = pybind11;

namespace skdecide {

// === PythonContainerProxy::value_type implementation ===

template <typename Texecution>
PythonContainerProxy<Texecution>::value_type::value_type() {}

template <typename Texecution>
PythonContainerProxy<Texecution>::value_type::value_type(
    const py::object &value) {
  _value = std::make_unique<ObjectType>(value);
}

template <typename Texecution>
PythonContainerProxy<Texecution>::value_type::value_type(
    const value_type &other) {
  other._value->copy(_value);
}

template <typename Texecution>
typename PythonContainerProxy<Texecution>::value_type &
PythonContainerProxy<Texecution>::value_type::operator=(
    const value_type &other) {
  other._value->copy(_value);
  return *this;
}

template <typename Texecution>
std::size_t PythonContainerProxy<Texecution>::value_type::hash() const {
  return _value->hash();
}

template <typename Texecution>
bool PythonContainerProxy<Texecution>::value_type::operator==(
    const value_type &other) const {
  return _value->equal(*(other._value));
}

// === PythonContainerProxy::value_type::PrimitiveType implementation ===

template <typename Texecution>
template <typename T>
PythonContainerProxy<Texecution>::value_type::PrimitiveType<T>::PrimitiveType(
    const T &value)
    : _value(value) {}

template <typename Texecution>
template <typename T>
PythonContainerProxy<Texecution>::value_type::PrimitiveType<
    T>::~PrimitiveType() {}

template <typename Texecution>
template <typename T>
void PythonContainerProxy<Texecution>::value_type::PrimitiveType<T>::copy(
    std::unique_ptr<BaseType> &other) const {
  other = std::make_unique<PrimitiveType<T>>(_value);
}

template <typename Texecution>
template <typename T>
std::size_t
PythonContainerProxy<Texecution>::value_type::PrimitiveType<T>::hash() const {
  return boost::hash_value(_value);
}

template <typename Texecution>
template <typename T>
bool PythonContainerProxy<Texecution>::value_type::PrimitiveType<T>::equal(
    const BaseType &other) const {
  const PrimitiveType<T> *o = dynamic_cast<const PrimitiveType<T> *>(&other);
  return ((o != nullptr) && (o->_value == _value));
}

// === PythonContainerProxy::value_type::ObjectType implementation ===

template <typename Texecution>
PythonContainerProxy<Texecution>::value_type::ObjectType::ObjectType(
    const py::object &value) {
  typename GilControl<Texecution>::Acquire acquire;
  this->_value = std::make_unique<py::object>(value);
}

template <typename Texecution>
PythonContainerProxy<Texecution>::value_type::ObjectType::ObjectType(
    const ObjectType &other) {
  typename GilControl<Texecution>::Acquire acquire;
  this->_value = std::make_unique<py::object>(*other._value);
}

template <typename Texecution>
typename PythonContainerProxy<Texecution>::value_type::ObjectType &
PythonContainerProxy<Texecution>::value_type::ObjectType::operator=(
    const ObjectType &other) {
  typename GilControl<Texecution>::Acquire acquire;
  this->_value = std::make_unique<py::object>(*other._value);
  return *this;
}

template <typename Texecution>
PythonContainerProxy<Texecution>::value_type::ObjectType::~ObjectType() {
  typename GilControl<Texecution>::Acquire acquire;
  _value.reset();
}

template <typename Texecution>
void PythonContainerProxy<Texecution>::value_type::ObjectType::copy(
    std::unique_ptr<BaseType> &other) const {
  other = std::make_unique<ObjectType>(*_value);
}

template <typename Texecution>
std::size_t
PythonContainerProxy<Texecution>::value_type::ObjectType::hash() const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    return skdecide::PythonHash<Texecution>()(*_value);
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string("SKDECIDE exception when hashing container item: ") +
        e->what());
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

template <typename Texecution>
bool PythonContainerProxy<Texecution>::value_type::ObjectType::equal(
    const BaseType &other) const {
  typename GilControl<Texecution>::Acquire acquire;
  try {
    const ObjectType *o = dynamic_cast<const ObjectType *>(&other);
    return ((o != nullptr) &&
            skdecide::PythonEqual<Texecution>()(*_value, *(o->_value)));
  } catch (const py::error_already_set *e) {
    Logger::error(
        std::string(
            "SKDECIDE exception when testing container item equality: ") +
        e->what());
    std::runtime_error err(e->what());
    delete e;
    throw err;
  }
}

// === PythonContainerProxy implementation ===

template <typename Texecution>
PythonContainerProxy<Texecution>::PythonContainerProxy() {}

template <typename Texecution>
PythonContainerProxy<Texecution>::PythonContainerProxy(
    const py::object &vector) {
  typename GilControl<Texecution>::Acquire acquire;
  if (py::isinstance<py::list>(vector)) {
    _implementation =
        std::make_unique<SequenceImplementation<py::list>>(vector);
  } else if (py::isinstance<py::tuple>(vector)) {
    _implementation =
        std::make_unique<SequenceImplementation<py::tuple>>(vector);
  } else if (py::isinstance<py::array>(vector)) {
    std::string dtype = py::str(vector.attr("dtype"));
    if (dtype == "float_") {
      _implementation = std::make_unique<NumpyImplementation<double>>(vector);
    } else if (dtype == "float32") {
      _implementation = std::make_unique<NumpyImplementation<float>>(vector);
    } else if (dtype == "float64") {
      _implementation = std::make_unique<NumpyImplementation<double>>(vector);
    } else if (dtype == "bool_") {
      _implementation = std::make_unique<NumpyImplementation<bool>>(vector);
    } else if (dtype == "int_") {
      _implementation = std::make_unique<NumpyImplementation<long int>>(vector);
    } else if (dtype == "intc") {
      _implementation = std::make_unique<NumpyImplementation<int>>(vector);
    } else if (dtype == "intp") {
      _implementation =
          std::make_unique<NumpyImplementation<std::size_t>>(vector);
    } else if (dtype == "int8") {
      _implementation =
          std::make_unique<NumpyImplementation<std::int8_t>>(vector);
    } else if (dtype == "int16") {
      _implementation =
          std::make_unique<NumpyImplementation<std::int16_t>>(vector);
    } else if (dtype == "int32") {
      _implementation =
          std::make_unique<NumpyImplementation<std::int32_t>>(vector);
    } else if (dtype == "int64") {
      _implementation =
          std::make_unique<NumpyImplementation<std::int64_t>>(vector);
    } else if (dtype == "uint8") {
      _implementation =
          std::make_unique<NumpyImplementation<std::uint8_t>>(vector);
    } else if (dtype == "uint16") {
      _implementation =
          std::make_unique<NumpyImplementation<std::uint16_t>>(vector);
    } else if (dtype == "uint32") {
      _implementation =
          std::make_unique<NumpyImplementation<std::uint32_t>>(vector);
    } else if (dtype == "uint64") {
      _implementation =
          std::make_unique<NumpyImplementation<std::uint64_t>>(vector);
    } else {
      Logger::error("Unhandled array dtype '" + dtype +
                    "' when parsing python sequence as numpy array");
      throw std::invalid_argument(
          "SKDECIDE exception: Unhandled array dtype '" + dtype +
          "' when parsing container as numpy array");
    }
  } else {
    Logger::error(
        "Unhandled container type '" +
        std::string(py::str(vector.attr("__class__").attr("__name__"))) +
        " (expecting list, tuple or numpy array)");
    throw std::invalid_argument(
        "Unhandled container type '" +
        std::string(py::str(vector.attr("__class__").attr("__name__"))) +
        " (expecting list, tuple or numpy array)");
  }
}

template <typename Texecution>
std::size_t PythonContainerProxy<Texecution>::size() const {
  return _implementation->size();
}

template <typename Texecution>
typename PythonContainerProxy<Texecution>::value_type
PythonContainerProxy<Texecution>::operator[](std::size_t index) const {
  return _implementation->at(index);
}

template <typename Texecution>
std::size_t PythonContainerProxy<Texecution>::hash() const {
  return _implementation->hash();
}

template <typename Texecution>
bool PythonContainerProxy<Texecution>::operator==(
    const PythonContainerProxy &other) const {
  return _implementation->equal(*(other._implementation));
}

// === PythonContainerProxy::BaseImplementation implementation ===

template <typename Texecution>
std::size_t PythonContainerProxy<Texecution>::BaseImplementation::hash() const {
  std::size_t seed = 0;
  for (std::size_t i = 0; i < size(); i++) {
    boost::hash_combine(seed, at(i));
  }
  return seed;
}

template <typename Texecution>
bool PythonContainerProxy<Texecution>::BaseImplementation::equal(
    const BaseImplementation &other) const {
  if (this->same_type(other) && (this->size() == other.size())) {
    for (std::size_t i = 0; i < this->size(); i++) {
      if (!(this->at(i) == other.at(i))) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

// === PythonContainerProxy::SequenceImplementation implementation ===

template <typename Texecution>
template <typename Tsequence>
PythonContainerProxy<Texecution>::SequenceImplementation<
    Tsequence>::SequenceImplementation(const py::object &vector) {
  typename GilControl<Texecution>::Acquire acquire;
  this->_vector = std::make_unique<Tsequence>(vector);
}

template <typename Texecution>
template <typename Tsequence>
PythonContainerProxy<Texecution>::SequenceImplementation<
    Tsequence>::SequenceImplementation(const SequenceImplementation &other) {
  typename GilControl<Texecution>::Acquire acquire;
  this->_vector = std::make_unique<Tsequence>(*other._vector);
}

template <typename Texecution>
template <typename Tsequence>
typename PythonContainerProxy<Texecution>::template SequenceImplementation<
    Tsequence> &
PythonContainerProxy<Texecution>::SequenceImplementation<Tsequence>::operator=(
    const SequenceImplementation &other) {
  typename GilControl<Texecution>::Acquire acquire;
  this->_vector = std::make_unique<Tsequence>(*other._vector);
  return *this;
}

template <typename Texecution>
template <typename Tsequence>
PythonContainerProxy<Texecution>::SequenceImplementation<
    Tsequence>::~SequenceImplementation() {
  typename GilControl<Texecution>::Acquire acquire;
  _vector.reset();
}

template <typename Texecution>
template <typename Tsequence>
std::size_t
PythonContainerProxy<Texecution>::SequenceImplementation<Tsequence>::size()
    const {
  typename GilControl<Texecution>::Acquire acquire;
  return _vector->size();
}

template <typename Texecution>
template <typename Tsequence>
typename PythonContainerProxy<Texecution>::value_type
PythonContainerProxy<Texecution>::SequenceImplementation<Tsequence>::at(
    std::size_t index) const {
  typename GilControl<Texecution>::Acquire acquire;
  return value_type((*_vector)[index]);
}

template <typename Texecution>
template <typename Tsequence>
bool PythonContainerProxy<Texecution>::SequenceImplementation<
    Tsequence>::same_type(const BaseImplementation &other) const {
  return dynamic_cast<const SequenceImplementation<Tsequence> *>(&other) !=
         nullptr;
}

// === PythonContainerProxy::NumpyImplementation implementation ===

template <typename Texecution>
template <typename T>
PythonContainerProxy<Texecution>::NumpyImplementation<T>::NumpyImplementation(
    const py::object &vector) {
  typename GilControl<Texecution>::Acquire acquire;
  _vector = std::make_unique<py::array_t<T>>(vector);
  _buffer = std::make_unique<py::buffer_info>(_vector->request());
}

template <typename Texecution>
template <typename T>
PythonContainerProxy<Texecution>::NumpyImplementation<T>::NumpyImplementation(
    const NumpyImplementation &other) {
  typename GilControl<Texecution>::Acquire acquire;
  this->_vector = std::make_unique<py::array_t<T>>(*other._vector);
  this->_buffer = std::make_unique<py::buffer_info>(*other._buffer);
}

template <typename Texecution>
template <typename T>
typename PythonContainerProxy<Texecution>::template NumpyImplementation<T> &
PythonContainerProxy<Texecution>::NumpyImplementation<T>::operator=(
    const NumpyImplementation &other) {
  typename GilControl<Texecution>::Acquire acquire;
  this->_vector = std::make_unique<py::array_t<T>>(*other._vector);
  this->_buffer = std::make_unique<py::buffer_info>(*other._buffer);
  return *this;
}

template <typename Texecution>
template <typename T>
PythonContainerProxy<Texecution>::NumpyImplementation<
    T>::~NumpyImplementation() {
  typename GilControl<Texecution>::Acquire acquire;
  _buffer.reset();
  _vector.reset();
}

template <typename Texecution>
template <typename T>
std::size_t
PythonContainerProxy<Texecution>::NumpyImplementation<T>::size() const {
  typename GilControl<Texecution>::Acquire acquire;
  return _vector->size();
}

template <typename Texecution>
template <typename T>
typename PythonContainerProxy<Texecution>::value_type
PythonContainerProxy<Texecution>::NumpyImplementation<T>::at(
    std::size_t index) const {
  typename GilControl<Texecution>::Acquire acquire;
  return value_type(((T *)_buffer->ptr)[index]);
}

template <typename Texecution>
template <typename T>
bool PythonContainerProxy<Texecution>::NumpyImplementation<T>::same_type(
    const BaseImplementation &other) const {
  return dynamic_cast<const NumpyImplementation<T> *>(&other) != nullptr;
}

} // namespace skdecide

#endif // SKDECIDE_PYTHON_CONTAINER_PROXY_IMPL_HH
