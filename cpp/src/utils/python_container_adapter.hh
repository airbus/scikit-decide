/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_CONTAINER_ADAPTER_HH
#define SKDECIDE_PYTHON_CONTAINER_ADAPTER_HH

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <boost/container_hash/hash.hpp>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "utils/execution.hh"

namespace py = pybind11;

namespace skdecide {

template <typename Texecution> struct PythonHash;
template <typename Texecution> struct PythonEqual;

template <typename Texecution>
class PythonContainerAdapter {
public :

    class value_type {
    public :
        value_type() {}
        
        value_type(const py::object& value) {
            _value = std::make_unique<ObjectType>(value);
        }

        template <typename T,
                  std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
        value_type(const T& value) {
            _value = std::make_unique<PrimitiveType<T>>(value);
        }

        value_type(const value_type& other) {
            other._value->copy(_value);
        }

        value_type& operator= (const value_type& other) {
            other._value->copy(_value);
            return *this;
        }

        std::size_t hash() const {
            return _value->hash();
        }

        bool operator== (const value_type& other) const {
            return _value->equal(*(other._value));
        }

    private :
        class BaseType {
        public :
            virtual ~BaseType() {}
            virtual void copy(std::unique_ptr<BaseType>& other) const =0;
            virtual std::size_t hash() const =0;
            virtual bool equal(const BaseType& other) const =0;
        };

        template <typename T>
        class PrimitiveType : public BaseType {
        public :
            PrimitiveType(const T& value) : _value(value) {}
            virtual ~PrimitiveType() {}

            virtual void copy(std::unique_ptr<BaseType>& other) const {
                other = std::make_unique<PrimitiveType<T>>(_value);
            }

            virtual std::size_t hash() const {
                return boost::hash_value(_value);
            }

            virtual bool equal(const BaseType& other) const {
                const PrimitiveType<T>* o = dynamic_cast<const PrimitiveType<T>*>(&other);
                return ((o != nullptr) && (o->_value == _value));
            }

        private :
            T _value;
        };

        class ObjectType : public BaseType {
        public :
            ObjectType(const py::object& value) {
                typename GilControl<Texecution>::Acquire acquire;
                this->_value = std::make_unique<py::object>(value);
            }

            ObjectType(const ObjectType& other) {
                typename GilControl<Texecution>::Acquire acquire;
                this->_value = std::make_unique<py::object>(*other.value);
            }

            ObjectType& operator=(const ObjectType& other) {
                typename GilControl<Texecution>::Acquire acquire;
                this->_value = std::make_unique<py::object>(*other.value);
                return *this;
            }

            virtual ~ObjectType() {
                typename GilControl<Texecution>::Acquire acquire;
                _value.reset();
            }

            virtual void copy(std::unique_ptr<BaseType>& other) const {
                other = std::make_unique<ObjectType>(*_value);
            }

            virtual std::size_t hash() const {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    return skdecide::PythonHash<Texecution>()(*_value);
                } catch(const py::error_already_set* e) {
                    spdlog::error(std::string("SKDECIDE exception when hashing state feature items: ") + e->what());
                    std::runtime_error err(e->what());
                    delete e;
                    throw err;
                }
            }

            virtual bool equal(const BaseType& other) const {
                typename GilControl<Texecution>::Acquire acquire;
                try {
                    const ObjectType* o = dynamic_cast<const ObjectType*>(&other);
                    return  ((o != nullptr) && skdecide::PythonEqual<Texecution>()(*_value, *(o->_value)));
                } catch(const py::error_already_set* e) {
                    spdlog::error(std::string("SKDECIDE exception when testing state feature items equality: ") + e->what());
                    std::runtime_error err(e->what());
                    delete e;
                    throw err;
                }
            }

        private :
            std::unique_ptr<py::object> _value;
        };

        std::unique_ptr<BaseType> _value;
    };

    PythonContainerAdapter() {}

    PythonContainerAdapter(const py::object& vector) {
        typename GilControl<Texecution>::Acquire acquire;
        if (py::isinstance<py::list>(vector)) {
            _implementation = std::make_unique<SequenceImplementation<py::list>>(vector);
        } else if (py::isinstance<py::tuple>(vector)) {
            _implementation = std::make_unique<SequenceImplementation<py::tuple>>(vector);
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
                _implementation = std::make_unique<NumpyImplementation<std::size_t>>(vector);
            } else if (dtype == "int8") {
                _implementation = std::make_unique<NumpyImplementation<std::int8_t>>(vector);
            } else if (dtype == "int16") {
                _implementation = std::make_unique<NumpyImplementation<std::int16_t>>(vector);
            } else if (dtype == "int32") {
                _implementation = std::make_unique<NumpyImplementation<std::int32_t>>(vector);
            } else if (dtype == "int64") {
                _implementation = std::make_unique<NumpyImplementation<std::int64_t>>(vector);
            } else if (dtype == "uint8") {
                _implementation = std::make_unique<NumpyImplementation<std::uint8_t>>(vector);
            } else if (dtype == "uint16") {
                _implementation = std::make_unique<NumpyImplementation<std::uint16_t>>(vector);
            } else if (dtype == "uint32") {
                _implementation = std::make_unique<NumpyImplementation<std::uint32_t>>(vector);
            } else if (dtype == "uint64") {
                _implementation = std::make_unique<NumpyImplementation<std::uint64_t>>(vector);
            } else {
                spdlog::error("Unhandled array dtype '" + dtype + "' when parsing python sequence as numpy array");
                throw std::invalid_argument("SKDECIDE exception: Unhandled array dtype '" + dtype +
                                            "' when parsing state features as numpy array");
            }
        } else {
            spdlog::error("Unhandled state feature type '" + std::string(py::str(vector.attr("__class__").attr("__name__"))) +
                           " (expecting list, tuple or numpy array)");
            throw std::invalid_argument("Unhandled state feature type '" + std::string(py::str(vector.attr("__class__").attr("__name__"))) +
                                        " (expecting list, tuple or numpy array)");
        }
    }

    std::size_t size() const {
        return _implementation->size();
    }

    value_type operator[](std::size_t index) const {
        return _implementation->at(index);
    }

    std::size_t hash() const {
        return _implementation->hash();
    }

    bool operator== (const value_type& other) const {
        return _implementation->equal(*(other._value));
    }

private :

    class BaseImplementation {
    public :
        virtual ~BaseImplementation() {}
        virtual std::size_t size() const =0;
        virtual value_type at(std::size_t index) const =0;
        virtual bool same_type(const BaseImplementation& other) const =0;

        virtual std::size_t hash() const {
            std::size_t seed = 0;
            for (std::size_t i = 0 ; i < size() ; i++) {
                boost::hash_combine(seed, at(i));
            }
            return seed;
        }

        virtual bool equal(const BaseImplementation& other) const {
            if (this->same_type(other) && (this->size() == other.size())) {
                for (std::size_t i = 0 ; i < this->size() ; i++) {
                    if(!(this->at(i) == other.at(i))) {
                        return false;
                    }
                }
                return true;
            } else {
                return false;
            }
        }
    };

    template <typename Tsequence>
    class SequenceImplementation : public BaseImplementation {
    public :
        SequenceImplementation(const py::object& vector) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_vector = std::make_unique<Tsequence>(vector);
        }

        SequenceImplementation(const SequenceImplementation& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_vector = std::make_unique<Tsequence>(*other._vector);
        }

        SequenceImplementation& operator=(const SequenceImplementation& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_vector = std::make_unique<Tsequence>(*other._vector);
            return *this;
        }

        virtual ~SequenceImplementation() {
            typename GilControl<Texecution>::Acquire acquire;
            _vector.reset();
        }

        virtual std::size_t size() const {
            typename GilControl<Texecution>::Acquire acquire;
            return _vector->size();
        }

        virtual value_type at(std::size_t index) const {
            typename GilControl<Texecution>::Acquire acquire;
            return value_type((*_vector)[index]);
        }

        virtual bool same_type(const BaseImplementation& other) const {
            return dynamic_cast<const SequenceImplementation<Tsequence>*>(&other) != nullptr;
        }

    private :
        std::unique_ptr<Tsequence> _vector;
    };

    template <typename T>
    class NumpyImplementation : public BaseImplementation {
    public :
        NumpyImplementation(const py::object& vector) {
            typename GilControl<Texecution>::Acquire acquire;
            _vector = std::make_unique<py::array_t<T>>(vector);
            _buffer = std::make_unique<py::buffer_info>(_vector->request());
        }

        NumpyImplementation(const NumpyImplementation& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_vector = std::make_unique<py::array_t<T>>(*other._vector);
            this->_buffer = std::make_unique<py::buffer_info>(*other._buffer);
        }

        NumpyImplementation& operator=(const NumpyImplementation& other) {
            typename GilControl<Texecution>::Acquire acquire;
            this->_vector = std::make_unique<py::array_t<T>>(*other._vector);
            this->_buffer = std::make_unique<py::buffer_info>(*other._buffer);
            return *this;
        }

        virtual ~NumpyImplementation() {
            typename GilControl<Texecution>::Acquire acquire;
            _buffer.reset();
            _vector.reset();
        }

        virtual std::size_t size() const {
            typename GilControl<Texecution>::Acquire acquire;
            return _vector->size();
        }

        virtual value_type at(std::size_t index) const {
            typename GilControl<Texecution>::Acquire acquire;
            return value_type(((T*) _buffer->ptr)[index]);
        }

        virtual bool same_type(const BaseImplementation& other) const {
            return dynamic_cast<const NumpyImplementation<T>*>(&other) != nullptr;
        }
    
    private :
        std::unique_ptr<py::array_t<T>> _vector;
        std::unique_ptr<py::buffer_info> _buffer;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};


inline std::size_t hash_value(const PythonContainerAdapter<skdecide::SequentialExecution>::value_type& o) {
    return o.hash();
}

inline std::size_t hash_value(const PythonContainerAdapter<skdecide::ParallelExecution>::value_type& o) {
    return o.hash();
}

} // namespace skdecide

#endif // SKDECIDE_PYTHON_CONTAINER_ADAPTER_HH
