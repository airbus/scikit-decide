/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PYTHON_CONTAINER_PROXY_HH
#define SKDECIDE_PYTHON_CONTAINER_PROXY_HH

#include <type_traits>
#include <memory>

namespace pybind11 {
    class object;
    struct buffer_info;
    class array;
}

namespace py = pybind11;

namespace skdecide {

template <typename Texecution> struct PythonHash;
template <typename Texecution> struct PythonEqual;

template <typename Texecution>
class PythonContainerProxy {
public :

    class value_type {
    public :
        value_type();
        value_type(const py::object& value);

        template <typename T,
                  std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
        value_type(const T& value) {
            _value = std::make_unique<PrimitiveType<T>>(value);
        }

        value_type(const value_type& other);
        value_type& operator= (const value_type& other);
        std::size_t hash() const;
        bool operator== (const value_type& other) const;

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
            PrimitiveType(const T& value);
            virtual ~PrimitiveType();
            virtual void copy(std::unique_ptr<BaseType>& other) const;
            virtual std::size_t hash() const;
            virtual bool equal(const BaseType& other) const;

        private :
            T _value;
        };

        class ObjectType : public BaseType {
        public :
            ObjectType(const py::object& value);
            ObjectType(const ObjectType& other);
            ObjectType& operator=(const ObjectType& other);
            virtual ~ObjectType();
            virtual void copy(std::unique_ptr<BaseType>& other) const;
            virtual std::size_t hash() const;
            virtual bool equal(const BaseType& other) const;

        private :
            std::unique_ptr<py::object> _value;
        };

        std::unique_ptr<BaseType> _value;
    };

    PythonContainerProxy();
    PythonContainerProxy(const py::object& vector);
    std::size_t size() const;
    value_type operator[](std::size_t index) const;
    std::size_t hash() const;
    bool operator== (const PythonContainerProxy& other) const;

private :

    class BaseImplementation {
    public :
        virtual ~BaseImplementation() {}
        virtual std::size_t size() const =0;
        virtual value_type at(std::size_t index) const =0;
        virtual bool same_type(const BaseImplementation& other) const =0;
        virtual std::size_t hash() const;
        virtual bool equal(const BaseImplementation& other) const;
    };

    template <typename Tsequence>
    class SequenceImplementation : public BaseImplementation {
    public :
        SequenceImplementation(const py::object& vector);
        SequenceImplementation(const SequenceImplementation& other);
        SequenceImplementation& operator=(const SequenceImplementation& other);
        virtual ~SequenceImplementation();
        virtual std::size_t size() const;
        virtual value_type at(std::size_t index) const;
        virtual bool same_type(const BaseImplementation& other) const;

    private :
        std::unique_ptr<Tsequence> _vector;
    };

    template <typename T>
    class NumpyImplementation : public BaseImplementation {
    public :
        NumpyImplementation(const py::object& vector);
        NumpyImplementation(const NumpyImplementation& other);
        NumpyImplementation& operator=(const NumpyImplementation& other);
        virtual ~NumpyImplementation();
        virtual std::size_t size() const;
        virtual value_type at(std::size_t index) const;
        virtual bool same_type(const BaseImplementation& other) const;
    
    private :
        std::unique_ptr<py::array> _vector;
        std::unique_ptr<py::buffer_info> _buffer;
    };

    std::unique_ptr<BaseImplementation> _implementation;
};

struct SequentialExecution;
struct ParallelExecution;
inline std::size_t hash_value(const PythonContainerProxy<SequentialExecution>::value_type& o) { return o.hash(); }
inline std::size_t hash_value(const PythonContainerProxy<ParallelExecution>::value_type& o) { return o.hash(); }

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/python_container_proxy_impl.hh"
#endif

#endif // SKDECIDE_PYTHON_CONTAINER_PROXY_HH
