/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_GYM_SPACES_HH
#define AIRLAPS_GYM_SPACES_HH

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <functional>
#include <cmath>
#include <list>
#include <map>

namespace py = pybind11;

namespace airlaps {

class GymSpace {
public :
    typedef enum {
        ENCODING_BYTE_VECTOR, // all numbers encoded as byte vectors without loss (large vector if there are many numbers requiring each 4 or 8 bytes)
        ENCODING_VARIABLE_VECTOR // each feature atom vector entry is an observation or action variable (clustering using 'space_relative_precision' for floating point value => compression with loss)
    } Encoding;

    GymSpace(unsigned int feature_atom_vector_begin = 0) : number_of_feature_atoms_(0), feature_atom_vector_begin_(feature_atom_vector_begin) {}
    virtual ~GymSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, Encoding encoding, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    inline unsigned int get_number_of_feature_atoms() const {return number_of_feature_atoms_;}
    virtual unsigned int get_number_of_tracked_atoms() const =0;
    std::vector<int> convert_element_to_feature_atoms(const py::object& element);
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const =0;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const =0;
    void enumerate(const std::function<void(const std::vector<int>&)>& f) const;
    virtual void enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const =0;
    
protected :
    unsigned int number_of_feature_atoms_;
    unsigned int feature_atom_vector_begin_; // index of the first element of this Gym space in the whole feature atom vector
    
private :
    template <template <Encoding E, typename ... Types> class C, typename ... Types>
    struct import_encoded_from_python {
        template <typename ... Args>
        std::unique_ptr<GymSpace> operator()(Encoding encoding, Args ... args) {
            switch (encoding) {
                case ENCODING_BYTE_VECTOR:
                    return C<ENCODING_BYTE_VECTOR, Types ...>::import_from_python(args ...);
                
                case ENCODING_VARIABLE_VECTOR:
                    return C<ENCODING_VARIABLE_VECTOR, Types ...>::import_from_python(args ...);
            }
            return std::unique_ptr<GymSpace>(); // dummy return to prevent compiler warning related to switch statement
        }
    };
};


template <GymSpace::Encoding E, typename T>
class BoxSpace : public GymSpace {
public :
    // constructor for variable vector encoding
    BoxSpace(const py::array_t<T>& low, const py::array_t<T>& high, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0)
    try : GymSpace(feature_atom_vector_begin), low_(low), high_(high), space_relative_precision_(space_relative_precision) {
        if (low_.ndim() != high_.ndim()) {
            throw std::domain_error("Gym box space's 'low' and 'high' arrays not of the same dimension");
        }
        for (unsigned int d = 0 ; d < low_.ndim() ; d++) {
            if (low_.shape(d) != high_.shape(d)) {
                throw std::domain_error("Gym box space's 'low' and 'high' arrays' dimension " + std::to_string(d) + " not of the same size");
            }
        }
        initialize_number_of_atoms();
    } catch (const std::exception& e) {
        throw e;
    }

    virtual ~BoxSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    
    virtual inline unsigned int get_number_of_tracked_atoms() const {
        return get_number_of_tracked_atoms_generic();
    }
    
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
        convert_element_to_feature_atoms_generic(element, feature_atoms);
    }

    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
        return convert_feature_atoms_to_element_generic(feature_atoms);
    }

    virtual void enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
        enumerate_generic(f, feature_atoms);
    }

private :
    mutable py::array_t<T> low_; // dirty trick to make py::array_t<T>::request() work with 'const this' since it seems it does not modify the array
    mutable py::array_t<T> high_; // dirty trick to make py::array_t<T>::request() work with 'const this' since it seems it does not modify the array
    double space_relative_precision_;

    inline static T sigmoid(const T& x, const T& decay) {
        return ((T) 1.0) / (((T) 1.0) + std::exp(-decay * x));
    }

    inline static T inv_sigmoid(const T& x, const T& decay) {
        return -std::log((((T) 1.0) / x) - ((T) 1.0)) / decay;
    }

    inline static T normalize(const T& x, const T& min, const T& max, const T& decay) {
        return (sigmoid(x, decay) - sigmoid(min, decay)) / (sigmoid(max, decay) - sigmoid(min, decay));
    }

    inline static T inv_normalize(const T& x, const T& min, const T& max, const T& decay) {
        return inv_sigmoid(sigmoid(min, decay) + (x * (sigmoid(max, decay) - sigmoid(min, decay))), decay);
    }

    // Initializer for byte vector encoding
    template <Encoding EE = E>
    inline std::enable_if_t<EE == GymSpace::ENCODING_BYTE_VECTOR, void> initialize_number_of_atoms() {
        number_of_feature_atoms_ = low_.size() * sizeof(T);
    }

    // Initializer for variable vector encoding
    template <Encoding EE = E>
    inline std::enable_if_t<EE == GymSpace::ENCODING_VARIABLE_VECTOR, void> initialize_number_of_atoms() {
        number_of_feature_atoms_ = low_.size();
    }

    // Get number of tracked atoms for byte vector encoding
    template <Encoding EE = E>
    inline std::enable_if_t<EE == GymSpace::ENCODING_BYTE_VECTOR, unsigned int> get_number_of_tracked_atoms_generic() const {
        return number_of_feature_atoms_ * 256;
    }

    // Get number of tracked atoms for variable vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_VARIABLE_VECTOR)  && std::is_integral<TT>::value, unsigned int> get_number_of_tracked_atoms_generic() const {
        return get_number_of_tracked_atoms_int();
    }

    // Get number of tracked atoms for variable vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_VARIABLE_VECTOR)  && std::is_floating_point<TT>::value, unsigned int> get_number_of_tracked_atoms_generic() const {
        return get_number_of_tracked_atoms_float();
    }

    unsigned int get_number_of_tracked_atoms_int() const;
    unsigned int get_number_of_tracked_atoms_float() const;

    // Convertor for byte vector encoding
    template <Encoding EE = E>
    inline std::enable_if_t<EE == GymSpace::ENCODING_BYTE_VECTOR, void>
    convert_element_to_feature_atoms_generic(const py::object& element, std::vector<int>& feature_atoms) const {
        convert_element_to_feature_atoms_byte(element, feature_atoms);
    }

    // Convertor for variable vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_VARIABLE_VECTOR) && std::is_integral<TT>::value, void>
    convert_element_to_feature_atoms_generic(const py::object& element, std::vector<int>& feature_atoms) const {
        convert_element_to_feature_atoms_int(element, feature_atoms);
    }

    // Convertor for variable vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_VARIABLE_VECTOR) && std::is_floating_point<TT>::value, void>
    convert_element_to_feature_atoms_generic(const py::object& element, std::vector<int>& feature_atoms) const {
        convert_element_to_feature_atoms_float(element, feature_atoms);
    }

    void convert_element_to_feature_atoms_byte(const py::object& element, std::vector<int>& feature_atoms) const;
    void convert_element_to_feature_atoms_int(const py::object& element, std::vector<int>& feature_atoms) const;
    void convert_element_to_feature_atoms_float(const py::object& element, std::vector<int>& feature_atoms) const;
    
    // Convertor for byte vector encoding
    template <Encoding EE = E>
    inline std::enable_if_t<EE == GymSpace::ENCODING_BYTE_VECTOR, py::object>
    convert_feature_atoms_to_element_generic(const std::vector<int>& feature_atoms) const {
        return convert_feature_atoms_to_element_byte(feature_atoms);
    }
    
    // Convertor for variable vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_VARIABLE_VECTOR) && std::is_integral<TT>::value, py::object>
    convert_feature_atoms_to_element_generic(const std::vector<int>& feature_atoms) const {
        return convert_feature_atoms_to_element_int(feature_atoms);
    }
    
    // Convertor for variable vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_VARIABLE_VECTOR) && std::is_floating_point<TT>::value, py::object>
    convert_feature_atoms_to_element_generic(const std::vector<int>& feature_atoms) const {
        return convert_feature_atoms_to_element_float(feature_atoms);
    }

    py::object convert_feature_atoms_to_element_byte(const std::vector<int>& feature_atoms) const;
    py::object convert_feature_atoms_to_element_int(const std::vector<int>& feature_atoms) const;
    py::object convert_feature_atoms_to_element_float(const std::vector<int>& feature_atoms) const;
    
    // Enumerator for byte vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_BYTE_VECTOR) && std::is_same<TT, bool>::value, void>
    enumerate_generic(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
        enumerate_byte_bool(0, low_.request(), high_.request(), f, feature_atoms);
    }

    // Enumerator for byte vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_BYTE_VECTOR) && std::is_integral<TT>::value && !std::is_same<TT, bool>::value, void>
    enumerate_generic(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
        enumerate_byte_int(0, low_.request(), high_.request(), f, feature_atoms);
    }

    // Enumerator for byte vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_BYTE_VECTOR) && std::is_floating_point<TT>::value, void>
    enumerate_generic(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
        enumerate_byte_float(0, low_.request(), high_.request(), f, feature_atoms);
    }

    // Enumerator for variable vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_VARIABLE_VECTOR) && std::is_same<TT, bool>::value, void>
    enumerate_generic(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
        enumerate_variable_bool(0, low_.request(), high_.request(), f, feature_atoms);
    }

    // Enumerator for variable vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_VARIABLE_VECTOR) && std::is_integral<TT>::value && !std::is_same<TT, bool>::value, void>
    enumerate_generic(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
        enumerate_variable_int(0, low_.request(), high_.request(), f, feature_atoms);
    }

    // Enumerator for variable vector encoding
    template <Encoding EE = E, typename TT = T>
    inline std::enable_if_t<(EE == GymSpace::ENCODING_VARIABLE_VECTOR) && std::is_floating_point<TT>::value, void>
    enumerate_generic(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
        enumerate_variable_float(0, low_.request(), high_.request(), f, feature_atoms);
    }

    void enumerate_byte_bool(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                             const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
    void enumerate_byte_int(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                            const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
    void enumerate_byte_float(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                              const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
    void enumerate_variable_bool(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                                 const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
    void enumerate_variable_int(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                                const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
    void enumerate_variable_float(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                                  const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
};


class DictSpace : public GymSpace {
public :
    DictSpace(const py::dict& spaces, Encoding encoding, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    virtual ~DictSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, Encoding encoding, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    virtual unsigned int get_number_of_tracked_atoms() const;
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const;
    virtual void enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;

private :
    std::map<std::string, std::unique_ptr<GymSpace>> spaces_;

    void enumerate(std::map<std::string, std::unique_ptr<GymSpace>>::const_iterator current_space,
                   const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
};


template <GymSpace::Encoding E>
class DiscreteSpace : public GymSpace {
public :
    DiscreteSpace(const py::int_& n, unsigned int feature_atom_vector_begin = 0);
    virtual ~DiscreteSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin = 0);
    virtual unsigned int get_number_of_tracked_atoms() const;
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const;
    virtual void enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
    
private :
    std::int64_t n_;
};


template <GymSpace::Encoding E>
class MultiBinarySpace : public GymSpace {
public :
    MultiBinarySpace(const py::int_& n, unsigned int feature_atom_vector_begin = 0);
    virtual ~MultiBinarySpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin = 0);
    virtual unsigned int get_number_of_tracked_atoms() const;
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const;
    virtual void enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
    
private :
    std::int8_t n_;

    void enumerate_byte(unsigned int current_bit, unsigned char current_byte,
                        const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
    void enumerate_variable(unsigned int current_bit,
                            const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
};


template <GymSpace::Encoding E>
class MultiDiscreteSpace : public GymSpace {
public :
    MultiDiscreteSpace(const py::array_t<unsigned int>& nvec, unsigned int feature_atom_vector_begin = 0);
    virtual ~MultiDiscreteSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin = 0);
    virtual unsigned int get_number_of_tracked_atoms() const;
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const;
    virtual void enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
    
private :
    mutable py::array_t<std::int64_t> nvec_; // dirty trick to make py::array_t<T>::request() work with 'const this' since it seems it does not modify the array

    void enumerate(unsigned int current_item, const py::buffer_info& buf,
                   const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
};


class TupleSpace : public GymSpace {
public :
    TupleSpace(const py::tuple& spaces, Encoding encoding, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    virtual ~TupleSpace() {}

    static std::unique_ptr<GymSpace> import_from_python(const py::object& gym_space, Encoding encoding, double space_relative_precision = 0.001, unsigned int feature_atom_vector_begin = 0);
    virtual unsigned int get_number_of_tracked_atoms() const;
    virtual void convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const;
    virtual py::object convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const;
    virtual void enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
    
private :
    std::list<std::unique_ptr<GymSpace>> spaces_;
    void enumerate(std::list<std::unique_ptr<GymSpace>>::const_iterator current_space,
                   const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const;
};

} // namespace airlaps

#endif // AIRLAPS_GYM_SPACES_HH
