/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "gym_spaces.hh"
#include <algorithm>

using namespace skdecide;

std::unique_ptr<GymSpace> GymSpace::import_from_python(const py::object& gym_space, Encoding encoding, double space_relative_precision, unsigned int feature_atom_vector_begin) {
    std::string space = py::str(gym_space.attr("__class__").attr("__name__"));
    
    if (space == "Box") {
        if (!py::hasattr(gym_space, "dtype")) {
            py::print("ERROR: Gym box space missing attribute 'dtype'");
            return std::unique_ptr<GymSpace>();
        }
        std::string dtype = py::str(gym_space.attr("dtype"));
        if (dtype == "bool_")
            return import_encoded_from_python<BoxSpace, bool>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "int_")
            return import_encoded_from_python<BoxSpace, long int>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "intc")
            return import_encoded_from_python<BoxSpace, int>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "intp")
            return import_encoded_from_python<BoxSpace, std::size_t>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "int8")
            return import_encoded_from_python<BoxSpace, std::int8_t>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "int16")
            return import_encoded_from_python<BoxSpace, std::int16_t>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "int32")
            return import_encoded_from_python<BoxSpace, std::int32_t>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "int64")
            return import_encoded_from_python<BoxSpace, std::int64_t>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "uint8")
            return import_encoded_from_python<BoxSpace, std::uint8_t>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "uint16")
            return import_encoded_from_python<BoxSpace, std::uint16_t>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "uint32")
            return import_encoded_from_python<BoxSpace, std::uint32_t>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "uint64")
            return import_encoded_from_python<BoxSpace, std::uint64_t>()(encoding, gym_space, 0, feature_atom_vector_begin);
        else if (dtype == "float_")
            return import_encoded_from_python<BoxSpace, double>()(encoding, gym_space, space_relative_precision, feature_atom_vector_begin);
        else if (dtype == "float32")
            return import_encoded_from_python<BoxSpace, float>()(encoding, gym_space, space_relative_precision, feature_atom_vector_begin);
        else if (dtype == "float64")
            return import_encoded_from_python<BoxSpace, double>()(encoding, gym_space, space_relative_precision, feature_atom_vector_begin);
        else {
            py::print("ERROR: Unhandled array dtype '" + dtype + "' when importing Gym box space bounds");
            return std::unique_ptr<GymSpace>();
        }
    } else if (space == "Dict") {
        return DictSpace::import_from_python(gym_space, encoding, space_relative_precision, feature_atom_vector_begin);
    } else if (space == "Discrete") {
        return import_encoded_from_python<DiscreteSpace>()(encoding, gym_space, feature_atom_vector_begin);
    } else if (space == "MultiBinary") {
        return import_encoded_from_python<MultiBinarySpace>()(encoding, gym_space, feature_atom_vector_begin);
    } else if (space == "MultiDiscrete") {
        return import_encoded_from_python<MultiDiscreteSpace>()(encoding, gym_space, feature_atom_vector_begin);
    } else if (space == "Tuple") {
        return TupleSpace::import_from_python(gym_space, encoding, space_relative_precision, feature_atom_vector_begin);
    } else {
        py::print("ERROR: Unhandled Gym space '" + space + "'");
        return std::unique_ptr<GymSpace>();
    }
}


std::vector<int> GymSpace::convert_element_to_feature_atoms(const py::object& element) {
    std::vector<int> feature_atoms(number_of_feature_atoms_);
    convert_element_to_feature_atoms(element, feature_atoms);
    return feature_atoms;
}


void GymSpace::enumerate(const std::function<void(const std::vector<int>&)>& f) const {
    std::vector<int> feature_atoms(number_of_feature_atoms_);
    enumerate(f, feature_atoms);
}


// --- BoxSpace ---

template <GymSpace::Encoding E, typename T>
std::unique_ptr<GymSpace> BoxSpace<E, T>::import_from_python(const py::object& gym_space, double space_relative_precision, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "low") || !py::hasattr(gym_space, "high")) {
            py::print("ERROR: Gym box space missing attributes 'low' or 'high'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::array_t<T>>(gym_space.attr("low")) || !py::isinstance<py::array_t<T>>(gym_space.attr("high"))) {
            py::print("ERROR: Gym box space's attributes 'low' or 'high' not numpy arrays");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<BoxSpace<E, T>>(py::cast<py::array_t<T>>(gym_space.attr("low")), py::cast<py::array_t<T>>(gym_space.attr("high")), space_relative_precision, feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when importing Gym box space");
        return std::unique_ptr<GymSpace>();
    } catch (const std::exception& e) {
        py::print("ERROR: " + std::string(e.what()));
        return std::unique_ptr<GymSpace>();
    }
}


template <GymSpace::Encoding E, typename T>
unsigned int BoxSpace<E, T>::get_number_of_tracked_atoms_int() const {
    py::buffer_info lbuf = low_.request();
    py::buffer_info hbuf = high_.request();
    unsigned int total = 0;
    for (unsigned int i = 0 ; i < lbuf.size ; i++) {
        total += ((T *) hbuf.ptr)[i] - ((T *) lbuf.ptr)[i];
    }
    return total;
}


template <GymSpace::Encoding E, typename T>
unsigned int BoxSpace<E, T>::get_number_of_tracked_atoms_float() const {
    return number_of_feature_atoms_ * ((unsigned int) std::floor(1.0 / space_relative_precision_));
}


template <GymSpace::Encoding E, typename T>
void BoxSpace<E, T>::convert_element_to_feature_atoms_byte(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array>(element)) {
            py::print("ERROR: Gym box space element not a numpy array");
        }
        py::array_t<T> celement;
        if (!py::isinstance<py::array_t<T>>(element)) {
            //py::print("WARNING: Gym box space element not of same 'dtype' than the box space (requires a copy of the element to the box space's 'dtype')");
            celement = py::cast<py::array_t<T>>(element.attr("astype")(low_.attr("dtype")));
        } else {
            celement = py::cast<py::array_t<T>>(element);
        }
        py::buffer_info buf = celement.request();
        if (low_.size() != buf.size) {
            py::print("ERROR: Gym box space element and 'low' array not of the same size");
            return;
        }
        for (unsigned int i = 0 ; i < buf.size ; i++) {
            unsigned char* byte_array = reinterpret_cast<unsigned char*>(&(((T *) buf.ptr)[i]));
            for (unsigned int j = 0 ; j < sizeof(T) ; j++) {
                feature_atoms[feature_atom_vector_begin_ + (i * sizeof(T)) + j] = (int) byte_array[j];
            }
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when converting Gym box space element to a feature atom vector");
    }
}


template <GymSpace::Encoding E, typename T>
void BoxSpace<E, T>::convert_element_to_feature_atoms_int(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array>(element)) {
            py::print("ERROR: Gym box space element not a numpy array");
        }
        py::array_t<T> celement;
        if (!py::isinstance<py::array_t<T>>(element)) {
            //py::print("WARNING: Gym box space element not of same 'dtype' than the box space (requires a copy of the element to the box space's 'dtype')");
            celement = py::cast<py::array_t<T>>(element.attr("astype")(low_.attr("dtype")));
        } else {
            celement = py::cast<py::array_t<T>>(element);
        }
        py::buffer_info buf = celement.request();
        py::buffer_info lbuf = low_.request();
        if (low_.size() != buf.size) {
            py::print("ERROR: Gym box space element and 'low' array not of the same size");
            return;
        }
        for (unsigned int i = 0 ; i < buf.size ; i++) {
            feature_atoms[feature_atom_vector_begin_ + i] = ((T *) buf.ptr)[i] - ((T *) lbuf.ptr)[i];
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when converting Gym box space element to a feature atom vector");
    }
}


template <GymSpace::Encoding E, typename T>
void BoxSpace<E, T>::convert_element_to_feature_atoms_float(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array>(element)) {
            py::print("ERROR: Gym box space element not a numpy array");
        }
        py::array_t<T> celement;
        if (!py::isinstance<py::array_t<T>>(element)) {
            //py::print("WARNING: Gym box space element not of same 'dtype' than the box space (requires a copy of the element to the box space's 'dtype')");
            celement = py::cast<py::array_t<T>>(element.attr("astype")(low_.attr("dtype")));
        } else {
            celement = py::cast<py::array_t<T>>(element);
        }
        py::buffer_info ebuf = celement.request();
        py::buffer_info lbuf = low_.request(); // request() does not change the array
        py::buffer_info hbuf = high_.request(); // request() does not change the array
        if (lbuf.size != ebuf.size) {
            py::print("ERROR: Gym box space element and 'low' array not of the same size");
            return;
        }
        for (unsigned int i = 0 ; i < ebuf.size ; i++) {
            feature_atoms[feature_atom_vector_begin_ + i] = (int) std::floor(normalize(((T*) ebuf.ptr)[i], ((T*) lbuf.ptr)[i], ((T*) hbuf.ptr)[i], space_relative_precision_) / space_relative_precision_);
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when converting Gym box space element to a feature atom vector");
    }
}


template <GymSpace::Encoding E, typename T>
py::object BoxSpace<E, T>::convert_feature_atoms_to_element_byte(const std::vector<int>& feature_atoms) const {
    const ssize_t* shape = low_.shape();
    py::array_t<T> result = py::array_t<T>(std::vector<ssize_t>(shape, shape + low_.ndim()));
    py::buffer_info buf = result.request();
    unsigned char* byte_array = new unsigned char[sizeof(T)];
    for (unsigned int i = 0 ; i < buf.size ; i++) {
        for (unsigned int j = 0 ; j < sizeof(T) ; j++) {
            byte_array[j] = (unsigned char) feature_atoms[feature_atom_vector_begin_ + (i * sizeof(T)) + j];
        }
        ((T*) buf.ptr)[i] = *reinterpret_cast<T*>(byte_array);
    }
    delete[] byte_array;
    return result;
}


template <GymSpace::Encoding E, typename T>
py::object BoxSpace<E, T>::convert_feature_atoms_to_element_int(const std::vector<int>& feature_atoms) const {
    const ssize_t* shape = low_.shape();
    py::array_t<T> result = py::array_t<T>(std::vector<ssize_t>(shape, shape + low_.ndim()));
    py::buffer_info buf = result.request();
    py::buffer_info lbuf = low_.request();
    for (unsigned int i = 0 ; i < buf.size ; i++) {
        ((T*) buf.ptr)[i] = ((T*) lbuf.ptr)[i] + feature_atoms[feature_atom_vector_begin_ + i];
    }
    return result;
}


template <GymSpace::Encoding E, typename T>
py::object BoxSpace<E, T>::convert_feature_atoms_to_element_float(const std::vector<int>& feature_atoms) const {
    const ssize_t* shape = low_.shape();
    py::array_t<T> result = py::array_t<T>(std::vector<ssize_t>(shape, shape + low_.ndim()));
    py::buffer_info ebuf = result.request();
    py::buffer_info lbuf = low_.request(); // request() does not change the array
    py::buffer_info hbuf = high_.request(); // request() does not change the array
    for (unsigned int i = 0 ; i < ebuf.size ; i++) {
        ((T*) ebuf.ptr)[i] = inv_normalize(std::max((T) 0.0, std::min((T) (feature_atoms[feature_atom_vector_begin_ + i] * space_relative_precision_), (T) 1.0)),
                                           ((T*) lbuf.ptr)[i], ((T*) hbuf.ptr)[i], space_relative_precision_);
    }
    return result;
}


template <GymSpace::Encoding E, typename T>
void BoxSpace<E, T>::enumerate_byte_bool(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                                        const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    if (((bool*) lbuf.ptr)[current_item] != ((bool*) hbuf.ptr)[current_item]) {
        for (auto b : {false, true}) {
            unsigned char* byte_array = reinterpret_cast<unsigned char*>(&b);
            for (unsigned int j = 0 ; j < sizeof(bool) ; j++) {
                feature_atoms[feature_atom_vector_begin_ + (current_item * sizeof(bool)) + j] = byte_array[j];
            }
            if ((current_item + 1) < low_.size()) {
                enumerate_byte_bool(current_item + 1, lbuf, hbuf, f, feature_atoms);
            } else {
                f(feature_atoms);
            }
        }
    } else {
        unsigned char* byte_array = reinterpret_cast<unsigned char*>(&(((bool*) lbuf.ptr)[current_item]));
        for (unsigned int j = 0 ; j < sizeof(bool) ; j++) {
            feature_atoms[feature_atom_vector_begin_ + (current_item * sizeof(bool)) + j] = byte_array[j];
        }
        if ((current_item + 1) < low_.size()) {
            enumerate_byte_bool(current_item + 1, lbuf, hbuf, f, feature_atoms);
        } else {
            f(feature_atoms);
        }
    }
}


template <GymSpace::Encoding E, typename T>
void BoxSpace<E, T>::enumerate_byte_int(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                                        const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    for (T i = ((T*) lbuf.ptr)[current_item] ; i <= ((T*) hbuf.ptr)[current_item] ; i++) {
        unsigned char* byte_array = reinterpret_cast<unsigned char*>(&i);
        for (unsigned int j = 0 ; j < sizeof(T) ; j++) {
            feature_atoms[feature_atom_vector_begin_ + (current_item * sizeof(T)) + j] = byte_array[j];
        }
        if ((current_item + 1) < low_.size()) {
            enumerate_byte_int(current_item + 1, lbuf, hbuf, f, feature_atoms);
        } else {
            f(feature_atoms);
        }
    }
}


template <GymSpace::Encoding E, typename T>
void BoxSpace<E, T>::enumerate_byte_float(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                                          const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    for (T i = ((T*) lbuf.ptr)[current_item] ; i <= ((T*) hbuf.ptr)[current_item] ; i += std::numeric_limits<T>::epsilon()) {
        unsigned char* byte_array = reinterpret_cast<unsigned char*>(&i);
        for (unsigned int j = 0 ; j < sizeof(T) ; j++) {
            feature_atoms[feature_atom_vector_begin_ + (current_item * sizeof(T)) + j] = byte_array[j];
        }
        if ((current_item + 1) < low_.size()) {
            enumerate_byte_float(current_item + 1, lbuf, hbuf, f, feature_atoms);
        } else {
            f(feature_atoms);
        }
    }
}


template <GymSpace::Encoding E, typename T>
void BoxSpace<E, T>::enumerate_variable_bool(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                                             const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    if (((bool*) lbuf.ptr)[current_item] != ((bool*) hbuf.ptr)[current_item]) {
        for (auto b : {false, true}) {
            feature_atoms[feature_atom_vector_begin_ + current_item] = b;
            if ((current_item + 1) < low_.size()) {
                enumerate_variable_bool(current_item + 1, lbuf, hbuf, f, feature_atoms);
            } else {
                f(feature_atoms);
            }
        }
    } else {
        feature_atoms[feature_atom_vector_begin_ + current_item] = ((bool*) lbuf.ptr)[current_item];
        if ((current_item + 1) < low_.size()) {
            enumerate_variable_bool(current_item + 1, lbuf, hbuf, f, feature_atoms);
        } else {
            f(feature_atoms);
        }
    }
}


template <GymSpace::Encoding E, typename T>
void BoxSpace<E, T>::enumerate_variable_int(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                                            const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    for (T i = ((T*) lbuf.ptr)[current_item] ; i <= ((T*) hbuf.ptr)[current_item] ; i++) {
        feature_atoms[feature_atom_vector_begin_ + current_item] = i - ((T*) lbuf.ptr)[current_item];
        if ((current_item + 1) < low_.size()) {
            enumerate_variable_int(current_item + 1, lbuf, hbuf, f, feature_atoms);
        } else {
            f(feature_atoms);
        }
    }
}


template <GymSpace::Encoding E, typename T>
void BoxSpace<E, T>::enumerate_variable_float(unsigned int current_item, const py::buffer_info & lbuf, const py::buffer_info & hbuf,
                                              const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    unsigned int nb_clusters = (unsigned int) std::floor(1.0 / space_relative_precision_);
    for (unsigned int i = 0 ; i <= nb_clusters ; i++) {
        feature_atoms[feature_atom_vector_begin_ + current_item] = i;
        if ((current_item + 1) < low_.size()) {
            enumerate_variable_float(current_item + 1, lbuf, hbuf, f, feature_atoms);
        } else {
            f(feature_atoms);
        }
    }
}


// --- DictSpace ---

DictSpace::DictSpace(const py::dict& spaces, Encoding encoding, double space_relative_precision, unsigned int feature_atom_vector_begin)
try : GymSpace(feature_atom_vector_begin) {
    number_of_feature_atoms_ = 0;
    for (auto s : spaces) {
        auto i = spaces_.insert(std::make_pair(py::cast<py::str>(s.first),
                                               std::move(GymSpace::import_from_python(py::cast<py::object>(s.second),
                                                                                      encoding,
                                                                                      space_relative_precision,
                                                                                      feature_atom_vector_begin + number_of_feature_atoms_)))
                               ).first;
        number_of_feature_atoms_ += i->second->get_number_of_feature_atoms();
    }
} catch (const py::cast_error& e) {
    throw std::logic_error("ERROR: Python casting error (python object of unexpected type) when importing Gym dict space");
}


std::unique_ptr<GymSpace> DictSpace::import_from_python(const py::object& gym_space, Encoding encoding, double space_relative_precision, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "spaces")) {
            py::print("ERROR: Gym dict space missing attribute 'spaces'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::dict>(gym_space.attr("spaces"))) {
            py::print("ERROR: Gym dict space's 'spaces' not of type 'dict'");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<DictSpace>(py::cast<py::dict>(gym_space.attr("spaces")), encoding, space_relative_precision, feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to interpret Gym dict space's subspaces as a python dict");
        return std::unique_ptr<GymSpace>();
    } catch (const std::exception& e) {
        py::print("ERROR: " + std::string(e.what()));
        return std::unique_ptr<GymSpace>();
    }
}


unsigned int DictSpace::get_number_of_tracked_atoms() const {
    unsigned int total = 0;
    for (const auto& s : spaces_) {
        total += s.second->get_number_of_tracked_atoms();
    }
    return total;
}


void DictSpace::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::dict>(element)) {
            py::print("ERROR: Gym dict space element not of type 'dict'");
            return;
        }
        py::dict d = py::cast<py::dict>(element);
        for (auto i : d) {
            auto e = spaces_.find(py::cast<py::str>(i.first));
            if (e == spaces_.end()) {
                py::print("ERROR: key '" + std::string(py::cast<py::str>(i.first)) + "' not in the Gym dict space key list");
                return;
            }
            e->second->convert_element_to_feature_atoms(py::cast<py::object>(i.second), feature_atoms);
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym dict space element to a feature atom vector");
    }
}


py::object DictSpace::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    py::dict result = py::dict();
    for (const auto& s : spaces_) {
        result[py::str(s.first)] = s.second->convert_feature_atoms_to_element(feature_atoms);
    }
    return result;
}


void DictSpace::enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    enumerate(spaces_.begin(), f, feature_atoms);
}


void DictSpace::enumerate(std::map<std::string, std::unique_ptr<GymSpace>>::const_iterator current_space,
                          const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    current_space->second->enumerate([this,&current_space,&f,&feature_atoms](const std::vector<int>& v)->void {
        std::map<std::string, std::unique_ptr<GymSpace>>::const_iterator next_space = current_space;
        if ((++next_space) != spaces_.end()) {
            enumerate(next_space, f, feature_atoms);
        } else {
            f(v); // v == feature_atoms
        }
    }, feature_atoms);
}


// -- DiscreteSpace ---

template <>
DiscreteSpace<GymSpace::ENCODING_BYTE_VECTOR>::DiscreteSpace(const py::int_& n, unsigned int feature_atom_vector_begin)
: GymSpace(feature_atom_vector_begin), n_(n) {
    number_of_feature_atoms_ = sizeof(std::int64_t);
}


template <>
DiscreteSpace<GymSpace::ENCODING_VARIABLE_VECTOR>::DiscreteSpace(const py::int_& n, unsigned int feature_atom_vector_begin)
: GymSpace(feature_atom_vector_begin), n_(n) {
    number_of_feature_atoms_ = 1;
}


template <GymSpace::Encoding E>
std::unique_ptr<GymSpace> DiscreteSpace<E>::import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "n")) {
            py::print("ERROR: Gym discrete space missing attribute 'n'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::int_>(gym_space.attr("n"))) {
            py::print("ERROR: Gym discrete space's 'n' not of type 'int'");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<DiscreteSpace<E>>(py::cast<py::int_>(gym_space.attr("n")), feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to interpret Gym discrete space size as a python int");
        return std::unique_ptr<GymSpace>();
    }
}


template <>
unsigned int DiscreteSpace<GymSpace::ENCODING_BYTE_VECTOR>::get_number_of_tracked_atoms() const {
    return number_of_feature_atoms_ * 256;
}


template <>
unsigned int DiscreteSpace<GymSpace::ENCODING_VARIABLE_VECTOR>::get_number_of_tracked_atoms() const {
    return n_;
}


template <>
void DiscreteSpace<GymSpace::ENCODING_BYTE_VECTOR>::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::int_>(element)) {
            py::print("ERROR: Gym discrete space element not of type 'int'");
            return;
        }
        std::int64_t myint = py::cast<py::int_>(element);
        unsigned char* byte_array = reinterpret_cast<unsigned char *>(&myint);
        for (unsigned i = 0 ; i < sizeof(std::int64_t) ; i++) {
            feature_atoms[feature_atom_vector_begin_ + i] = byte_array[i];
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym discrete space element to a feature atom vector");
    }
}


template <>
void DiscreteSpace<GymSpace::ENCODING_VARIABLE_VECTOR>::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::int_>(element)) {
            py::print("ERROR: Gym discrete space element not of type 'int'");
            return;
        }
        feature_atoms[feature_atom_vector_begin_] = py::cast<py::int_>(element);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym discrete space element to a feature atom vector");
    }
}


template <>
py::object DiscreteSpace<GymSpace::ENCODING_BYTE_VECTOR>::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    unsigned char* byte_array = new unsigned char[sizeof(std::int64_t)];
    for (unsigned int i = 0 ; i < sizeof(std::int64_t) ; i++) {
        byte_array[i] = feature_atoms[feature_atom_vector_begin_ + i];
    }
    std::int64_t myint = *reinterpret_cast<std::int64_t*>(byte_array);
    delete[] byte_array;
    return py::int_(myint);
}


template <>
py::object DiscreteSpace<GymSpace::ENCODING_VARIABLE_VECTOR>::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    return py::int_(feature_atoms[feature_atom_vector_begin_]);
}


template <>
void DiscreteSpace<GymSpace::ENCODING_BYTE_VECTOR>::enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    for (std::int64_t i = 0 ; i < n_ ; i++) {
        unsigned char* byte_array = reinterpret_cast<unsigned char *>(&i);
        for (unsigned int j = 0 ; j < sizeof(std::int64_t) ; j++) {
            feature_atoms[feature_atom_vector_begin_ + j] = byte_array[j];
        }
        f(feature_atoms);
    }
}


template <>
void DiscreteSpace<GymSpace::ENCODING_VARIABLE_VECTOR>::enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    for (unsigned int i = 0 ; i < n_ ; i++) {
        feature_atoms[feature_atom_vector_begin_] = i;
        f(feature_atoms);
    }
}


// --- MultiBinarySpace ---

template <>
MultiBinarySpace<GymSpace::ENCODING_BYTE_VECTOR>::MultiBinarySpace(const py::int_& n, unsigned int feature_atom_vector_begin)
: GymSpace(feature_atom_vector_begin), n_(n) {
    number_of_feature_atoms_ = (n_ % 8 == 0) ? (n_ / 8) : ((n_ / 8) + 1);
}


template <>
MultiBinarySpace<GymSpace::ENCODING_VARIABLE_VECTOR>::MultiBinarySpace(const py::int_& n, unsigned int feature_atom_vector_begin)
: GymSpace(feature_atom_vector_begin), n_(n) {
    number_of_feature_atoms_ = n_;
}


template <GymSpace::Encoding E>
std::unique_ptr<GymSpace> MultiBinarySpace<E>::import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "n")) {
            py::print("ERROR: Gym multi-binary space missing attribute 'n'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::int_>(gym_space.attr("n"))) {
            py::print("ERROR: Gym multi-binary space's 'n' not of type 'int'");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<MultiBinarySpace<E>>(py::cast<py::int_>(gym_space.attr("n")), feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to interpret Gym multi-binary space dimension as a python int");
        return std::unique_ptr<GymSpace>();
    }
}


template <>
unsigned int MultiBinarySpace<GymSpace::ENCODING_BYTE_VECTOR>::get_number_of_tracked_atoms() const {
    return number_of_feature_atoms_ * 256;
}


template <>
unsigned int MultiBinarySpace<GymSpace::ENCODING_VARIABLE_VECTOR>::get_number_of_tracked_atoms() const {
    return n_;
}


template <>
void MultiBinarySpace<GymSpace::ENCODING_BYTE_VECTOR>::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array>(element)) {
            py::print("ERROR: Gym multi-binary space element not a numpy array");
        }
        py::array_t<std::int8_t> celement;
        if (!py::isinstance<py::array_t<std::int8_t>>(element)) {
            //py::print("WARNING: Gym multi-binary space element not of int8 'dtype' (requires a copy of the element to the int8 'dtype')");
            celement = py::cast<py::array_t<std::int8_t>>(element.attr("astype")(py::module::import("numpy").attr("dtype")("int8")));
        } else {
            celement = py::cast<py::array_t<std::int8_t>>(element);
        }
        py::buffer_info buf = celement.request();
        if (buf.size != n_) {
            py::print("ERROR: Gym multi-binary space element numpy array not of same dimension as the space's dimension");
            return;
        }
        unsigned int current_bit = 0;
        for (unsigned int i = 0 ; i < number_of_feature_atoms_ ; i++) {
            unsigned char c = 0;
            for (unsigned int j = 0 ; j < 8 ; j++) {
                if (((current_bit + j) < buf.size) && (((std::int8_t*) buf.ptr)[current_bit + j] == 1)) {
                    c |= (1 << j);
                }
            }
            feature_atoms[feature_atom_vector_begin_ + i] = c;
            current_bit += 8;
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym multi-binary space element to a feature atom vector");
    }
}


template <>
void MultiBinarySpace<GymSpace::ENCODING_VARIABLE_VECTOR>::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array>(element)) {
            py::print("ERROR: Gym multi-binary space element not a numpy array");
        }
        py::array_t<std::int8_t> celement;
        if (!py::isinstance<py::array_t<std::int8_t>>(element)) {
            //py::print("WARNING: Gym multi-binary space element not of int8 'dtype' (requires a copy of the element to the int8 'dtype')");
            celement = py::cast<py::array_t<std::int8_t>>(element.attr("astype")(py::module::import("numpy").attr("dtype")("int8")));
        } else {
            celement = py::cast<py::array_t<std::int8_t>>(element);
        }
        py::buffer_info buf = celement.request();
        if (buf.size != n_) {
            py::print("ERROR: Gym multi-binary space element numpy array not of same dimension as the space's dimension");
            return;
        }
        for (unsigned int i = 0 ; i < n_ ; i++) {
            feature_atoms[feature_atom_vector_begin_ + i] = ((std::int8_t*) buf.ptr)[i];
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym multi-binary space element to a feature atom vector");
    }
}


template <>
py::object MultiBinarySpace<GymSpace::ENCODING_BYTE_VECTOR>::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    py::array_t<std::int8_t> result = py::array_t<std::int8_t>(n_);
    py::buffer_info buf = result.request();
    unsigned int current_bit = 0;
    for (unsigned int i = 0 ; i < number_of_feature_atoms_ ; i++) {
        unsigned char c = feature_atoms[feature_atom_vector_begin_ + i];
        for (unsigned int j = 0 ; j < 8 ; j++) {
            if ((current_bit + j) < buf.size) {
                ((std::int8_t*) buf.ptr)[current_bit + j] = (std::int8_t) c % 2;
                c /= 2;
            } else {
                break;
            }
        }
        current_bit += 8;
    }
    return result;
}


template <>
py::object MultiBinarySpace<GymSpace::ENCODING_VARIABLE_VECTOR>::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    py::array_t<std::int8_t> result = py::array_t<std::int8_t>(n_);
    py::buffer_info buf = result.request();
    for (unsigned int i = 0 ; i < n_ ; i++) {
        ((std::int8_t*) buf.ptr)[i] = feature_atoms[feature_atom_vector_begin_ + i];
    }
    return result;
}


template <>
void MultiBinarySpace<GymSpace::ENCODING_BYTE_VECTOR>::enumerate_byte(unsigned int current_bit, unsigned char current_byte,
                                                                      const std::function<void(const std::vector<int>&)>& f,
                                                                      std::vector<int>& feature_atoms) const {
    for (auto e : {0, 1}) {
        unsigned int byte_count = current_bit % 8;
        unsigned char updated_byte = current_byte;
        if (e == 1) {
            updated_byte |= (1 << byte_count);
        }
        if ((byte_count == 7) || ((current_bit + 1) == n_)) {
            feature_atoms[feature_atom_vector_begin_ + (current_bit / 8)] = updated_byte;
            updated_byte = 0;
        }
        if ((current_bit + 1) < n_) {
            enumerate_byte(current_bit + 1, updated_byte, f, feature_atoms);
        } else {
            f(feature_atoms);
        }
    }
}


template <>
void MultiBinarySpace<GymSpace::ENCODING_VARIABLE_VECTOR>::enumerate_variable(unsigned int current_bit,
                                                                              const std::function<void(const std::vector<int>&)>& f,
                                                                              std::vector<int>& feature_atoms) const {
    for (auto e : {0, 1}) {
        feature_atoms[feature_atom_vector_begin_ + current_bit] = e;
        if ((current_bit + 1) < n_) {
            enumerate_variable(current_bit + 1, f, feature_atoms);
        } else {
            f(feature_atoms);
        }
    }
}


template <>
void MultiBinarySpace<GymSpace::ENCODING_BYTE_VECTOR>::enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    enumerate_byte(0, 0, f, feature_atoms);
}


template <>
void MultiBinarySpace<GymSpace::ENCODING_VARIABLE_VECTOR>::enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    enumerate_variable(0, f, feature_atoms);
}


// --- MultiDiscreteSpace ---

template <>
MultiDiscreteSpace<GymSpace::ENCODING_BYTE_VECTOR>::MultiDiscreteSpace(const py::array_t<unsigned int>& nvec, unsigned int feature_atom_vector_begin)
try : GymSpace(feature_atom_vector_begin), nvec_(nvec) {
    if (nvec_.ndim() != 1) {
        throw std::domain_error("Gym multi-discrete space dimension different from 1");
    }
    number_of_feature_atoms_ = nvec_.size() * sizeof(std::int64_t);
} catch (const std::exception& e) {
    throw;
}


template <>
MultiDiscreteSpace<GymSpace::ENCODING_VARIABLE_VECTOR>::MultiDiscreteSpace(const py::array_t<unsigned int>& nvec, unsigned int feature_atom_vector_begin)
try : GymSpace(feature_atom_vector_begin), nvec_(nvec) {
    if (nvec_.ndim() != 1) {
        throw std::domain_error("Gym multi-discrete space dimension different from 1");
    }
    number_of_feature_atoms_ = nvec_.size();
} catch (const std::exception& e) {
    throw;
}


template <GymSpace::Encoding E>
std::unique_ptr<GymSpace> MultiDiscreteSpace<E>::import_from_python(const py::object& gym_space, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "nvec")) {
            py::print("ERROR: Gym multi-discrete space missing attribute 'nvec'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::array_t<std::int64_t>>(gym_space.attr("nvec"))) {
            py::print("ERROR: Gym multi-discrete space's 'nvec' not a numpy array");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<MultiDiscreteSpace<E>>(py::cast<py::array_t<std::int64_t>>(gym_space.attr("nvec")), feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to interpret Gym multi-discrete space dimension vector as a python array of positive integers");
        return std::unique_ptr<GymSpace>();
    }
}


template <>
unsigned int MultiDiscreteSpace<GymSpace::ENCODING_BYTE_VECTOR>::get_number_of_tracked_atoms() const {
    return number_of_feature_atoms_ * 256;
}


template <>
unsigned int MultiDiscreteSpace<GymSpace::ENCODING_VARIABLE_VECTOR>::get_number_of_tracked_atoms() const {
    py::buffer_info buf = nvec_.request();
    unsigned int total = 0;
    for (unsigned int i = 0 ; i < nvec_.size() ; i++) {
        total += ((std::int64_t*) buf.ptr)[i];
    }
    return total;
}


template <>
void MultiDiscreteSpace<GymSpace::ENCODING_BYTE_VECTOR>::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array>(element)) {
            py::print("ERROR: Gym multi-discrete space element not a numpy array");
        }
        py::array_t<std::int64_t> celement;
        if (!py::isinstance<py::array_t<std::int64_t>>(element)) {
            //py::print("WARNING: Gym multi-discrete space element not of int8 'dtype' (requires a copy of the element to the int64 'dtype')");
            celement = py::cast<py::array_t<std::int64_t>>(element.attr("astype")(py::module::import("numpy").attr("dtype")("int64")));
        } else {
            celement = py::cast<py::array_t<std::int64_t>>(element);
        }
        py::buffer_info buf = celement.request();
        if (buf.size != nvec_.size()) {
            py::print("ERROR: Gym multi-discrete space element numpy array not of same dimension as the space's dimension");
            return;
        }
        for (unsigned int i = 0 ; i < nvec_.size() ; i++) {
            unsigned char* byte_array = reinterpret_cast<unsigned char*>(&(((std::int64_t*) buf.ptr)[i]));
            for (unsigned int j = 0 ; j < sizeof(std::int64_t) ; j++) {
                feature_atoms[feature_atom_vector_begin_ + (i * sizeof(std::int64_t)) + j] = byte_array[j];
            }
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym multi-discrete space element to a feature atom vector");
    }
}


template <>
void MultiDiscreteSpace<GymSpace::ENCODING_VARIABLE_VECTOR>::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::array>(element)) {
            py::print("ERROR: Gym multi-discrete space element not a numpy array");
        }
        py::array_t<std::int64_t> celement;
        if (!py::isinstance<py::array_t<std::int64_t>>(element)) {
            //py::print("WARNING: Gym multi-discrete space element not of int8 'dtype' (requires a copy of the element to the int64 'dtype')");
            celement = py::cast<py::array_t<std::int64_t>>(element.attr("astype")(py::module::import("numpy").attr("dtype")("int64")));
        } else {
            celement = py::cast<py::array_t<std::int64_t>>(element);
        }
        py::buffer_info buf = celement.request();
        if (buf.size != nvec_.size()) {
            py::print("ERROR: Gym multi-discrete space element numpy array not of same dimension as the space's dimension");
            return;
        }
        for (unsigned int i = 0 ; i < nvec_.size() ; i++) {
            feature_atoms[feature_atom_vector_begin_ + i] = ((std::int64_t*) buf.ptr)[i];
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym multi-discrete space element to a feature atom vector");
    }
}


template <>
py::object MultiDiscreteSpace<GymSpace::ENCODING_BYTE_VECTOR>::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    py::array_t<std::int64_t> result = py::array_t<std::int64_t>(nvec_.size());
    py::buffer_info buf = result.request();
    unsigned char* byte_array = new unsigned char[sizeof(std::int64_t)];
    for (unsigned int i = 0 ; i < nvec_.size() ; i++) {
        for (unsigned int j = 0 ; j < sizeof(std::int64_t) ; j++) {
            byte_array[j] = feature_atoms[feature_atom_vector_begin_ + (i * sizeof(std::int64_t)) + j];
        }
        ((std::int64_t*) buf.ptr)[i] = *reinterpret_cast<std::int64_t*>(byte_array);
    }
    delete[] byte_array;
    return result;
}


template <>
py::object MultiDiscreteSpace<GymSpace::ENCODING_VARIABLE_VECTOR>::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    py::array_t<std::int64_t> result = py::array_t<std::int64_t>(nvec_.size());
    py::buffer_info buf = result.request();
    for (unsigned int i = 0 ; i < nvec_.size() ; i++) {
        ((std::int64_t*) buf.ptr)[i] = feature_atoms[feature_atom_vector_begin_ + i];
    }
    return result;
}


template <GymSpace::Encoding E>
void MultiDiscreteSpace<E>::enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    enumerate(0, nvec_.request(), f, feature_atoms);
}


template <>
void MultiDiscreteSpace<GymSpace::ENCODING_BYTE_VECTOR>::enumerate(unsigned int current_item, const py::buffer_info& buf,
                                                                   const std::function<void(const std::vector<int>&)>& f,
                                                                   std::vector<int>& feature_atoms) const {
    for (unsigned int i = 0 ; i < ((std::int64_t*) buf.ptr)[current_item] ; i++) {
        unsigned char* byte_array = reinterpret_cast<unsigned char*>(&i);
        for (unsigned int j = 0 ; j < sizeof(std::int64_t) ; j++) {
            feature_atoms[feature_atom_vector_begin_ + (current_item * sizeof(std::int64_t)) + j] = byte_array[j];
        }
        if ((current_item + 1) < nvec_.size()) {
            enumerate(current_item + 1, buf, f, feature_atoms);
        } else {
            f(feature_atoms);
        }
    }
}


template <>
void MultiDiscreteSpace<GymSpace::ENCODING_VARIABLE_VECTOR>::enumerate(unsigned int current_item, const py::buffer_info& buf,
                                                                       const std::function<void(const std::vector<int>&)>& f,
                                                                       std::vector<int>& feature_atoms) const {
    for (unsigned int i = 0 ; i < ((std::int64_t*) buf.ptr)[current_item] ; i++) {
        feature_atoms[feature_atom_vector_begin_ + current_item] = i;
        if ((current_item + 1) < nvec_.size()) {
            enumerate(current_item + 1, buf, f, feature_atoms);
        } else {
            f(feature_atoms);
        }
    }
}


// --- TupleSpace ---

TupleSpace::TupleSpace(const py::tuple& spaces, Encoding encoding, double space_relative_precision, unsigned int feature_atom_vector_begin)
: GymSpace(feature_atom_vector_begin) {
    number_of_feature_atoms_ = 0;
    for (auto s : spaces) {
        spaces_.push_back(std::move(GymSpace::import_from_python(py::cast<py::object>(s), encoding, space_relative_precision, feature_atom_vector_begin + number_of_feature_atoms_)));
        number_of_feature_atoms_ += spaces_.back()->get_number_of_feature_atoms();
    }
}


std::unique_ptr<GymSpace> TupleSpace::import_from_python(const py::object& gym_space, Encoding encoding, double space_relative_precision, unsigned int feature_atom_vector_begin) {
    try {
        if (!py::hasattr(gym_space, "spaces")) {
            py::print("ERROR: Gym tuple space missing attribute 'spaces'");
            return std::unique_ptr<GymSpace>();
        }
        if (!py::isinstance<py::tuple>(gym_space.attr("spaces"))) {
            py::print("ERROR: Gym tuple space's 'spaces' not of type 'tuple'");
            return std::unique_ptr<GymSpace>();
        }
        return std::make_unique<TupleSpace>(py::cast<py::tuple>(gym_space.attr("spaces")), encoding, space_relative_precision, feature_atom_vector_begin);
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to interpret Gym tuple space's subspaces as a python tuple");
        return std::unique_ptr<GymSpace>();
    }
}


unsigned int TupleSpace::get_number_of_tracked_atoms() const {
    unsigned int total = 0;
    for (const auto& s : spaces_) {
        total += s->get_number_of_tracked_atoms();
    }
    return total;
}


void TupleSpace::convert_element_to_feature_atoms(const py::object& element, std::vector<int>& feature_atoms) const {
    try {
        if (!py::isinstance<py::tuple>(element)) {
            py::print("ERROR: Gym tuple space element not of type 'tuple'");
            return;
        }
        py::tuple t = py::cast<py::tuple>(element);
        unsigned int i = 0;
        for (auto s = spaces_.begin() ; s != spaces_.end() ; s++) {
            if (i >= t.size()) {
                py::print("ERROR: Gym tuple space element size less than the space's tuple size");
                return;
            }
            (*s)->convert_element_to_feature_atoms(t[i], feature_atoms);
            i++;
        }
        if (i < t.size()) {
            py::print("ERROR: Gym tuple space element size larger than the space's tuple size");
            return;
        }
    } catch (const py::cast_error& e) {
        py::print("ERROR: Python casting error (python object of unexpected type) when trying to convert Gym tuple space element to a feature atom vector");
    }
}


py::object TupleSpace::convert_feature_atoms_to_element(const std::vector<int>& feature_atoms) const {
    py::tuple result = py::tuple(spaces_.size());
    unsigned int i = 0;
    for (const auto& s : spaces_) {
        result[i] = s->convert_feature_atoms_to_element(feature_atoms);
        i++;
    }
    return result;
}


void TupleSpace::enumerate(const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    enumerate(spaces_.begin(), f, feature_atoms);
}


void TupleSpace::enumerate(std::list<std::unique_ptr<GymSpace>>::const_iterator current_space,
                           const std::function<void(const std::vector<int>&)>& f, std::vector<int>& feature_atoms) const {
    (*current_space)->enumerate([this,&current_space,&f,&feature_atoms](const std::vector<int>& v)->void {
        std::list<std::unique_ptr<GymSpace>>::const_iterator next_space = current_space;
        if ((++next_space) != spaces_.end()) {
            enumerate(next_space, f, feature_atoms);
        } else {
            f(v); // v == feature_atoms
        }
    }, feature_atoms);
}
