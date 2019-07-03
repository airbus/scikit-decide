#include <exception>
#include <sstream>
#include "object.hh"

using namespace airlaps::pddl;


Object::Object(const std::string& name)
    : _name(name) {
}


Object::Object(const Object& other)
: _name(other._name), _types(other._types) {

}


Object& Object::operator=(const Object& other) {
    this->_name = other._name;
    this->_types = other._types;
    return *this;
}


const std::string& Object::get_name() const {
    return _name;
}


void Object::add_type(const Type::Ptr& t) {
    if (!_types.insert(t).second) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t->get_name() +
                               "' already in the set of types of object '" +
                               this->get_name() +
                               "'");
    }
}


void Object::remove_type(const Type::Ptr& t) {
    if (_types.erase(t) == 0) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t->get_name() +
                               "' not in the set of types of object '" +
                               this->get_name() +
                               "'");
    }
}


const Type::Ptr& Object::get_type(const std::string& t) const {
    Type::Set::const_iterator i = _types.find(std::make_shared<Type>(t));
    if (i == _types.end()) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t +
                               "' not in the set of types of object '" +
                               this->get_name() +
                               "'");
    } else {
        return *i;
    }
}


const Type::Set& Object::get_types() const {
    return _types;
}


std::string Object::print() const {
    std::ostringstream o;
    o << *this;
    return o.str();
}


std::ostream& operator<< (std::ostream& o, const Object& ob) {
    o << ob.get_name();
    if (!ob.get_types().empty()) {
        o << " - ";
        if (ob.get_types().size() > 1) {
            o << "(either";
            for (const auto& t : ob.get_types()) {
                o << " " << t->get_name();
            }
            o << ")";
        } else {
            o << (*ob.get_types().begin())->get_name();
        }
    }
    return o;
}
