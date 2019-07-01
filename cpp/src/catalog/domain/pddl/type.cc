#include <exception>
#include <sstream>
#include "type.hh"

using namespace airlaps::pddl;

Type::Type(const std::string& name)
    : _name(name) {
}


Type::Type(const Type& other)
: _name(other._name), _parents(other._parents) {

}


Type& Type::operator=(const Type& other) {
    this->_name = other._name;
    this->_parents = other._parents;
    return *this;
}


const std::string& Type::get_name() const {
    return _name;
}


void Type::add_parent(const Type& t) {
    if (!_parents.insert(&t).second) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t.get_name() +
                               "' already in the set of parent types of type '" +
                               this->get_name() +
                               "'");
    }
}


void Type::remove_parent(const Type& t) {
    if (_parents.erase(&t) == 0) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t.get_name() +
                               "' not in the set of parent types of type '" +
                               this->get_name() +
                               "'");
    }
}


const Type::ParentSet& Type::get_parents() const {
    return _parents;
}


std::string Type::print() const {
    std::ostringstream o;
    o << *this;
    return o.str();
}


std::ostream& operator<< (std::ostream& o, const Type& t) {
    o << t.get_name();
    if (!t.get_parents().empty()) {
        o << " -";
        if (t.get_parents().size() > 1) o << " either";
        for (const auto& p : t.get_parents()) {
            o << " " << p->get_name();
        }
    }
    return o;
}
