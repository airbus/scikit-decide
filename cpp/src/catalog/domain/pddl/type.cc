#include <exception>
#include <sstream>
#include "type.hh"

using namespace airlaps::pddl;

const Type::Ptr Type::_object = std::make_shared<Type>("object");
const Type::Ptr Type::_number = std::make_shared<Type>("number");


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


void Type::add_parent(const Ptr& t) {
    if (!_parents.insert(t).second) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t->get_name() +
                               "' already in the set of parent types of type '" +
                               this->get_name() +
                               "'");
    }
}


void Type::remove_parent(const Ptr& t) {
    if (_parents.erase(t) == 0) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t->get_name() +
                               "' not in the set of parent types of type '" +
                               this->get_name() +
                               "'");
    }
}


const Type::Ptr& Type::get_parent(const std::string& t) const {
    Set::const_iterator i = _parents.find(std::make_shared<Type>(t));
    if (i == _parents.end()) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t +
                               "' not in the set of parent types of type '" +
                               this->get_name() +
                               "'");
    } else {
        return *i;
    }
}


const Type::Set& Type::get_parents() const {
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
        o << " - ";
        if (t.get_parents().size() > 1) {
            o << "(either";
            for (const auto& p : t.get_parents()) {
                o << " " << p->get_name();
            }
            o << ")";
        } else {
            o << (*t.get_parents().begin())->get_name();
        }
    }
    return o;
}
