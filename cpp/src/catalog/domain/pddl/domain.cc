#include <ostream>
#include <sstream>
#include "domain.hh"

using namespace airlaps::pddl;

void Domain::set_name(const std::string& name) {
    _name = name;
}


const std::string& Domain::get_name() const {
    return _name;
}


void Domain::set_requirements(const Requirements& requirements) {
    _requirements = requirements;
}


const Requirements& Domain::get_requirements() const {
    return _requirements;
}


void Domain::add_type(const Type& t) {
    if (!_types.emplace(t).second) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t.get_name() +
                               "' already in the set of types of domain '" +
                               this->get_name() +
                               "'");
    }
}


Type& Domain::add_type(const std::string& t) {
    std::pair<TypeSet::iterator, bool> i = _types.insert(t);
    if (!i.second) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t +
                               "' already in the set of types of domain '" +
                               this->get_name() +
                               "'");
    } else {
        return const_cast<Type&>(*i.first); // the name of a type cannot be changed after construction of the type so we are safe
    }
}


void Domain::remove_type(const Type& t) {
    if (_types.erase(t) == 0) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t.get_name() +
                               "' not in the set of types of domain '" +
                               this->get_name() +
                               "'");
    }
}


void Domain::remove_type(const std::string& t) {
    remove_type(Type(t));
}


const Domain::TypeSet& Domain::get_types() const {
    return _types;
}


std::string Domain::print() const {
    std::ostringstream o;
    o << *this;
    return o.str();
}


std::ostream& operator<<(std::ostream& o, const Domain& d) {
    o << "(define (domain " << d.get_name() << ")" << std::endl <<
         d.get_requirements() ;
    if (!d.get_types().empty()) {
        o << "(:types";
        for (const auto& t : d.get_types()) {
            o << " " << t;
        }
        o << ")";
    }
    o << ")" << std::endl;
    return o;
}
