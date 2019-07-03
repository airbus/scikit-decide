#include <ostream>
#include <sstream>
#include <stack>
#include <algorithm>

#include "domain.hh"

using namespace airlaps::pddl;

Domain::Domain() {
    add_type(Type::object());
    add_type(Type::number());
}


void Domain::set_name(const std::string& name) {
    _name = name;
}


const std::string& Domain::get_name() const {
    return _name;
}


// REQUIREMENTS

void Domain::set_requirements(const Requirements& requirements) {
    _requirements = requirements;
}


const Requirements& Domain::get_requirements() const {
    return _requirements;
}


// TYPES

const Type::Ptr& Domain::add_type(const Type::Ptr& t) {
    if (!_types.emplace(t).second) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t->get_name() +
                               "' already in the set of types of domain '" +
                               this->get_name() +
                               "'");
    } else {
        return t;
    }
}


const Type::Ptr& Domain::add_type(const std::string& t) {
    std::pair<Type::Set::iterator, bool> i = _types.emplace(std::make_shared<Type>(t));
    if (!i.second) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t +
                               "' already in the set of types of domain '" +
                               this->get_name() +
                               "'");
    } else {
        return *i.first;
    }
}


void Domain::remove_type(const Type::Ptr& t) {
    if (_types.erase(t) == 0) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t->get_name() +
                               "' not in the set of types of domain '" +
                               this->get_name() +
                               "'");
    }
}


void Domain::remove_type(const std::string& t) {
    remove_type(std::make_shared<Type>(t));
}


const Type::Ptr& Domain::get_type(const std::string& t) const {
    Type::Set::const_iterator i = _types.find(std::make_shared<Type>(t));
    if (i == _types.end()) {
        throw std::logic_error("AIRLAPS exception: type '" +
                               t +
                               "' not in the set of types of domain '" +
                               this->get_name() +
                               "'");
    } else {
        return *i;
    }
}


const Type::Set& Domain::get_types() const {
    return _types;
}


// CONSTANTS

const Object::Ptr& Domain::add_constant(const Object::Ptr& o) {
    if (!_constants.emplace(o).second) {
        throw std::logic_error("AIRLAPS exception: object '" +
                               o->get_name() +
                               "' already in the set of constants of domain '" +
                               this->get_name() +
                               "'");
    } else {
        return o;
    }
}


const Object::Ptr& Domain::add_constant(const std::string& o) {
    std::pair<Object::Set::iterator, bool> i = _constants.emplace(std::make_shared<Object>(o));
    if (!i.second) {
        throw std::logic_error("AIRLAPS exception: object '" +
                               o +
                               "' already in the set of constants of domain '" +
                               this->get_name() +
                               "'");
    } else {
        return *i.first;
    }
}


void Domain::remove_constant(const Object::Ptr& o) {
    if (_constants.erase(o) == 0) {
        throw std::logic_error("AIRLAPS exception: constant '" +
                               o->get_name() +
                               "' not in the set of constants of domain '" +
                               this->get_name() +
                               "'");
    }
}


void Domain::remove_constant(const std::string& o) {
    remove_constant(std::make_shared<Object>(o));
}


const Object::Ptr& Domain::get_constant(const std::string& o) const {
    Object::Set::const_iterator i = _constants.find(std::make_shared<Object>(o));
    if (i == _constants.end()) {
        throw std::logic_error("AIRLAPS exception: object '" +
                               o +
                               "' not in the set of constants of domain '" +
                               this->get_name() +
                               "'");
    } else {
        return *i;
    }
}


const Object::Set& Domain::get_constants() const {
    return _constants;
}


std::string Domain::print() const {
    std::ostringstream o;
    o << *this;
    return o.str();
}


std::ostream& operator<<(std::ostream& o, const Domain& d) {
    o << "(define (domain " << d.get_name() << ")" << std::endl;

    o << d.get_requirements() << std::endl ;

    if (!d.get_types().empty()) {
        // Extract the types in correct order, i.e. highest types first
        std::stack<Type::Set> levels;
        Type::Set frontier = d.get_types();
        while (!frontier.empty()) {
            Type::Set new_frontier;
            for (const auto& t: frontier) {
                for (const auto& p: t->get_parents()) {
                    new_frontier.insert(p);
                }
            }
            Type::Set l;
            std::set_difference(frontier.begin(), frontier.end(),
                                new_frontier.begin(), new_frontier.end(),
                                std::inserter(l, l.begin()));
            levels.push(l);
            frontier = new_frontier;
        }
        o << "(:types";
        while (!levels.empty()) {
            for (auto& t : levels.top()) {
                if (t != Type::object() && t != Type::number()) {
                    o << " " << *t;
                }
            }
            levels.pop();
        }
        o << ")" << std::endl;
    }

    if (!d.get_constants().empty()) {
        o << "(:constants";
        for (const auto& c : d.get_constants()) {
            o << " " << *c;
        }
        o << ")" << std::endl;
    }

    o << ")" << std::endl;
    
    return o;
}
