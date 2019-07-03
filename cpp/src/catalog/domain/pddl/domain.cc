#include <ostream>
#include <sstream>
#include <stack>
#include <algorithm>

#include "domain.hh"
#include "symbol_container_helper.hh"

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
    return SymbolContainerHelper::add(this, "domain", _types, "types", t, "type");
}


const Type::Ptr& Domain::add_type(const std::string& t) {
    return SymbolContainerHelper::add(this, "domain", _types, "types", t, "type");
}


void Domain::remove_type(const Type::Ptr& t) {
    return SymbolContainerHelper::remove(this, "domain", _types, "types", t, "type");
}


void Domain::remove_type(const std::string& t) {
    return SymbolContainerHelper::remove(this, "domain", _types, "types", t, "type");
}


const Type::Ptr& Domain::get_type(const std::string& t) const {
    return SymbolContainerHelper::get(this, "domain", _types, "types", t, "type");
}


const Type::Set& Domain::get_types() const {
    return _types;
}


// CONSTANTS

const Object::Ptr& Domain::add_constant(const Object::Ptr& o) {
    return SymbolContainerHelper::add(this, "domain", _constants, "constants", o, "object");
}


const Object::Ptr& Domain::add_constant(const std::string& o) {
    return SymbolContainerHelper::add(this, "domain", _constants, "constants", o, "object");
}


void Domain::remove_constant(const Object::Ptr& o) {
    SymbolContainerHelper::remove(this, "domain", _constants, "constants", o, "object");
}


void Domain::remove_constant(const std::string& o) {
    SymbolContainerHelper::remove(this, "domain", _constants, "constants", o, "object");
}


const Object::Ptr& Domain::get_constant(const std::string& o) const {
    return SymbolContainerHelper::get(this, "domain", _constants, "constants", o, "object");
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
