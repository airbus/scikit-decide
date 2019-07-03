#include <exception>
#include <sstream>

#include "type.hh"
#include "symbol_container_helper.hh"

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
    SymbolContainerHelper::add(this, "type", _parents, "parent types", t, "type");
}


void Type::remove_parent(const Ptr& t) {
    SymbolContainerHelper::remove(this, "type", _parents, "parent types", t, "type");
}


const Type::Ptr& Type::get_parent(const std::string& t) const {
    return SymbolContainerHelper::get(this, "type", _parents, "parent types", t, "type");
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
