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