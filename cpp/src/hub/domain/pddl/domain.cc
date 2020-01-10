/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ostream>
#include <sstream>
#include <stack>
#include <algorithm>
#include <iterator>

#include "domain.hh"
#include "type.hh"
#include "object.hh"
#include "variable.hh"
#include "predicate.hh"
#include "function.hh"

using namespace skdecide::pddl;

Domain::Domain() {
    add_type(Type::object());
    add_type(Type::number());
}


void Domain::set_name(const std::string& name) {
    Identifier::set_name(name);
}


// REQUIREMENTS

void Domain::set_requirements(const Requirements& requirements) {
    _requirements = requirements;
}


const Requirements& Domain::get_requirements() const {
    return _requirements;
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
        std::stack<Domain::TypeSet> levels;
        Domain::TypeSet frontier = d.get_types();
        while (!frontier.empty()) {
            Domain::TypeSet new_frontier;
            for (const auto& t: frontier) {
                for (const auto& p: t->get_types()) {
                    new_frontier.insert(p);
                }
            }
            Domain::TypeSet l;
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

    if (!d.get_objects().empty()) {
        o << "(:constants";
        for (const auto& c : d.get_objects()) {
            o << " " << *c;
        }
        o << ")" << std::endl;
    }

    if (!d.get_predicates().empty()) {
        o << "(:predicates";
        for (const auto& p : d.get_predicates()) {
            o << " " << *p;
        }
        o << ")" << std::endl;
    }

    if (!d.get_functions().empty()) {
        o << "(:functions";
        for (const auto& f : d.get_functions()) {
            o << " " << *f;
        }
        o << ")" << std::endl;
    }

    o << ")" << std::endl;
    
    return o;
}
