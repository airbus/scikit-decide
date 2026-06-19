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
#include "impl/associative_container_impl.hh"
#include "type.hh"
#include "object.hh"
#include "variable.hh"
#include "predicate.hh"
#include "derived_predicate.hh"
#include "function.hh"
#include "class.hh"

using namespace skdecide::pddl;

Domain::Domain(const std::string &name) : Identifier(name) {}

Domain::~Domain() {}

void Domain::set_requirements(const Requirements::Ptr &requirements) {
  if (_requirements && _requirements->has_typing()) {
    remove_type(Type::object());
  }
  if (_requirements && _requirements->has_fluents()) {
    remove_type(Type::number());
  }
  _requirements = requirements;
  if (requirements->has_typing()) {
    add_type(Type::object());
  }
  if (requirements->has_fluents()) {
    add_type(Type::number());
  }
}

const Requirements::Ptr &Domain::get_requirements() const {
  return _requirements;
}

void Domain::set_constraints(const Formula::Ptr &constraints) {
  _constraints = constraints;
}

const Formula::Ptr &Domain::get_constraints() const { return _constraints; }

std::string Domain::print() const {
  std::ostringstream o;
  o << *this;
  return o.str();
}

namespace skdecide {
namespace pddl {
std::ostream &operator<<(std::ostream &o, const Domain &d) {
  o << "(define (domain " << d.get_name() << ")" << std::endl;

  if (d.get_requirements()) {
    o << *(d.get_requirements()) << std::endl;
  }

  if (!d.get_types().empty()) {
    // Extract the types in correct order, i.e. highest types first
    std::stack<Domain::TypeSet> levels;
    Domain::TypeSet frontier = d.get_types();
    while (!frontier.empty()) {
      Domain::TypeSet new_frontier;
      for (const auto &t : frontier) {
        for (const auto &p : t->get_types()) {
          new_frontier.insert(p);
        }
      }
      Domain::TypeSet l;
      for (const auto &f : frontier) {
        if (new_frontier.find(f) == new_frontier.end()) {
          l.insert(f);
        }
      }
      levels.push(l);
      frontier = new_frontier;
    }
    o << "(:types";
    while (!levels.empty()) {
      for (auto &t : levels.top()) {
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
    for (const auto &c : d.get_objects()) {
      o << " " << *c;
    }
    o << ")" << std::endl;
  }

  if (!d.get_predicates().empty()) {
    o << "(:predicates";
    for (const auto &p : d.get_predicates()) {
      o << " " << *p;
    }
    o << ")" << std::endl;
  }

  if (!d.get_functions().empty()) {
    o << "(:functions";
    for (const auto &f : d.get_functions()) {
      o << " " << *f;
    }
    o << ")" << std::endl;
  }

  if (!d.get_classes().empty()) {
    o << "(:classes";
    for (const auto &c : d.get_classes()) {
      o << " " << c->get_name();
    }
    o << ")" << std::endl;
  }

  if (d.get_constraints()) {
    o << "(:constraints " << *(d.get_constraints()) << ")" << std::endl;
  }

  for (const auto &a : d.get_actions()) {
    o << *a << std::endl;
  }

  for (const auto &e : d.get_events()) {
    o << *e << std::endl;
  }

  for (const auto &p : d.get_actions()) {
    o << *p << std::endl;
  }

  for (const auto &da : d.get_durative_actions()) {
    o << *da << std::endl;
  }

  for (const auto &dp : d.get_derived_predicates()) {
    o << *dp << std::endl;
  }

  for (const auto &c : d.get_classes()) {
    o << *c << std::endl;
  }

  o << ")" << std::endl;

  return o;
}
} // namespace pddl
} // namespace skdecide
