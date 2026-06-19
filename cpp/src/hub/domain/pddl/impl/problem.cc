/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ostream>
#include <sstream>
#include <stack>
#include <algorithm>
#include <iterator>

#include "problem.hh"
#include "assignment_effect.hh"

using namespace skdecide::pddl;

Problem::Problem(const std::string &name) : Identifier(name) {
  _initial_effect = std::make_shared<ConjunctionEffect>();
}

Problem::~Problem() {}

void Problem::set_domain(const Domain::Ptr &domain) { _domain = domain; }

const Domain::Ptr &Problem::get_domain() const { return _domain; }

void Problem::set_requirements(const Requirements::Ptr &requirements) {
  _requirements = requirements;
}

const Requirements::Ptr &Problem::get_requirements() const {
  return _requirements;
}

void Problem::set_initial_effect(const ConjunctionEffect::Ptr &initial_effect) {
  _initial_effect = initial_effect;
}

const ConjunctionEffect::Ptr &Problem::get_initial_effect() const {
  return _initial_effect;
}

void Problem::set_goal(const Formula::Ptr &goal) { _goal = goal; }

const Formula::Ptr &Problem::get_goal() const { return _goal; }

void Problem::set_constraints(const Formula::Ptr &constraints) {
  _constraints = constraints;
}

const Formula::Ptr &Problem::get_constraints() const { return _constraints; }

void Problem::set_metric(const Expression::Ptr &metric) { _metric = metric; }

const Expression::Ptr &Problem::get_metric() const { return _metric; }

void Problem::set_goal_reward(const Expression::Ptr &goal_reward) {
  _goal_reward = goal_reward;
}

const Expression::Ptr &Problem::get_goal_reward() const { return _goal_reward; }

std::string Problem::print() const {
  std::ostringstream o;
  o << *this;
  return o.str();
}

namespace skdecide {
namespace pddl {
std::ostream &operator<<(std::ostream &o, const Problem &p) {
  if (!p.get_domain()) {
    throw std::logic_error("SKDECIDE exception: undefined problem's domain");
  }

  if (!p.get_initial_effect()) {
    throw std::logic_error(
        "SKDECIDE exception: undefined problem's initial state");
  }

  if (!p.get_goal()) {
    throw std::logic_error("SKDECIDE exception: undefined problem's goal");
  }

  o << "(define (problem " << p.get_name() << ") "
    << "(:domain " << p.get_domain()->get_name() << ")" << std::endl;

  if (p.get_requirements()) {
    o << *(p.get_requirements()) << std::endl;
  }

  if (!p.get_objects().empty()) {
    o << "(:objects";
    for (const auto &c : p.get_objects()) {
      o << " " << *c;
    }
    o << ")" << std::endl;
  }

  o << "(:init" << std::endl;
  for (const auto &e : p.get_initial_effect()->get_effects()) {
    AssignEffect::Ptr ae = std::dynamic_pointer_cast<AssignEffect>(e);
    if (ae) {
      o << "   (= " << *(ae->get_function()) << " " << *(ae->get_expression())
        << ")" << std::endl;
    } else {
      o << "   " << *e << std::endl;
    }
  }
  o << ")" << std::endl;

  o << "(:goal " << *(p.get_goal()) << ")" << std::endl;

  if (p.get_constraints()) {
    o << "(:constraints " << *(p.get_constraints()) << ")" << std::endl;
  }

  if (p.get_goal_reward()) {
    o << "(:goal-reward " << *(p.get_goal_reward()) << ")" << std::endl;
  }

  if (p.get_metric()) {
    o << "(:metric " << *(p.get_metric()) << ")" << std::endl;
  }

  o << ")" << std::endl;

  return o;
}
} // namespace pddl
} // namespace skdecide
