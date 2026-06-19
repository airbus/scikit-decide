/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_PROBLEM_HH
#define SKDECIDE_PDDL_PROBLEM_HH

#include <string>
#include <memory>
#include <unordered_set>

#include "domain.hh"
#include "identifier.hh"
#include "requirements.hh"
#include "expression.hh"
#include "aggregation_effect.hh"

namespace skdecide {

namespace pddl {

class Problem : public Identifier,
                public ObjectContainer<Problem>,
                public PreferenceContainer<Problem> {
public:
  static constexpr char class_name[] = "problem";

  typedef std::shared_ptr<Problem> Ptr;

  Problem(const std::string &name);
  virtual ~Problem();

  void set_domain(const Domain::Ptr &domain);
  const Domain::Ptr &get_domain() const;

  void set_requirements(const Requirements::Ptr &requirements);
  const Requirements::Ptr &get_requirements() const;

  void set_initial_effect(const ConjunctionEffect::Ptr &initial_effect);
  const ConjunctionEffect::Ptr &get_initial_effect() const;

  void set_goal(const Formula::Ptr &goal);
  const Formula::Ptr &get_goal() const;

  void set_constraints(const Formula::Ptr &constraints);
  const Formula::Ptr &get_constraints() const;

  void set_metric(const Expression::Ptr &metric);
  const Expression::Ptr &get_metric() const;

  void set_goal_reward(const Expression::Ptr &goal_reward);
  const Expression::Ptr &get_goal_reward() const;

  virtual std::string print() const;

  typedef ObjectContainer<Problem>::ObjectPtr ObjectPtr;
  typedef ObjectContainer<Problem>::ObjectSet ObjectSet;
  typedef PreferenceContainer<Problem>::PreferenceSet PreferenceSet;
  typedef PreferenceContainer<Problem>::PreferencePtr PreferencePtr;

private:
  Domain::Ptr _domain;
  Requirements::Ptr _requirements;
  ConjunctionEffect::Ptr _initial_effect;
  Formula::Ptr _goal;
  Formula::Ptr _constraints;
  Expression::Ptr _metric;
  Expression::Ptr _goal_reward;
};

// Problem printing operator
std::ostream &operator<<(std::ostream &o, const Problem &p);

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PROBLEM_HH
