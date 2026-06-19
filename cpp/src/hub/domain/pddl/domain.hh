/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DOMAIN_HH
#define SKDECIDE_PDDL_DOMAIN_HH

#include <string>
#include <memory>
#include <unordered_set>

#include "identifier.hh"
#include "requirements.hh"
#include "type_container.hh"
#include "object_container.hh"
#include "predicate_container.hh"
#include "function_container.hh"
#include "derived_predicate_container.hh"
#include "class_container.hh"
#include "preference_container.hh"
#include "operator_container.hh"
#include "operator.hh"
#include "formula.hh"

namespace skdecide {

namespace pddl {

class Domain : public Identifier,
               public TypeContainer<Domain>,
               public ObjectContainer<Domain>,
               public PredicateContainer<Domain>,
               public FunctionContainer<Domain>,
               public DerivedPredicateContainer<Domain>,
               public ClassContainer<Domain>,
               public PreferenceContainer<Domain>,
               public ActionContainer<Domain>,
               public DurativeActionContainer<Domain>,
               public EventContainer<Domain>,
               public ProcessContainer<Domain> {
public:
  static constexpr char class_name[] = "domain";

  typedef std::shared_ptr<Domain> Ptr;

  Domain(const std::string &name);
  virtual ~Domain();

  void set_requirements(const Requirements::Ptr &requirements);
  const Requirements::Ptr &get_requirements() const;

  void set_constraints(const Formula::Ptr &constraints);
  const Formula::Ptr &get_constraints() const;

  virtual std::string print() const;

  typedef TypeContainer<Domain>::TypePtr TypePtr;
  typedef TypeContainer<Domain>::TypeSet TypeSet;
  typedef ObjectContainer<Domain>::ObjectPtr ObjectPtr;
  typedef ObjectContainer<Domain>::ObjectSet ObjectSet;
  typedef PredicateContainer<Domain>::PredicatePtr PredicatePtr;
  typedef PredicateContainer<Domain>::PredicateSet PredicateSet;
  typedef DerivedPredicateContainer<Domain>::DerivedPredicatePtr
      DerivedPredicatePtr;
  typedef DerivedPredicateContainer<Domain>::DerivedPredicateSet
      DerivedPredicateSet;
  typedef FunctionContainer<Domain>::FunctionPtr FunctionPtr;
  typedef FunctionContainer<Domain>::FunctionSet FunctionSet;
  typedef ClassContainer<Domain>::ClassPtr ClassPtr;
  typedef ClassContainer<Domain>::ClassSet ClassSet;
  typedef PreferenceContainer<Domain>::PreferenceSet PreferenceSet;
  typedef PreferenceContainer<Domain>::PreferencePtr PreferencePtr;
  typedef OperatorContainer<Domain, Action>::OperatorPtr ActionPtr;
  typedef OperatorContainer<Domain, Action>::OperatorSet ActionSet;
  typedef OperatorContainer<Domain, DurativeAction>::OperatorPtr
      DurativeActionPtr;
  typedef OperatorContainer<Domain, DurativeAction>::OperatorSet
      DurativeActionSet;
  typedef OperatorContainer<Domain, Event>::OperatorPtr EventPtr;
  typedef OperatorContainer<Domain, Event>::OperatorSet EventSet;
  typedef OperatorContainer<Domain, Process>::OperatorPtr ProcessPtr;
  typedef OperatorContainer<Domain, Process>::OperatorSet ProcessSet;

private:
  Requirements::Ptr _requirements;
  Formula::Ptr _constraints;
};

// Domain printing operator
std::ostream &operator<<(std::ostream &o, const Domain &d);

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_DOMAIN_HH
