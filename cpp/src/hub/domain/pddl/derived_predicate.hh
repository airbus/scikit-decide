/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DERIVED_PREDICATE_HH
#define SKDECIDE_PDDL_DERIVED_PREDICATE_HH

#include "predicate.hh"
#include "formula.hh"

namespace skdecide {

namespace pddl {

class DerivedPredicate : public Predicate {
public:
  static constexpr char class_name[] = "derived predicate";

  typedef std::shared_ptr<DerivedPredicate> Ptr;

  DerivedPredicate(const std::string &name);
  DerivedPredicate(const std::string &name, const Formula::Ptr &formula);
  DerivedPredicate(const Predicate::Ptr &predicate,
                   const Formula::Ptr &formula);
  DerivedPredicate(const DerivedPredicate &other);
  DerivedPredicate &operator=(const DerivedPredicate &other);
  virtual ~DerivedPredicate();

  void set_formula(const Formula::Ptr &formula);
  const Formula::Ptr &get_formula() const;

  std::ostream &print(std::ostream &o) const;

private:
  Formula::Ptr _formula;
};

// Derived predicate printing operator
std::ostream &operator<<(std::ostream &o, const DerivedPredicate &d);

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_DERIVED_PREDICATE_HH
