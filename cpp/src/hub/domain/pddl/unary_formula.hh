/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_UNARY_FORMULA_HH
#define SKDECIDE_PDDL_UNARY_FORMULA_HH

#include "formula.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> class UnaryFormula : public Formula {
public:
  typedef std::shared_ptr<UnaryFormula<Derived>> Ptr;

  UnaryFormula() {}

  UnaryFormula(const Formula::Ptr &formula) : _formula(formula) {}

  UnaryFormula(const UnaryFormula<Derived> &other) : _formula(other._formula) {}

  UnaryFormula<Derived> &operator=(const UnaryFormula<Derived> &other);

  virtual ~UnaryFormula() {}

  void set_formula(const Formula::Ptr &formula);

  const Formula::Ptr &get_formula() const;

  virtual std::ostream &print(std::ostream &o) const;

protected:
  Formula::Ptr _formula;
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/unary_formula_impl.hh"
#endif

#endif // SKDECIDE_PDDL_UNARY_FORMULA_HH
