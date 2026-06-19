/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_EQUALITY_FORMULA_HH
#define SKDECIDE_PDDL_EQUALITY_FORMULA_HH

#include "formula.hh"
#include "term_container.hh"

namespace skdecide {

namespace pddl {

class EqualityFormula : public Formula, public TermContainer<EqualityFormula> {
public:
  static constexpr char class_name[] = "equality formula";

  typedef std::shared_ptr<EqualityFormula> Ptr;
  typedef TermContainer<EqualityFormula>::TermPtr TermPtr;
  typedef TermContainer<EqualityFormula>::TermVector TermVector;

  EqualityFormula();
  EqualityFormula(const TermContainer<EqualityFormula> &terms);
  EqualityFormula(const EqualityFormula &other);
  EqualityFormula &operator=(const EqualityFormula &other);
  virtual ~EqualityFormula();

  std::string get_name() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_EQUALITY_FORMULA_HH
