/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_QUANTIFIED_FORMULA_HH
#define SKDECIDE_PDDL_QUANTIFIED_FORMULA_HH

#include "formula.hh"
#include "variable_container.hh"

namespace skdecide {

namespace pddl {

template <typename Derived>
class QuantifiedFormula : public Formula, public VariableContainer<Derived> {
public:
  typedef std::shared_ptr<QuantifiedFormula<Derived>> Ptr;
  typedef typename VariableContainer<Derived>::VariablePtr VariablePtr;
  typedef typename VariableContainer<Derived>::VariableVector VariableVector;

  QuantifiedFormula();
  QuantifiedFormula(const Formula::Ptr &formula,
                    const VariableContainer<Derived> &variables);
  QuantifiedFormula(const QuantifiedFormula &other);
  QuantifiedFormula &operator=(const QuantifiedFormula &other);
  virtual ~QuantifiedFormula();

  QuantifiedFormula &set_formula(const Formula::Ptr &formula);
  const Formula::Ptr &get_formula() const;

  static const char *get_name();

  virtual std::ostream &print(std::ostream &o) const;

private:
  Formula::Ptr _formula;
};

class UniversalFormula : public QuantifiedFormula<UniversalFormula> {
public:
  static constexpr char class_name[] = "forall";

  typedef std::shared_ptr<UniversalFormula> Ptr;
  typedef QuantifiedFormula<UniversalFormula> VariablePtr;
  typedef QuantifiedFormula<UniversalFormula> VariableVector;

  UniversalFormula();
  UniversalFormula(const Formula::Ptr &formula,
                   const VariableContainer<UniversalFormula> &variables);
  UniversalFormula(const UniversalFormula &other);
  UniversalFormula &operator=(const UniversalFormula &other);
  virtual ~UniversalFormula();

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

class ExistentialFormula : public QuantifiedFormula<ExistentialFormula> {
public:
  static constexpr char class_name[] = "exists";

  typedef std::shared_ptr<ExistentialFormula> Ptr;
  typedef QuantifiedFormula<ExistentialFormula> VariablePtr;
  typedef QuantifiedFormula<ExistentialFormula> VariableVector;

  ExistentialFormula();
  ExistentialFormula(const Formula::Ptr &formula,
                     const VariableContainer<ExistentialFormula> &variables);
  ExistentialFormula(const ExistentialFormula &other);
  ExistentialFormula &operator=(const ExistentialFormula &other);
  virtual ~ExistentialFormula();

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/quantified_formula_impl.hh"
#endif

#endif // SKDECIDE_PDDL_QUANTIFIED_FORMULA_HH
