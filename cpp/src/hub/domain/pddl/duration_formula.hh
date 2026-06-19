/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DURATION_FORMULA_HH
#define SKDECIDE_PDDL_DURATION_FORMULA_HH

#include "formula.hh"

namespace skdecide {

namespace pddl {

class DurativeAction;

class DurationFormula : Formula {
public:
  typedef std::shared_ptr<DurationFormula> Ptr;
  typedef std::shared_ptr<DurativeAction> DurativeActionPtr;

  DurationFormula();
  DurationFormula(const DurativeActionPtr &durative_action);
  DurationFormula(const DurationFormula &other);
  DurationFormula &operator=(const DurationFormula &other);
  virtual ~DurationFormula();

  void set_durative_action(const DurativeActionPtr &durative_action);
  const DurativeActionPtr &get_durative_action() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;

private:
  DurativeActionPtr _durative_action;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_DURATION_FORMULA_HH
