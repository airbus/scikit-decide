/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_TIMED_FORMULA_HH
#define SKDECIDE_PDDL_TIMED_FORMULA_HH

#include "unary_formula.hh"

namespace skdecide {

namespace pddl {

class AtStartFormula : public UnaryFormula<AtStartFormula> {
public:
  static constexpr char class_name[] = "at start";

  typedef std::shared_ptr<AtStartFormula> Ptr;

  AtStartFormula() {}

  AtStartFormula(const Formula::Ptr &formula)
      : UnaryFormula<AtStartFormula>(formula) {}

  AtStartFormula(const AtStartFormula &other)
      : UnaryFormula<AtStartFormula>(other) {}

  AtStartFormula &operator=(const AtStartFormula &other);

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

class AtEndFormula : public UnaryFormula<AtEndFormula> {
public:
  static constexpr char class_name[] = "at end";

  typedef std::shared_ptr<AtEndFormula> Ptr;

  AtEndFormula() {}

  AtEndFormula(const Formula::Ptr &formula)
      : UnaryFormula<AtEndFormula>(formula) {}

  AtEndFormula(const AtEndFormula &other) : UnaryFormula<AtEndFormula>(other) {}

  AtEndFormula &operator=(const AtEndFormula &other);

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

class OverAllFormula : public UnaryFormula<OverAllFormula> {
public:
  static constexpr char class_name[] = "over all";

  typedef std::shared_ptr<OverAllFormula> Ptr;

  OverAllFormula() {}

  OverAllFormula(const Formula::Ptr &formula)
      : UnaryFormula<OverAllFormula>(formula) {}

  OverAllFormula(const OverAllFormula &other)
      : UnaryFormula<OverAllFormula>(other) {}

  OverAllFormula &operator=(const OverAllFormula &other);

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_TIMED_FORMULA_HH
