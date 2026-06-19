/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_CONSTRAINT_FORMULA_HH
#define SKDECIDE_PDDL_CONSTRAINT_FORMULA_HH

#include "formula.hh"
#include "unary_formula.hh"
#include "binary_formula.hh"
#include "number.hh"

namespace skdecide {

namespace pddl {

class AlwaysFormula : public UnaryFormula<AlwaysFormula> {
public:
  static constexpr char class_name[] = "always";

  typedef std::shared_ptr<AlwaysFormula> Ptr;

  AlwaysFormula();
  AlwaysFormula(const Formula::Ptr &formula);
  AlwaysFormula(const AlwaysFormula &other);
  AlwaysFormula &operator=(const AlwaysFormula &other);
  virtual ~AlwaysFormula();

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

class SometimeFormula : public UnaryFormula<SometimeFormula> {
public:
  static constexpr char class_name[] = "sometime";

  typedef std::shared_ptr<SometimeFormula> Ptr;

  SometimeFormula();
  SometimeFormula(const Formula::Ptr &formula);
  SometimeFormula(const SometimeFormula &other);
  SometimeFormula &operator=(const SometimeFormula &other);
  virtual ~SometimeFormula();

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

class AtMostOnceFormula : public UnaryFormula<AtMostOnceFormula> {
public:
  static constexpr char class_name[] = "at-most-once";

  typedef std::shared_ptr<AtMostOnceFormula> Ptr;

  AtMostOnceFormula();
  AtMostOnceFormula(const Formula::Ptr &formula);
  AtMostOnceFormula(const AtMostOnceFormula &other);
  AtMostOnceFormula &operator=(const AtMostOnceFormula &other);
  virtual ~AtMostOnceFormula();

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

class WithinFormula : public UnaryFormula<WithinFormula> {
public:
  static constexpr char class_name[] = "within";

  typedef std::shared_ptr<WithinFormula> Ptr;

  WithinFormula();
  WithinFormula(const Formula::Ptr &formula, const Number::Ptr &deadline);
  WithinFormula(const WithinFormula &other);
  WithinFormula &operator=(const WithinFormula &other);
  virtual ~WithinFormula();

  void set_deadline(const Number::Ptr &deadline);
  const Number::Ptr &get_deadline() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;

private:
  Number::Ptr _deadline;
};

class HoldAfterFormula : public UnaryFormula<HoldAfterFormula> {
public:
  static constexpr char class_name[] = "hold-after";

  typedef std::shared_ptr<HoldAfterFormula> Ptr;

  HoldAfterFormula();
  HoldAfterFormula(const Formula::Ptr &formula, const Number::Ptr &from);
  HoldAfterFormula(const HoldAfterFormula &other);
  HoldAfterFormula &operator=(const HoldAfterFormula &other);
  virtual ~HoldAfterFormula();

  void set_from(const Number::Ptr &from);
  const Number::Ptr &get_from() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;

private:
  Number::Ptr _from;
};

class HoldDuringFormula : public UnaryFormula<HoldDuringFormula> {
public:
  static constexpr char class_name[] = "hold-during";

  typedef std::shared_ptr<HoldDuringFormula> Ptr;

  HoldDuringFormula();
  HoldDuringFormula(const Formula::Ptr &formula, const Number::Ptr &from,
                    const Number::Ptr &deadline);
  HoldDuringFormula(const HoldDuringFormula &other);
  HoldDuringFormula &operator=(const HoldDuringFormula &other);
  virtual ~HoldDuringFormula();

  void set_from(const Number::Ptr &from);
  const Number::Ptr &get_from() const;

  void set_deadline(const Number::Ptr &deadline);
  const Number::Ptr &get_deadline() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;

private:
  Number::Ptr _from;
  Number::Ptr _deadline;
};

class SometimeAfterFormula : public BinaryFormula<SometimeAfterFormula> {
public:
  static constexpr char class_name[] = "sometime-after";

  typedef std::shared_ptr<SometimeAfterFormula> Ptr;

  SometimeAfterFormula();
  SometimeAfterFormula(const Formula::Ptr &left_formula,
                       const Formula::Ptr &right_formula);
  SometimeAfterFormula(const SometimeAfterFormula &other);
  SometimeAfterFormula &operator=(const SometimeAfterFormula &other);
  virtual ~SometimeAfterFormula();

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

class SometimeBeforeFormula : public BinaryFormula<SometimeBeforeFormula> {
public:
  static constexpr char class_name[] = "sometime-before";

  typedef std::shared_ptr<SometimeBeforeFormula> Ptr;

  SometimeBeforeFormula();
  SometimeBeforeFormula(const Formula::Ptr &left_formula,
                        const Formula::Ptr &right_formula);
  SometimeBeforeFormula(const SometimeBeforeFormula &other);
  SometimeBeforeFormula &operator=(const SometimeBeforeFormula &other);
  virtual ~SometimeBeforeFormula();

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

class AlwaysWithinFormula : public BinaryFormula<AlwaysWithinFormula> {
public:
  static constexpr char class_name[] = "always-within";

  typedef std::shared_ptr<AlwaysWithinFormula> Ptr;

  AlwaysWithinFormula();
  AlwaysWithinFormula(const Formula::Ptr &left_formula,
                      const Formula::Ptr &right_formula,
                      const Number::Ptr &deadline);
  AlwaysWithinFormula(const AlwaysWithinFormula &other);
  AlwaysWithinFormula &operator=(const AlwaysWithinFormula &other);
  virtual ~AlwaysWithinFormula();

  void set_deadline(const Number::Ptr &deadline);
  const Number::Ptr &get_deadline() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;

private:
  Number::Ptr _deadline;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_CONSTRAINT_FORMULA_HH
