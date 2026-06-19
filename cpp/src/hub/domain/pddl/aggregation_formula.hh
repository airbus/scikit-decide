/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_AGGREGATION_FORMULA_HH
#define SKDECIDE_PDDL_AGGREGATION_FORMULA_HH

#include "formula.hh"
#include <vector>

namespace skdecide {

namespace pddl {

template <typename Derived> class AggregationFormula : public Formula {
public:
  typedef std::shared_ptr<AggregationFormula<Derived>> Ptr;
  typedef Formula::Ptr FormulaPtr;
  typedef std::vector<Formula::Ptr> FormulaVector;

  AggregationFormula();
  AggregationFormula(const AggregationFormula &other);
  AggregationFormula &operator=(const AggregationFormula &other);
  virtual ~AggregationFormula();

  AggregationFormula &append_formula(const Formula::Ptr &formula);

  /**
   * Removes the formula at a given index.
   * Throws an exception if the given index exceeds the size of the
   * aggregation formula
   */
  AggregationFormula &remove_formula(const std::size_t &index);

  /**
   * Gets the formula at a given index.
   * Throws an exception if the given index exceeds the size of the
   * aggregation formula
   */
  const Formula::Ptr &formula_at(const std::size_t &index);

  const FormulaVector &get_formulas() const;
  virtual std::ostream &print(std::ostream &o) const;

private:
  FormulaVector _formulas;
};

class ConjunctionFormula : public AggregationFormula<ConjunctionFormula> {
public:
  static constexpr char class_name[] = "and";

  typedef std::shared_ptr<ConjunctionFormula> Ptr;
  typedef AggregationFormula<ConjunctionFormula>::FormulaPtr FormulaPtr;
  typedef std::vector<FormulaPtr> FormulaVector;

  ConjunctionFormula();
  ConjunctionFormula(const ConjunctionFormula &other);
  ConjunctionFormula &operator=(const ConjunctionFormula &other);
  virtual ~ConjunctionFormula();

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;

  void collect_positive_atoms(const Task &task, const Binding &binding,
                              const AtomCallback &callback) const override;
};

class DisjunctionFormula : public AggregationFormula<DisjunctionFormula> {
public:
  static constexpr char class_name[] = "or";

  typedef std::shared_ptr<DisjunctionFormula> Ptr;
  typedef AggregationFormula<DisjunctionFormula>::FormulaPtr FormulaPtr;
  typedef std::vector<FormulaPtr> FormulaVector;

  DisjunctionFormula();
  DisjunctionFormula(const DisjunctionFormula &other);
  DisjunctionFormula &operator=(const DisjunctionFormula &other);
  virtual ~DisjunctionFormula();

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/aggregation_formula_impl.hh"
#endif

#endif // SKDECIDE_PDDL_AGGREGATION_FORMULA_HH
