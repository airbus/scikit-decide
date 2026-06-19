/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_BINARY_FORMULA_HH
#define SKDECIDE_PDDL_BINARY_FORMULA_HH

#include "formula.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> class BinaryFormula : public Formula {
public:
  typedef std::shared_ptr<BinaryFormula<Derived>> Ptr;

  BinaryFormula();
  BinaryFormula(const Formula::Ptr &left_formula,
                const Formula::Ptr &right_formula);
  BinaryFormula(const BinaryFormula<Derived> &other);
  BinaryFormula<Derived> &operator=(const BinaryFormula<Derived> &other);
  virtual ~BinaryFormula();

  void set_left_formula(const Formula::Ptr &formula);
  const Formula::Ptr &get_left_formula() const;

  void set_right_formula(const Formula::Ptr &formula);
  const Formula::Ptr &get_right_formula() const;

  virtual std::ostream &print(std::ostream &o) const;

protected:
  Formula::Ptr _left_formula;
  Formula::Ptr _right_formula;
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/binary_formula_impl.hh"
#endif

#endif // SKDECIDE_PDDL_BINARY_FORMULA_HH
