/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_PREFERENCE_HH
#define SKDECIDE_PDDL_PREFERENCE_HH

#include "formula.hh"
#include "identifier.hh"

namespace skdecide {

namespace pddl {

class Preference : public Formula, public Identifier {
public:
  static constexpr char class_name[] = "preference";

  typedef std::shared_ptr<Preference> Ptr;

  Preference();
  Preference(const std::string &name);
  Preference(const Formula::Ptr &formula,
             const std::string &name = "anonymous");
  Preference(const Preference &other);
  Preference &operator=(const Preference &other);
  virtual ~Preference();

  void set_name(const std::string &name);

  Preference &set_formula(const Formula::Ptr &formula);
  const Formula::Ptr &get_formula() const;

  virtual std::ostream &print(std::ostream &o) const;

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const override;

private:
  Formula::Ptr _formula;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PREFERENCE_HH
