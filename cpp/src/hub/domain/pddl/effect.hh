/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_EFFECT_HH
#define SKDECIDE_PDDL_EFFECT_HH

#include <functional>
#include <memory>
#include <ostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "semantics/fwd.hh"
#include "semantics/state.hh"

namespace skdecide {

namespace pddl {

enum class DeterminizationMode { MostProbable, Random };

class Effect {
public:
  typedef std::shared_ptr<Effect> Ptr;

  using Outcome = std::pair<double, State>;
  using Outcomes = std::vector<Outcome>;

  using AtomCallback = std::function<void(int, const GroundTuple &)>;
  using CostCallback = std::function<void(double)>;

  virtual ~Effect();

  virtual std::ostream &print(std::ostream &o) const = 0;
  std::string print() const;

  virtual Outcomes apply(const State &state, const Task &task,
                         const Binding &binding) const = 0;

  virtual void collect_add_atoms(const Task &task, const Binding &binding,
                                 const AtomCallback &callback) const;

  virtual void collect_cost_increase(const Task &task, const Binding &binding,
                                     const CostCallback &callback) const;

  virtual bool is_probabilistic() const;

  virtual Ptr determinize(const Ptr &self, DeterminizationMode mode,
                          std::mt19937 &rng) const;

  virtual std::vector<Ptr> all_determinizations(const Ptr &self,
                                                std::mt19937 &rng) const;
};

// Effect printing operator
std::ostream &operator<<(std::ostream &o, const Effect &e);

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_EFFECT_HH
