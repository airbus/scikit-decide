/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_META_INNER_SOLVER_BASE_HH
#define SKDECIDE_META_INNER_SOLVER_BASE_HH

#include "utils/associative_container_deducer.hh"

namespace skdecide {

template <typename Tdomain> class MetaInnerSolverBase {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;

  virtual ~MetaInnerSolverBase() {}

  virtual void solve(const State &s) = 0;
  virtual void clear() = 0;
  virtual bool is_solution_defined_for(const State &s) = 0;
  virtual const Action &get_best_action(const State &s) = 0;
  virtual Value get_best_value(const State &s) = 0;
  virtual typename SetTypeDeducer<State>::Set get_explored_states() const = 0;
};

} // namespace skdecide

#endif // SKDECIDE_META_INNER_SOLVER_BASE_HH
