/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_META_INNER_SOLVER_HH
#define SKDECIDE_META_INNER_SOLVER_HH

#include <memory>

#include "meta_inner_solver_base.hh"

namespace skdecide {

template <typename TSolver, typename TDomain>
class MetaInnerSolver : public MetaInnerSolverBase<TDomain> {
public:
  using State = typename TDomain::State;
  using Action = typename TDomain::Action;
  using Value = typename TDomain::Value;

  MetaInnerSolver(std::unique_ptr<TSolver> solver);

  void solve(const State &s) override;
  void clear() override;
  bool is_solution_defined_for(const State &s) override;
  const Action &get_best_action(const State &s) override;
  Value get_best_value(const State &s) override;
  typename SetTypeDeducer<State>::Set get_explored_states() const override;

private:
  std::unique_ptr<TSolver> _solver;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/meta_inner_solver_impl.hh"
#endif

#endif // SKDECIDE_META_INNER_SOLVER_HH
