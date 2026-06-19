/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MAKE_ENTRY_HH
#define SKDECIDE_MAKE_ENTRY_HH

#include "hub/solver/inner_solver/inner_solver_registry.hh"
#include "hub/solver/inner_solver/meta_inner_solver.hh"
#include "hub/solver/inner_solver/impl/meta_inner_solver_impl.hh"

namespace skdecide {

// Generic factory for registering any solver as an inner solver.
// TSolver is a fully-specialized solver type (e.g. AOStarSolver<Domain,
// Texecution> or FRETSolver<Domain, Texecution, MetaInnerSolverProxy>). The
// solver must provide a static create_from_params() method.
template <typename TSolver, typename Domain, typename Texecution>
InnerSolverEntry<Domain, Texecution> make_entry(const char *name,
                                                bool supports_tv) {
  using MIS = MetaInnerSolver<TSolver, Domain>;
  using Entry = InnerSolverEntry<Domain, Texecution>;

  Entry entry;
  entry.name = name;
  entry.supports_terminal_value = supports_tv;

  entry.create =
      [](Domain &d, typename Entry::GoalChecker gc, typename Entry::Heuristic h,
         typename Entry::TerminalValue tv, const InnerSolverParams &p,
         bool verbose) -> std::unique_ptr<typename Entry::InnerSolver> {
    return std::make_unique<MIS>(TSolver::create_from_params(
        d, std::move(gc), std::move(h), std::move(tv), p, verbose));
  };

  return entry;
}

} // namespace skdecide

#endif // SKDECIDE_MAKE_ENTRY_HH
