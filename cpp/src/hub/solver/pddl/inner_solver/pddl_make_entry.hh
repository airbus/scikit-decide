/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_MAKE_ENTRY_HH
#define SKDECIDE_PDDL_MAKE_ENTRY_HH

#include "hub/solver/pddl/inner_solver/pddl_inner_solver_registry.hh"

namespace skdecide {

namespace pddl {

template <typename TSolver, typename Texecution>
PddlInnerSolverEntry<Texecution> make_pddl_entry(const char *name) {
  using Entry = PddlInnerSolverEntry<Texecution>;

  Entry entry;
  entry.name = name;

  entry.create =
      [](PddlDeterministicDomain &d, const InnerSolverParams &p,
         bool verbose) -> std::unique_ptr<typename Entry::InnerSolver> {
    return TSolver::create_from_params(d, p, verbose);
  };

  return entry;
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_MAKE_ENTRY_HH
