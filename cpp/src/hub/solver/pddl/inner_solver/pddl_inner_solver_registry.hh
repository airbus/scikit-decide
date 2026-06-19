/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_INNER_SOLVER_REGISTRY_HH
#define SKDECIDE_PDDL_INNER_SOLVER_REGISTRY_HH

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hub/solver/inner_solver/inner_solver_registry.hh"
#include "hub/solver/inner_solver/meta_inner_solver_base.hh"
#include "hub/solver/pddl/pddl_domain_adapter.hh"

namespace skdecide {

namespace pddl {

template <typename Texecution> struct PddlInnerSolverEntry {
  using Domain = PddlDeterministicDomain;
  using InnerSolver = MetaInnerSolverBase<Domain>;

  const char *name;

  std::function<std::unique_ptr<InnerSolver>(Domain &,
                                             const InnerSolverParams &, bool)>
      create;
};

template <typename Texecution>
const std::vector<PddlInnerSolverEntry<Texecution>> &
get_pddl_inner_solver_registry();

template <typename Texecution>
const PddlInnerSolverEntry<Texecution> &
find_pddl_inner_solver(const std::string &name);

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "hub/solver/pddl/inner_solver/impl/pddl_inner_solver_registry_impl.hh"
#endif

#endif // SKDECIDE_PDDL_INNER_SOLVER_REGISTRY_HH
