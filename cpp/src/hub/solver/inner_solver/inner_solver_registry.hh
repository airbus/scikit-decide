/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_INNER_SOLVER_REGISTRY_HH
#define SKDECIDE_INNER_SOLVER_REGISTRY_HH

#include <any>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "meta_inner_solver_base.hh"

namespace skdecide {

struct InnerSolverParams {
  std::unordered_map<std::string, std::any> _params;

  template <typename T>
  T get(const std::string &key, const T &default_value) const {
    auto it = _params.find(key);
    if (it == _params.end())
      return default_value;
    return std::any_cast<T>(it->second);
  }

  template <typename T> void set(const std::string &key, T value) {
    _params[key] = std::move(value);
  }
};

template <typename Domain, typename Texecution> struct InnerSolverEntry {
  using State = typename Domain::State;
  using Action = typename Domain::Action;
  using Value = typename Domain::Value;
  using Predicate = typename Domain::Predicate;

  using GoalChecker = std::function<Predicate(Domain &, const State &)>;
  using Heuristic = std::function<Value(Domain &, const State &)>;
  using TerminalValue = std::function<Value(const State &)>;
  using InnerSolver = MetaInnerSolverBase<Domain>;

  const char *name;
  bool supports_terminal_value;

  std::function<std::unique_ptr<InnerSolver>(Domain &, GoalChecker, Heuristic,
                                             TerminalValue,
                                             const InnerSolverParams &, bool)>
      create;
};

template <typename Domain, typename Texecution>
const std::vector<InnerSolverEntry<Domain, Texecution>> &
get_inner_solver_registry();

template <typename Domain, typename Texecution>
const InnerSolverEntry<Domain, Texecution> &
find_inner_solver(const std::string &name);

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "hub/solver/inner_solver/impl/inner_solver_registry_impl.hh"
#endif

#endif // SKDECIDE_INNER_SOLVER_REGISTRY_HH
