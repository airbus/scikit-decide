/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_SSIPP_HH
#define SKDECIDE_SSIPP_HH

#include <chrono>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

/**
 * @brief SSiPP (Short-Sighted Probabilistic Planner) from Trevizan & Veloso
 * (ICAPS 2012).
 *
 * SSiPP repeatedly builds short-sighted sub-SSPs (bounded BFS to depth t),
 * solves each optimally with an inner solver, and accumulates a global value
 * function. Boundary states at distance t get V(s) as goal cost, guiding
 * search toward true goals.
 *
 * @tparam Tdomain Domain type
 * @tparam Texecution_policy Execution policy (Sequential or Parallel)
 * @tparam TinnerSolver Inner SSP solver template (LRTDPSolver, ILAOStarSolver,
 *   or LDFSSolver)
 */
template <typename Tdomain, typename Texecution_policy,
          template <typename, typename> class TinnerSolver>
class SSiPPSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef typename Domain::Predicate Predicate;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<bool(const SSiPPSolver &, Domain &)> CallbackFunctor;

  typedef TinnerSolver<Tdomain, Texecution_policy> InnerSolver;

  typedef std::function<std::unique_ptr<InnerSolver>(
      Domain &, GoalCheckerFunctor, HeuristicFunctor)>
      InnerSolverFactory;

  template <typename... InnerSolverArgs>
  SSiPPSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, std::size_t depth = 3,
      double discount = 1.0, double epsilon = 0.001,
      std::size_t max_iterations = 10000,
      const CallbackFunctor &callback = [](const SSiPPSolver &,
                                           Domain &) { return false; },
      bool verbose = false, InnerSolverArgs &&...inner_solver_args);

  void clear();

  void solve(const State &s);

  bool is_solution_defined_for(const State &s);
  const Action &get_best_action(const State &s);
  Value get_best_value(const State &s) const;

  std::size_t get_nb_explored_states() const;
  std::size_t get_nb_sub_ssps() const;
  std::size_t get_solving_time() const;

  typename SetTypeDeducer<State>::Set get_explored_states() const;
  typename SetTypeDeducer<State>::Set get_current_subssp_states() const;
  typename SetTypeDeducer<State>::Set get_boundary_states() const;

private:
  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  std::size_t _depth;
  double _discount;
  double _epsilon;
  std::size_t _max_iterations;
  InnerSolverFactory _inner_solver_factory;
  CallbackFunctor _callback;
  bool _verbose;

  typedef typename MapTypeDeducer<State, double>::Map ValueFunctionMap;
  typedef typename MapTypeDeducer<State, Action>::Map PolicyMap;

  ValueFunctionMap _value_function;
  PolicyMap _policy;

  typename SetTypeDeducer<State>::Set _current_subssp_states;
  typename SetTypeDeducer<State>::Set _boundary_states;

  std::size_t _nb_sub_ssps;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  void build_short_sighted_ssp(const State &s);
  void solve_subssp(const State &s);
  double get_value(const State &s) const;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/ssipp_impl.hh"
#endif

#endif // SKDECIDE_SSIPP_HH
