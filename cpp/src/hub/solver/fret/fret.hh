/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_FRET_HH
#define SKDECIDE_FRET_HH

#include <chrono>
#include <functional>
#include <memory>
#include <vector>

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

/**
 * @brief FRET (Find, Revise, Eliminate Traps) from Kolobov et al.
 * (ICAPS 2011).
 *
 * FRET is a meta-solver for Generalized Stochastic Shortest Path MDPs.
 * It iterates: (1) Find-and-Revise — run an inner solver to convergence,
 * (2) Eliminate-Traps — detect traps via Tarjan SCC on the greedy graph
 * and adjust values. Converges to V* even in the presence of dead ends
 * and 0-cost cycles.
 *
 * @tparam Tdomain Domain type
 * @tparam Texecution_policy Execution policy
 * @tparam TinnerSolver Inner SSP solver template
 */
template <typename Tdomain, typename Texecution_policy,
          template <typename, typename> class TinnerSolver>
class FRETSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef typename Domain::Predicate Predicate;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<bool(const FRETSolver &, Domain &)> CallbackFunctor;

  typedef TinnerSolver<Tdomain, Texecution_policy> InnerSolver;

private:
  template <typename T, typename = void>
  struct has_terminal_value_functor : std::false_type {};
  template <typename T>
  struct has_terminal_value_functor<
      T, std::void_t<typename T::TerminalValueFunctor>> : std::true_type {};

  static_assert(
      has_terminal_value_functor<InnerSolver>::value,
      "FRET requires an inner solver that defines TerminalValueFunctor "
      "(i.e. supports terminal_value in its constructor). "
      "Use LRTDPSolver, LDFSSolver, or VISolver. ILAOStarSolver is not "
      "supported because it does not allow overriding terminal state "
      "values, which FRET needs to propagate dead-end costs across "
      "iterations.");

public:
  typedef std::function<std::unique_ptr<InnerSolver>(
      Domain &, GoalCheckerFunctor, HeuristicFunctor)>
      InnerSolverFactory;

  template <typename... InnerSolverArgs>
  FRETSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, double discount = 1.0,
      double epsilon = 0.001, double dead_end_cost = 10000.0,
      const CallbackFunctor &callback = [](const FRETSolver &,
                                           Domain &) { return false; },
      bool verbose = false, InnerSolverArgs &&...inner_solver_args);

  void clear();
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  Value get_best_value(const State &s) const;

  std::size_t get_nb_explored_states() const;
  std::size_t get_nb_fret_iterations() const;
  std::size_t get_nb_traps_eliminated() const;
  std::size_t get_solving_time() const;

  typename SetTypeDeducer<State>::Set get_explored_states() const;
  typename SetTypeDeducer<State>::Set get_dead_end_states() const;
  std::vector<typename SetTypeDeducer<State>::Set> get_trapped_sccs() const;

private:
  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  double _discount;
  double _epsilon;
  double _dead_end_cost;
  InnerSolverFactory _inner_solver_factory;
  CallbackFunctor _callback;
  bool _verbose;

  typedef typename MapTypeDeducer<State, double>::Map ValueFunctionMap;
  typedef typename MapTypeDeducer<State, Action>::Map PolicyMap;

  ValueFunctionMap _value_function;
  PolicyMap _policy;
  typename SetTypeDeducer<State>::Set _dead_end_states;
  std::vector<typename SetTypeDeducer<State>::Set> _trapped_sccs;

  std::size_t _nb_fret_iterations;
  std::size_t _nb_traps_eliminated;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  double get_value(const State &s) const;

  // Greedy graph: adjacency list (state → list of successor states)
  typedef typename MapTypeDeducer<State, std::vector<State>>::Map GreedyGraph;

  void build_greedy_graph(InnerSolver &inner, GreedyGraph &graph);

  // Tarjan SCC on greedy graph
  struct TarjanData {
    std::size_t idx;
    std::size_t low;
    bool on_stack;
  };
  std::vector<std::vector<State>> tarjan_scc(const GreedyGraph &graph);

  // Returns true if traps were found and eliminated
  bool eliminate_traps(InnerSolver &inner);

  void extract_proper_policy(InnerSolver &inner, const State &s);
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/fret_impl.hh"
#endif

#endif // SKDECIDE_FRET_HH
