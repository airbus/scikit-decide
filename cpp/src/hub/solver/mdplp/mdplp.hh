/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MDPLP_HH
#define SKDECIDE_MDPLP_HH

#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

enum class LPVariant { Primal, Dual };

inline LPVariant lp_variant_from_string(const std::string &s) {
  if (s == "primal")
    return LPVariant::Primal;
  if (s == "dual")
    return LPVariant::Dual;
  throw std::invalid_argument("Unknown LP variant: '" + s +
                              "'. Use 'primal' or 'dual'.");
}

/**
 * @brief LP solver for MDPs (primal and dual formulations).
 *
 * Enumerates the full reachable state space via BFS, then solves a linear
 * program using HiGHS. Supports both primal (value variables) and dual
 * (occupation measure) formulations.
 *
 * @tparam Tdomain Domain type
 * @tparam Texecution_policy Execution policy
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class MDPLPSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef typename Domain::Predicate Predicate;
  typedef Texecution_policy ExecutionPolicy;

  enum class LPCallbackEvent {
    SolverIteration,
    LPProgress,
  };

  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<Value(const State &)> TerminalValueFunctor;
  typedef std::function<bool(const MDPLPSolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Construct an MDP LP solver.
   *
   * @param domain The MDP domain to solve.
   * @param heuristic Functor h(domain, state) -> Value used as the initial
   *   value estimate for LP warm-starting.
   * @param terminal_value Functor f(state) -> Value assigning a fixed value to
   *   terminal (absorbing) states. Defaults to Value(0.0, false).
   * @param variant LP formulation: LPVariant::Primal (value variables, one per
   *   state) or LPVariant::Dual (occupation measures, one per state-action
   *   pair). Defaults to LPVariant::Dual.
   * @param discount Discount factor gamma. Must be < 1 for LP feasibility in
   *   general MDPs. Defaults to 0.99.
   * @param epsilon Convergence threshold for policy extraction. Defaults to
   *   0.001.
   * @param lp_infinity Upper bound used for LP variable bounds and constraint
   *   coefficients in HiGHS. Defaults to 1e20.
   * @param lp_callback_interval Fire the callback every N simplex iterations
   *   during the LP solve, reporting intermediate values. 0 disables LP-level
   *   callbacks. Defaults to 0.
   * @param callback Functor called after solving or during LP iterations;
   *   return true to stop early. Defaults to never stop.
   * @param verbose Whether to log progress messages. Defaults to false.
   */
  MDPLPSolver(
      Domain &domain, const HeuristicFunctor &heuristic,
      const TerminalValueFunctor &terminal_value =
          [](const State &) { return Value(0.0, false); },
      LPVariant variant = LPVariant::Dual, double discount = 0.99,
      double epsilon = 0.001, double lp_infinity = 1e20,
      std::size_t lp_callback_interval = 0,
      const CallbackFunctor &callback = [](const MDPLPSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  Value get_best_value(const State &s) const;

  std::size_t get_nb_states() const;
  std::size_t get_nb_lp_variables() const;
  std::size_t get_nb_lp_constraints() const;
  std::size_t get_solving_time() const;
  typename SetTypeDeducer<State>::Set get_explored_states() const;
  LPCallbackEvent get_callback_event() const { return _last_callback_event; }

  template <typename Params>
  static std::unique_ptr<MDPLPSolver> create_from_params(
      Domain &domain,
      std::function<Predicate(Domain &, const State &)> goal_checker,
      std::function<Value(Domain &, const State &)> heuristic,
      std::function<Value(const State &)> terminal_value, const Params &params,
      bool verbose);

private:
  Domain &_domain;
  HeuristicFunctor _heuristic;
  TerminalValueFunctor _terminal_value;
  LPVariant _variant;
  double _discount;
  double _epsilon;
  double _lp_infinity;
  std::size_t _lp_callback_interval;
  CallbackFunctor _callback;
  bool _verbose;
  LPCallbackEvent _last_callback_event = LPCallbackEvent::SolverIteration;

  struct ActionNode;

  struct StateNode {
    State state;
    std::list<std::unique_ptr<ActionNode>> actions;
    ActionNode *best_action;
    double best_value;
    bool terminal;
    std::size_t index;

    struct Key {
      const State &operator()(const StateNode &sn) const { return sn.state; }
    };

    StateNode(const State &s);
  };

  struct ActionNode {
    Action action;
    std::list<std::tuple<double, double, StateNode *>> outcomes;
    double value;

    ActionNode(const Action &a);
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  Graph _graph;
  std::vector<StateNode *> _non_terminal_states;

  ExecutionPolicy _execution_policy;

  std::size_t _nb_lp_variables;
  std::size_t _nb_lp_constraints;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  void enumerate_reachable_states(const State &s);
  void expand(StateNode &s);
  void solve_primal_lp();
  void solve_dual_lp(const State &s0);
  void extract_policy_from_values();
};

/**
 * @brief LP solver for undiscounted SSPs (primal and dual formulations).
 *
 * Like MDPLPSolver but for Stochastic Shortest Path problems: discount=1,
 * goal states with V(g)=0, positive costs. The LP is feasible when a proper
 * policy exists (all states can reach a goal).
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class SSPLPSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef typename Domain::Predicate Predicate;
  typedef Texecution_policy ExecutionPolicy;

  enum class LPCallbackEvent {
    SolverIteration,
    LPProgress,
  };

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<bool(const SSPLPSolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Construct an SSP LP solver.
   *
   * @param domain The SSP domain to solve.
   * @param goal_checker Functor returning true when a state is a goal. Goal
   *   states have V(g) = 0.
   * @param heuristic Functor h(domain, state) -> Value used as the initial
   *   value estimate for LP warm-starting.
   * @param variant LP formulation: LPVariant::Primal (value variables) or
   *   LPVariant::Dual (occupation measures). Defaults to LPVariant::Dual.
   * @param epsilon Convergence threshold for policy extraction. Defaults to
   *   0.001.
   * @param lp_infinity Upper bound used for LP variable bounds and constraint
   *   coefficients in HiGHS. Defaults to 1e20.
   * @param lp_callback_interval Fire the callback every N simplex iterations
   *   during the LP solve, reporting intermediate values. 0 disables LP-level
   *   callbacks. Defaults to 0.
   * @param callback Functor called after solving or during LP iterations;
   *   return true to stop early. Defaults to never stop.
   * @param verbose Whether to log progress messages. Defaults to false.
   */
  SSPLPSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, LPVariant variant = LPVariant::Dual,
      double epsilon = 0.001, double lp_infinity = 1e20,
      std::size_t lp_callback_interval = 0,
      const CallbackFunctor &callback = [](const SSPLPSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  Value get_best_value(const State &s) const;

  std::size_t get_nb_states() const;
  std::size_t get_nb_lp_variables() const;
  std::size_t get_nb_lp_constraints() const;
  std::size_t get_solving_time() const;
  typename SetTypeDeducer<State>::Set get_explored_states() const;
  LPCallbackEvent get_callback_event() const { return _last_callback_event; }

  template <typename Params>
  static std::unique_ptr<SSPLPSolver> create_from_params(
      Domain &domain,
      std::function<Predicate(Domain &, const State &)> goal_checker,
      std::function<Value(Domain &, const State &)> heuristic,
      std::function<Value(const State &)> terminal_value, const Params &params,
      bool verbose);

private:
  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  LPVariant _variant;
  double _epsilon;
  double _lp_infinity;
  std::size_t _lp_callback_interval;
  CallbackFunctor _callback;
  bool _verbose;
  LPCallbackEvent _last_callback_event = LPCallbackEvent::SolverIteration;

  struct ActionNode;

  struct StateNode {
    State state;
    std::list<std::unique_ptr<ActionNode>> actions;
    ActionNode *best_action;
    double best_value;
    bool terminal;
    bool goal;
    std::size_t index;

    struct Key {
      const State &operator()(const StateNode &sn) const { return sn.state; }
    };

    StateNode(const State &s);
  };

  struct ActionNode {
    Action action;
    std::list<std::tuple<double, double, StateNode *>> outcomes;
    double value;

    ActionNode(const Action &a);
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  Graph _graph;
  std::vector<StateNode *> _non_goal_states;

  ExecutionPolicy _execution_policy;

  std::size_t _nb_lp_variables;
  std::size_t _nb_lp_constraints;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  void enumerate_reachable_states(const State &s);
  void expand(StateNode &s);
  void solve_primal_lp();
  void solve_dual_lp(const State &s0);
  void extract_policy_from_values();
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/mdplp_impl.hh"
#endif

#endif // SKDECIDE_MDPLP_HH
