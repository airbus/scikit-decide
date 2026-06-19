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

  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<Value(const State &)> TerminalValueFunctor;
  typedef std::function<bool(const MDPLPSolver &, Domain &)> CallbackFunctor;

  MDPLPSolver(
      Domain &domain, const HeuristicFunctor &heuristic,
      const TerminalValueFunctor &terminal_value =
          [](const State &) { return Value(0.0, false); },
      LPVariant variant = LPVariant::Dual, double discount = 0.99,
      double epsilon = 0.001, double lp_infinity = 1e20,
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

private:
  Domain &_domain;
  HeuristicFunctor _heuristic;
  TerminalValueFunctor _terminal_value;
  LPVariant _variant;
  double _discount;
  double _epsilon;
  double _lp_infinity;
  CallbackFunctor _callback;
  bool _verbose;

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

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<bool(const SSPLPSolver &, Domain &)> CallbackFunctor;

  SSPLPSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, LPVariant variant = LPVariant::Dual,
      double epsilon = 0.001, double lp_infinity = 1e20,
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

private:
  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  LPVariant _variant;
  double _epsilon;
  double _lp_infinity;
  CallbackFunctor _callback;
  bool _verbose;

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
