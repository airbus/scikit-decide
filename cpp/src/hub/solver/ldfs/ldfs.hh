/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_LDFS_HH
#define SKDECIDE_LDFS_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <stack>
#include <vector>
#include <list>
#include <chrono>
#include <limits>

#include "utils/associative_container_deducer.hh"
#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"
#include "hub/solver/inner_solver/inner_solver_traits.hh"

namespace skdecide {

/**
 * @brief Labeled Depth-First Search (LDFS) solver for goal-oriented MDPs.
 *
 * From Bonet & Geffner, "Learning Depth-First Search: A Unified Approach to
 * Heuristic Search in Deterministic and Non-Deterministic Settings, and Its
 * Application to MDPs", ICAPS 2008. This implements the LDFS(MDP) algorithm
 * from Figure 4 of the paper.
 *
 * LDFS is a systematic alternative to LRTDP for solving Stochastic Shortest
 * Path (SSP) problems. It performs iterated depth-first searches from the
 * initial state, trying all consistent actions (those within epsilon of
 * optimal) and recursing into successors. Tarjan's SCC algorithm is
 * integrated into the DFS to detect cyclic components in the greedy policy
 * graph: when all states in an SCC are consistent, the entire component is
 * labeled as solved.
 *
 * The algorithm uses cost minimization: V(s) = min_a [C(s,a) + gamma *
 * sum_s' P(s'|s,a) * V(s')].
 *
 * Goal states are absorbing with V(s) = 0. Non-goal terminal states have
 * their value set by the terminal_value functor (defaults to cost=0).
 * A goal_checker functor is required: without goal states and discount=1.0,
 * the algorithm will not converge.
 *
 * @tparam Tdomain Type of the domain class
 * @tparam Texecution_policy Type of the execution policy (SequentialExecution
 * or ParallelExecution for parallel action-transition generation)
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class LDFSSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<Value(const State &)> TerminalValueFunctor;
  typedef std::function<bool(const LDFSSolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Constructs a new LDFS solver instance.
   *
   * @param domain The domain instance to solve.
   * @param goal_checker Functor testing whether a state is a goal. Goal
   *   states are absorbing with V(s) = 0.
   * @param heuristic Functor returning the heuristic cost estimate for a
   *   state, used to initialize V(s) for newly discovered states. An
   *   admissible heuristic (h(s) <= V*(s)) accelerates convergence.
   * @param terminal_value Functor assigning a fixed value to non-goal
   *   terminal (absorbing) states. Use Value(large_cost) for dead ends.
   *   When nullptr (default, recommended for SSPs), dead-end states are
   *   initialised with the heuristic estimate instead of a fixed value,
   *   which prevents infinity propagation through Bellman updates and lets
   *   LDFS naturally steer away from them. Pass a functor only when
   *   dead-ends are unavoidable and you need explicit penalty values.
   * @param discount Value function's discount factor. Defaults to 1.0.
   * @param epsilon Maximum Bellman error allowed to label a state as solved
   *   during the check_solved procedure. Defaults to 0.001.
   * @param max_depth Maximum DFS depth per driver iteration. 0 means
   *   unlimited. When reached, the DFS backtracks as if the state were
   *   unsolved; the driver loop retries, preserving correctness.
   *   Defaults to 0 (unlimited).
   * @param callback Functor called at the end of each LDFS pass, taking
   *   the solver and domain as arguments and returning true to stop.
   *   Defaults to always returning false.
   * @param verbose Whether to log verbose debug messages. Defaults to false.
   */
  LDFSSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic,
      const TerminalValueFunctor &terminal_value = nullptr,
      double discount = 1.0, double epsilon = 0.001, std::size_t max_depth = 0,
      const CallbackFunctor &callback = [](const LDFSSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  Value get_best_value(const State &s) const;

  std::size_t get_nb_explored_states() const;
  std::size_t get_nb_tip_states() const;
  std::size_t get_solving_time() const;

  typename SetTypeDeducer<State>::Set get_explored_states() const;
  typename SetTypeDeducer<State>::Set get_solved_states() const;
  std::vector<typename SetTypeDeducer<State>::Set>
  get_strongly_connected_components() const;
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map policy() const;
  std::vector<std::pair<State, Action>> get_last_trajectory() const;

  template <typename Params>
  static std::unique_ptr<LDFSSolver> create_from_params(
      Domain &domain,
      std::function<Predicate(Domain &, const State &)> goal_checker,
      std::function<Value(Domain &, const State &)> heuristic,
      std::function<Value(const State &)> terminal_value, const Params &params,
      bool verbose);

protected:
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;
  typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

  static constexpr std::size_t IDX_UNDEF =
      std::numeric_limits<std::size_t>::max();

  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  TerminalValueFunctor _terminal_value;
  bool _use_terminal_value;
  atomic_double _discount;
  atomic_double _epsilon;
  std::size_t _max_depth;
  CallbackFunctor _callback;
  bool _verbose;
  ExecutionPolicy _execution_policy;

  struct ActionNode;

  struct StateNode {
    State state;
    std::list<std::unique_ptr<ActionNode>> actions;
    ActionNode *best_action;
    atomic_double best_value;
    bool goal;
    bool terminal;
    bool solved;
    bool active;
    std::size_t idx;
    std::size_t low;

    StateNode(const State &s);

    struct Key {
      const State &operator()(const StateNode &sn) const;
    };
  };

  struct ActionNode {
    Action action;
    std::list<std::tuple<double, double, StateNode *>>
        outcomes; // (probability, cost, next_state_node)
    double value;

    ActionNode(const Action &a);
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  Graph _graph;
  std::vector<std::vector<StateNode *>> _sccs;
  std::size_t _nb_tip_states;
  std::size_t _tarjan_index;
  std::stack<StateNode *> _tarjan_stack;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
  std::vector<StateNode *> _last_trajectory;

  void expand(StateNode &s);
  double q_value(ActionNode &a);
  void ldfs_mdp(StateNode &root);
  void clear_active_flags();

  bool _last_rv;
};

/**
 * @brief IDA* solver for deterministic planning problems.
 *
 * From Bonet & Geffner (ICAPS 2008, Proposition 6): LDFS with transposition
 * tables reduces to IDA* over deterministic models when the value function
 * is monotone (admissible).
 *
 * Compared to LDFS, this specialization removes parameters that are
 * irrelevant for deterministic planning:
 * - terminal_value: non-goal terminals are dead-ends, never on a solution
 * - discount: always 1.0 (undiscounted shortest path)
 * - epsilon: always 0 (exact optimality in deterministic settings)
 *
 * A static_assert verifies at compile time that the domain provides
 * get_next_state(), which is the signature of deterministic transitions.
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class IDAstarSolver : public LDFSSolver<Tdomain, Texecution_policy> {
  static_assert(has_get_next_state<Tdomain>::value,
                "IDAstarSolver requires a deterministic domain providing "
                "get_next_state(state, action)");

  typedef LDFSSolver<Tdomain, Texecution_policy> Base;

public:
  typedef typename Tdomain::State State;
  typedef typename Tdomain::Action Action;
  typedef typename Tdomain::Value Value;
  typedef typename Base::GoalCheckerFunctor GoalCheckerFunctor;
  typedef typename Base::HeuristicFunctor HeuristicFunctor;
  typedef typename Base::CallbackFunctor CallbackFunctor;

  /**
   * @brief Constructs a new IDA* solver instance.
   *
   * This is the primary constructor exposing only the parameters relevant to
   * deterministic search. Parameters irrelevant for deterministic planning
   * (terminal_value, discount, epsilon) are fixed internally.
   *
   * @param domain The deterministic domain instance to solve.
   * @param goal_checker Functor testing whether a state is a goal.
   * @param heuristic Functor returning the heuristic cost estimate for a
   *   state. Must be admissible (h(s) <= V*(s)) for optimality.
   * @param max_depth Maximum DFS depth per iteration. 0 means unlimited.
   *   Defaults to 0.
   * @param callback Functor called at the end of each IDA* iteration,
   *   taking the solver and domain as arguments and returning true to stop.
   *   Defaults to always returning false.
   * @param verbose Whether to log verbose debug messages. Defaults to false.
   */
  IDAstarSolver(
      Tdomain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, std::size_t max_depth = 0,
      const CallbackFunctor &callback = [](const Base &,
                                           Tdomain &) { return false; },
      bool verbose = false);

  /**
   * @brief Constructs an IDA* solver with the full LDFS parameter signature.
   *
   * This constructor accepts the same signature as LDFSSolver for
   * compatibility with meta-solvers (e.g. SSiPP, FRET) that construct
   * inner solvers generically. The terminal_value, discount, and epsilon
   * parameters are accepted but ignored since IDA* fixes them internally.
   *
   * @param domain The deterministic domain instance to solve.
   * @param goal_checker Functor testing whether a state is a goal.
   * @param heuristic Functor returning the heuristic cost estimate.
   * @param terminal_value Ignored (accepted for LDFS signature compatibility).
   * @param discount Ignored (always 1.0 for IDA*).
   * @param epsilon Ignored (always 0 for exact deterministic search).
   * @param max_depth Maximum DFS depth per iteration. 0 means unlimited.
   * @param callback Functor called at the end of each IDA* iteration.
   * @param verbose Whether to log verbose debug messages.
   */
  IDAstarSolver(Tdomain &domain, const GoalCheckerFunctor &goal_checker,
                const HeuristicFunctor &heuristic,
                const typename Base::TerminalValueFunctor &, double, double,
                std::size_t max_depth, const CallbackFunctor &callback,
                bool verbose);

  std::vector<Action> get_plan(const State &s) const;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/ldfs_impl.hh"
#endif

#endif // SKDECIDE_LDFS_HH
