/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * This is the skdecide implementation of Policy Iteration.
 */
#ifndef SKDECIDE_PI_HH
#define SKDECIDE_PI_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>
#include <list>
#include <chrono>

#include "utils/associative_container_deducer.hh"
#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

/**
 * @brief Policy Iteration solver for Markov Decision Processes.
 *
 * The algorithm proceeds in three phases:
 *
 * 1. **State enumeration**: BFS from the initial state, expanding all
 *    reachable states via get_applicable_actions() and
 *    get_next_state_distribution(). All transition probabilities and rewards
 *    are cached in a graph. States where is_terminal() returns true are
 *    treated as absorbing states whose value is set by the terminal_value
 *    functor (defaults to reward=0 for goal-like terminals; a large negative
 *    reward can be returned for dead-end-like terminals).
 *
 * 2. **Policy evaluation**: For the current policy pi, iteratively computes
 *    V^pi(s) = R(s,pi(s)) + gamma * sum_s' P(s'|s,pi(s)) * V^pi(s') using
 *    Gauss-Seidel sweeps until the maximum residual drops below epsilon.
 *
 * 3. **Policy improvement**: For each state, greedily selects the action
 *    maximizing Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) * V(s'). If
 *    the policy changes, go back to step 2. Otherwise, the algorithm has
 *    converged to the optimal policy.
 *
 * **Heuristic initialization (non-standard extension)**: Classical PI
 * initializes V(s) = 0 for all states. This implementation accepts an
 * optional heuristic functor h(s) returning a Value used to initialize
 * V(s) = h(s).reward() during state enumeration. The default returns
 * Value(reward=0), recovering standard PI.
 *
 * **Initial policy (optional warm-start)**: By default the initial policy
 * selects the first applicable action in each state. An optional
 * initial_policy functor can be provided to seed pi(s) with a domain-
 * specific action, reducing the number of evaluate/improve iterations.
 *
 * @tparam Tdomain Type of the domain class
 * @tparam Texecution_policy Type of the execution policy (SequentialExecution
 * for sequential evaluation sweeps, or ParallelExecution for Jacobi-style
 * parallel evaluation and parallel action-transition generation)
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class PISolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<Value(const State &)> TerminalValueFunctor;
  typedef std::function<Action(Domain &, const State &)> InitialPolicyFunctor;
  typedef std::function<bool(const PISolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Construct a Policy Iteration solver.
   *
   * @param domain The MDP domain to solve.
   * @param heuristic Functor h(domain, state) -> Value used to initialize
   *   V(s) = h(s).reward() during state enumeration (non-standard warm-start).
   *   Defaults to Value(reward=0) = standard PI.
   * @param terminal_value Functor f(state) -> Value assigning a fixed value to
   *   terminal (absorbing) states. Use Value(reward=0) for goal-like terminals
   *   and a large negative reward for dead-end-like terminals. Defaults to
   *   Value(0.0, true).
   * @param initial_policy Optional functor pi(domain, state) -> action to seed
   *   the initial policy. When provided, each state's policy is initialized to
   *   the returned action (falling back to the first applicable action if not
   *   applicable). Defaults to nullptr (first applicable action).
   * @param discount Discount factor gamma in [0, 1]. Defaults to 0.999.
   * @param epsilon Maximum Bellman residual for policy evaluation convergence.
   *   Defaults to 0.001.
   * @param max_eval_sweeps Maximum Gauss-Seidel sweeps per policy evaluation
   *   phase. 0 means unlimited (exact evaluation). A positive value yields
   *   modified policy iteration, useful when discount=1.0. Defaults to 0.
   * @param callback Functor called at the end of each evaluate/improve
   *   iteration; return true to stop early. Defaults to never stop.
   * @param verbose Whether to log progress messages. Defaults to false.
   */
  PISolver(
      Domain &domain, const HeuristicFunctor &heuristic,
      const TerminalValueFunctor &terminal_value =
          [](const State &) { return Value(0.0, true); },
      const InitialPolicyFunctor &initial_policy = nullptr,
      double discount = 0.999, double epsilon = 0.001,
      std::size_t max_eval_sweeps = 0,
      const CallbackFunctor &callback = [](const PISolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s) const;
  Value get_best_value(const State &s) const;

  std::size_t get_nb_explored_states() const;
  std::size_t get_nb_iterations() const;
  std::size_t get_solving_time() const;

  typename SetTypeDeducer<State>::Set get_explored_states() const;
  typename SetTypeDeducer<State>::Set get_policy_changed_states() const;
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map policy() const;

  /**
   * @brief Mark a state as a dead-end with the given cost. The state's value
   * is set to dead_end_cost, its best action is cleared, and it is excluded
   * from future policy evaluation. This method is provided for meta-solvers
   * like FRET that detect trap states and penalize them between solving
   * iterations.
   *
   * @param s State to mark as dead-end
   * @param dead_end_cost Cost value to assign to the dead-end state
   */
  void set_state_dead_end(const State &s, double dead_end_cost);

  template <typename Params>
  static std::unique_ptr<PISolver> create_from_params(
      Domain &domain,
      std::function<Predicate(Domain &, const State &)> goal_checker,
      std::function<Value(Domain &, const State &)> heuristic,
      std::function<Value(const State &)> terminal_value, const Params &params,
      bool verbose);

private:
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;
  typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

  Domain &_domain;
  HeuristicFunctor _heuristic;
  TerminalValueFunctor _terminal_value;
  InitialPolicyFunctor _initial_policy;
  atomic_double _discount;
  atomic_double _epsilon;
  std::size_t _max_eval_sweeps;
  CallbackFunctor _callback;
  bool _verbose;
  ExecutionPolicy _execution_policy;

  struct ActionNode;

  struct StateNode {
    State state;
    std::list<std::unique_ptr<ActionNode>> actions;
    ActionNode *best_action;
    atomic_double best_value;
    bool terminal;
    bool dead_end;
    bool policy_changed;

    StateNode(const State &s);

    struct Key {
      const State &operator()(const StateNode &sn) const;
    };
  };

  struct ActionNode {
    Action action;
    std::list<std::tuple<double, double, StateNode *>>
        outcomes; // (probability, reward, next_state_node)
    double value;

    ActionNode(const Action &a);
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  Graph _graph;
  std::vector<StateNode *> _non_terminal_states;
  std::size_t _nb_iterations;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  void enumerate_reachable_states(const State &s);
  void expand(StateNode &s);
  void initialize_policy();
  bool evaluate_policy();
  bool improve_policy();
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/pi_impl.hh"
#endif

#endif // SKDECIDE_PI_HH
