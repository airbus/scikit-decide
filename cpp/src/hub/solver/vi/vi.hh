/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * This is the skdecide implementation of Value Iteration.
 */
#ifndef SKDECIDE_VI_HH
#define SKDECIDE_VI_HH

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
 * @brief Value Iteration solver for Markov Decision Processes.
 *
 * The algorithm proceeds in two phases:
 *
 * 1. **State enumeration**: BFS from the initial state, expanding all
 *    reachable states via get_applicable_actions() and
 *    get_next_state_distribution(). All transition probabilities and costs
 *    are cached in a graph. States where is_terminal() returns true are
 *    treated as absorbing states whose value is set by the terminal_value
 *    functor (defaults to reward=0, which models goal-like terminals; a
 *    large negative reward can be returned for dead-end-like terminals).
 *
 * 2. **Synchronous Bellman sweeps**: Iterates over all non-terminal states,
 *    applying the Bellman backup V(s) = max_a [R(s,a) + gamma * sum_s'
 *    P(s'|s,a) * V(s')] until the maximum residual across all states drops
 *    below epsilon. With ParallelExecution, Bellman updates are applied to
 *    all states simultaneously within each sweep (Jacobi-style), which
 *    preserves convergence guarantees.
 *
 * **Heuristic initialization (non-standard extension)**: Classical Value
 * Iteration initializes V(s) = 0 for all states. This implementation
 * accepts an optional heuristic functor h(s) returning a Value used to
 * initialize V(s) = h(s).reward() during state enumeration. When h is
 * admissible (i.e. h(s) >= V*(s) for reward-maximizing problems), this
 * warm-start preserves correctness and typically reduces the number of
 * Bellman sweeps needed for convergence. The default heuristic returns
 * Value(reward=0) for all states, recovering standard VI.
 *
 * @tparam Tdomain Type of the domain class
 * @tparam Texecution_policy Type of the execution policy (SequentialExecution
 * for sequential Bellman sweeps, or ParallelExecution for Jacobi-style
 * parallel updates within each sweep and parallel action-transition
 * generation during state enumeration)
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class VISolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<Value(const State &)> TerminalValueFunctor;
  typedef std::function<bool(const VISolver &, Domain &)> CallbackFunctor;

  VISolver(
      Domain &domain, const HeuristicFunctor &heuristic,
      const TerminalValueFunctor &terminal_value =
          [](const State &) { return Value(0.0, true); },
      double discount = 0.999, double epsilon = 0.001,
      std::size_t max_sweeps = 0,
      const CallbackFunctor &callback = [](const VISolver &,
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
  typename SetTypeDeducer<State>::Set get_converged_states() const;
  typename SetTypeDeducer<State>::Set get_states_updated_in_last_sweep() const;
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map policy() const;

private:
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;
  typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

  Domain &_domain;
  HeuristicFunctor _heuristic;
  TerminalValueFunctor _terminal_value;
  atomic_double _discount;
  atomic_double _epsilon;
  std::size_t _max_sweeps;
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
    bool converged;
    bool updated_in_last_sweep;

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
  std::vector<StateNode *> _non_terminal_states;
  std::size_t _nb_iterations;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  void enumerate_reachable_states(const State &s);
  void expand(StateNode &s);
  double bellman_update(StateNode &s);
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/vi_impl.hh"
#endif

#endif // SKDECIDE_VI_HH
