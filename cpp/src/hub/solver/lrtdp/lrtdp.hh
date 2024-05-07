/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * This is the skdecide implementation of LRTDP from the paper
 * "Labeled RTDP: Improving the Convergence of Real-Time Dynamic
 * Programming" from Bonet and Geffner (ICAPS 2003)
 */
#ifndef SKDECIDE_LRTDP_HH
#define SKDECIDE_LRTDP_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <stack>
#include <list>
#include <chrono>
#include <random>

#include "utils/associative_container_deducer.hh"
#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

/**
 * @brief This is the skdecide implementation of "Labeled RTDP: Improving the
 * Convergence of Real-Time Dynamic Programming" by Blai Bonet and Hector
 * Geffner (ICAPS 2003)
 *
 * @tparam Tdomain Type of the domain class
 * @tparam Texecution_policy Type of the execution policy (one of
 * 'SequentialExecution' to execute rollouts in sequence, or 'ParallelExecution'
 * to execute rollouts in parallel on different threads)
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class LRTDPSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef typename Domain::Predicate Predicate;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Predicate(Domain &, const State &, const std::size_t *)>
      GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &, const std::size_t *)>
      HeuristicFunctor;
  typedef std::function<bool(const LRTDPSolver &, Domain &,
                             const std::size_t *)>
      CallbackFunctor;

  /**
   * @brief Construct a new LRTDPSolver object
   *
   * @param domain The domain instance
   * @param goal_checker Functor taking as arguments the domain, a state object
   * and the thread ID from which it is called, and returning true if the state
   * is the goal
   * @param heuristic Functor taking as arguments the domain, a state object and
   * the thread ID from which it is called, and returning the heuristic estimate
   * from the state to the goal
   * @param use_labels Boolean indicating whether labels must be used (true) or
   * not (false, in which case the algorithm is equivalent to the standard RTDP)
   * @param time_budget Maximum solving time in milliseconds
   * @param rollout_budget Maximum number of rollouts (deactivated when
   * use_labels is true)
   * @param max_depth Maximum depth of each LRTDP trial (rollout)
   * @param residual_moving_average_window Number of latest computed residual
   * values to memorize in order to compute the average Bellman error (residual)
   * at the root state of the search (deactivated when use_labels
   * is true)
   * @param epsilon Maximum Bellman error (residual) allowed to decide that a
   * state is solved, or to decide when no label is used that the value
   * function of the root state of the search has converged (in the latter case:
   * the root state's Bellman error is averaged over the
   * residual_moving_average_window, deactivated when
   * use_labels is true)
   * @param discount Value function's discount factor
   * @param online_node_garbage Boolean indicating whether the search graph
   * which is no more reachable from the root solving state should be
   * deleted (true) or not (false)
   * @param debug_logs Boolean indicating whether debugging messages should be
   * logged (true) or not (false)
   * @param callback Functor called at the end of each LRTDP trial (rollout),
   * taking as arguments the solver, the domain and the thread ID from which it
   * is called, and returning true if the solver must be stopped
   */
  LRTDPSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, bool use_labels = true,
      std::size_t time_budget = 3600000, std::size_t rollout_budget = 100000,
      std::size_t max_depth = 1000,
      std::size_t residual_moving_average_window = 100, double epsilon = 0.001,
      double discount = 1.0, bool online_node_garbage = false,
      bool debug_logs = false,
      const CallbackFunctor &callback = [](const LRTDPSolver &, Domain &,
                                           const std::size_t *) {
        return false;
      });

  /**
   * @brief Clears the search graph, thus preventing from reusing previous
   * search results)
   *
   */
  void clear();

  /**
   * @brief Call the LRTDP algorithm
   *
   * @param s Root state of the search from which LRTDP trials are launched
   */
  void solve(const State &s);

  /**
   * @brief Indicates whether the solution policy is defined for a given state
   *
   * @param s State for which an entry is searched in the policy graph
   * @return true If the state has been explored and an action is defined in
   * this state
   * @return false If the state has not been explored or no action is defined in
   * this state
   */
  bool is_solution_defined_for(const State &s) const;

  /**
   * @brief Get the best computed action in terms of best Q-value in a given
   * state (throws a runtime error exception if no action is defined in the
   * given state, which is why it is advised to call
   * LRTDPSolver::is_solution_defined_for before). The search
   * subgraph which is no more reachable after executing the returned action is
   * also deleted if node garbage was set to true in the LRTDPSolver instance's
   * constructor.
   *
   * @param s State for which the best action is requested
   * @return const Action& Best computed action
   */
  const Action &get_best_action(const State &s);

  /**
   * @brief Get the best Q-value in a given state (throws a runtime
   * error exception if no action is defined in the given state, which is why it
   * is advised to call LRTDPSolver::is_solution_defined_for before)
   *
   * @param s State from which the best Q-value is requested
   * @return double Maximum Q-value of the given state over the applicable
   * actions in this state
   */
  double get_best_value(const State &s) const;

  /**
   * @brief Get the number of states present in the search graph (which can be
   * lower than the number of actually explored states if node garbage was
   * set to true in the LRTDPSolver instance's constructor)
   *
   * @return std::size_t Number of states present in the search graph
   */
  std::size_t get_nb_explored_states() const;

  /**
   * @brief Get the number of rollouts since the beginning of the search from
   * the root solving state
   *
   * @return std::size_t Number of rollouts (LRTDP trials)
   */
  std::size_t get_nb_rollouts() const;

  /**
   * @brief Get the average Bellman error (residual)
   * at the root state of the search, or an infinite value if the number of
   * computed residuals is lower than the epsilon moving average window set in
   * the LRTDPSolver instance's constructor
   *
   * @return double Bellman error at the root state of the search averaged over
   * the epsilon moving average window
   */
  double get_residual_moving_average() const;

  /**
   * @brief Get the solving time in milliseconds since the beginning of the
   * search from the root solving state
   *
   * @return std::size_t Solving time in milliseconds
   */
  std::size_t get_solving_time();

  /**
   * @brief Get the (partial) solution policy defined for the states for which
   * the Q-value has been updated at least once (which is optimal if the
   * algorithm has converged and labels are used); warning: only defined over
   * the states reachable from the last root solving state when node garbage was
   * set to True in the LRTDPSolver instance's constructor
   *
   * @return Mapping from states to pairs of action and best Q-value
   */
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map
  get_policy() const;

private:
  typedef typename ExecutionPolicy::template atomic<std::size_t> atomic_size_t;
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;
  typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  bool _use_labels;
  atomic_size_t _time_budget;
  atomic_size_t _rollout_budget;
  atomic_size_t _max_depth;
  atomic_size_t _residual_moving_average_window;
  atomic_double _epsilon;
  atomic_double _discount;
  bool _online_node_garbage;
  atomic_bool _debug_logs;
  CallbackFunctor _callback;
  ExecutionPolicy _execution_policy;

  std::unique_ptr<std::mt19937> _gen;
  typename ExecutionPolicy::Mutex _gen_mutex;
  typename ExecutionPolicy::Mutex _time_mutex;
  typename ExecutionPolicy::Mutex _residuals_protect;

  atomic_double _residual_moving_average;
  std::list<double> _residuals;

  struct ActionNode;

  struct StateNode {
    State state;
    std::list<std::unique_ptr<ActionNode>> actions;
    ActionNode *best_action;
    atomic_double best_value;
    atomic_double goal;
    atomic_bool solved;
    typename ExecutionPolicy::Mutex mutex;

    StateNode(const State &s);

    struct Key {
      const State &operator()(const StateNode &sn) const;
    };
  };

  struct ActionNode {
    Action action;
    std::vector<std::tuple<double, double, StateNode *>>
        outcomes; // next state nodes owned by _graph
    std::discrete_distribution<> dist;
    atomic_double value;

    ActionNode(const Action &a);
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  Graph _graph;
  StateNode *_current_state;
  atomic_size_t _nb_rollouts;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  void expand(StateNode *s, const std::size_t *thread_id);
  double q_value(ActionNode *a);
  ActionNode *greedy_action(StateNode *s, const std::size_t *thread_id);
  void update(StateNode *s, const std::size_t *thread_id);
  StateNode *pick_next_state(ActionNode *a);
  double residual(StateNode *s, const std::size_t *thread_id);
  bool check_solved(StateNode *s, const std::size_t *thread_id);
  void trial(StateNode *s, const std::size_t *thread_id);
  void compute_reachable_subgraph(StateNode *node,
                                  std::unordered_set<StateNode *> &subgraph);
  void remove_subgraph(std::unordered_set<StateNode *> &root_subgraph,
                       std::unordered_set<StateNode *> &child_subgraph);
  void update_residual_moving_average(const StateNode &node,
                                      const double &node_record_value);
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/lrtdp_impl.hh"
#endif

#endif // SKDECIDE_LRTDP_HH
