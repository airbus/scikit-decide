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
  typedef std::function<bool(const std::size_t &, const std::size_t &,
                             const double &, const double &)>
      WatchdogFunctor;

  // If use_labels is true, then rollout_budget and epsilon moving average are
  // voided
  LRTDPSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, bool use_labels = true,
      std::size_t time_budget = 3600000, std::size_t rollout_budget = 100000,
      std::size_t max_depth = 1000,
      std::size_t epsilon_moving_average_window = 100, double epsilon = 0.001,
      double discount = 1.0, bool online_node_garbage = false,
      bool debug_logs = false,
      const WatchdogFunctor &watchdog = [](const std::size_t &,
                                           const std::size_t &, const double &,
                                           const double &) { return true; });

  // clears the solver (clears the search graph, thus preventing from reusing
  // previous search results)
  void clear();

  // solves from state s using heuristic function h
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s);
  double get_best_value(const State &s) const;
  std::size_t get_nb_of_explored_states() const;
  std::size_t get_nb_rollouts() const;
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map policy() const;

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
  atomic_size_t _epsilon_moving_average_window;
  atomic_double _epsilon;
  atomic_double _discount;
  bool _online_node_garbage;
  atomic_bool _debug_logs;
  WatchdogFunctor _watchdog;
  ExecutionPolicy _execution_policy;

  std::unique_ptr<std::mt19937> _gen;
  typename ExecutionPolicy::Mutex _gen_mutex;
  typename ExecutionPolicy::Mutex _time_mutex;
  typename ExecutionPolicy::Mutex _epsilons_protect;

  atomic_double _epsilon_moving_average;
  std::list<double> _epsilons;

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

  void expand(StateNode *s, const std::size_t *thread_id);
  double q_value(ActionNode *a);
  ActionNode *greedy_action(StateNode *s, const std::size_t *thread_id);
  void update(StateNode *s, const std::size_t *thread_id);
  StateNode *pick_next_state(ActionNode *a);
  double residual(StateNode *s, const std::size_t *thread_id);
  bool
  check_solved(StateNode *s,
               const std::chrono::time_point<std::chrono::high_resolution_clock>
                   &start_time,
               const std::size_t *thread_id);
  void trial(StateNode *s,
             const std::chrono::time_point<std::chrono::high_resolution_clock>
                 &start_time,
             const std::size_t *thread_id);
  void compute_reachable_subgraph(StateNode *node,
                                  std::unordered_set<StateNode *> &subgraph);
  void remove_subgraph(std::unordered_set<StateNode *> &root_subgraph,
                       std::unordered_set<StateNode *> &child_subgraph);
  std::size_t update_epsilon_moving_average(const StateNode &node,
                                            const double &node_record_value);
  std::size_t
  elapsed_time(const std::chrono::time_point<std::chrono::high_resolution_clock>
                   &start_time);
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/lrtdp_impl.hh"
#endif

#endif // SKDECIDE_LRTDP_HH
