/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MARTDP_HH
#define SKDECIDE_MARTDP_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <queue>
#include <list>
#include <chrono>
#include <random>

#include "utils/associative_container_deducer.hh"
#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class MARTDPSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::Agent Agent;
  typedef typename Domain::State State;
  typedef typename Domain::State::Data AgentState;
  typedef typename Domain::Event Action;
  typedef typename Domain::Action::Data AgentAction;
  typedef typename Domain::Value Value;
  typedef typename Domain::Predicate Predicate;
  typedef typename Domain::EnvironmentOutcome EnvironmentOutcome;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<std::pair<Value, Action>(Domain &, const State &)>
      HeuristicFunctor;
  typedef std::function<bool(const std::size_t &, const std::size_t &,
                             const double &, const double &)>
      WatchdogFunctor;

  MARTDPSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, std::size_t time_budget = 3600000,
      std::size_t rollout_budget = 100000, std::size_t max_depth = 1000,
      std::size_t max_feasibility_trials = 0, // will then choose nb_agents
      double graph_expansion_rate = 0.1,
      std::size_t epsilon_moving_average_window = 100,
      double epsilon = 0.0, // not a stopping criterion by default
      double discount = 1.0, double action_choice_noise = 0.1,
      const double &dead_end_cost = 10e4, bool online_node_garbage = false,
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
  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  std::size_t _time_budget;
  std::size_t _rollout_budget;
  std::size_t _max_depth;
  std::size_t _max_feasibility_trials;
  double _graph_expansion_rate;
  std::size_t _epsilon_moving_average_window;
  double _epsilon;
  double _discount;
  double _dead_end_cost;
  bool _online_node_garbage;
  bool _debug_logs;
  WatchdogFunctor _watchdog;
  ExecutionPolicy _execution_policy;
  std::unique_ptr<std::mt19937> _gen;
  typename ExecutionPolicy::Mutex _gen_mutex;
  typename ExecutionPolicy::Mutex _time_mutex;

  double _epsilon_moving_average;
  std::list<double> _epsilons;

  struct StateNode;

  struct ActionNode {
    typedef typename SetTypeDeducer<ActionNode, Action>::Set ActionSet;
    Action action;
    typedef std::unordered_map<StateNode *, std::pair<double, std::size_t>>
        OutcomeMap; // next state nodes owned by _graph
    OutcomeMap outcomes;
    std::vector<typename OutcomeMap::iterator> dist_to_outcome;
    std::discrete_distribution<> dist;
    std::size_t expansions_count;
    std::vector<double> value;
    double all_value;
    StateNode *parent;

    ActionNode(const Action &a);
    ActionNode(const ActionNode &a);

    struct Key {
      const Action &operator()(const ActionNode &an) const;
    };
  };

  struct StateNode {
    typedef typename SetTypeDeducer<ActionNode, Action>::Set ActionSet;
    State state;
    ActionNode *action; // best action in the graph
    std::unique_ptr<Action>
        best_action; // can be an action in the graph or the heuristic one
    std::size_t expansions_count;
    ActionSet actions;
    std::vector<double> value;
    double all_value;
    std::vector<bool> goal;
    bool all_goal;
    std::vector<bool> termination;
    bool all_termination;
    std::unordered_set<ActionNode *> parents;

    StateNode(const State &s);
    StateNode(const StateNode &s);

    struct Key {
      const State &operator()(const StateNode &sn) const;
    };
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  Graph _graph;
  StateNode *_current_state;
  std::size_t _nb_rollouts;
  std::size_t _nb_agents;
  std::vector<Agent> _agents;
  std::vector<std::vector<std::size_t>> _agents_orders;
  std::bernoulli_distribution _action_choice_noise_dist;

  void expand_state(StateNode *s);
  StateNode *expand_action(ActionNode *a);
  bool generate_more_actions(StateNode *s);
  ActionNode *greedy_action(StateNode *s);
  StateNode *pick_next_state(ActionNode *a);
  void backtrack_values(StateNode *s);
  void initialize_node(StateNode &n, const Predicate &termination);
  void trial(StateNode *s,
             const std::chrono::time_point<std::chrono::high_resolution_clock>
                 &start_time);
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

#endif // SKDECIDE_MA_RTDP_HH
