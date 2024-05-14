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
#include "utils/logging.hh"

namespace skdecide {

/**
 * @brief This is an experimental implementation of a skdecide-specific
 * centralized multi-agent version of the RTDP algorithm ("Learning to act
 * using real-time dynamic programming" by Barto, Bradtke and Singh, AIJ 1995)
 * where the team's cost is the sum of individual costs and the joint applicable
 * actions in a given joint state are sampled to avoid a combinatorial explosion
 * of the joint action branching factor. This algorithm can (currently) only run
 * on a single CPU.
 *
 * @tparam Tdomain Type of the domain class
 */
template <typename Tdomain> class MARTDPSolver {
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

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<std::pair<Value, Action>(Domain &, const State &)>
      HeuristicFunctor;
  typedef std::function<bool(const MARTDPSolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Construct a new MARTDPSolver object
   *
   * @param domain The domain instance
   * @param goal_checker Functor taking as arguments the domain and a joint
   * state object, and returning true if the state is the goal
   * @param heuristic Functor taking as arguments the domain and a state, and
   * returning a pair of dictionary from agents to the individual heuristic
   * estimates from the state to the goal, and of dictionary from agents to best
   * guess individual actions the joint cost of the multi-agent domain being
   * decomposed as the sum of agents' costs)
   * @param time_budget Maximum solving time in milliseconds
   * @param rollout_budget Maximum number of rollouts
   * @param max_depth Maximum depth of each RTDP trial (rollout)
   * @param max_feasibility_trials Number of trials for a given agent's
   * applicable action to insert it in the joint applicable action set by
   * reshuffling the agents' actions applicability ordering (set to the number
   * of agents in the domain if it is equal to 0 in this constructor)
   * @param graph_expansion_rate Value $rs$ such that the probability of trying
   * to generate more joint applicable actions in a given state node with
   * already $na$ generated applicable actions is equal to $e^{-rs \cdot na}$
   * @param residual_moving_average_window Number of latest computed residual
   * values to memorize in order to compute the average Bellman error (residual)
   * at the root state of the search
   * @param epsilon Maximum Bellman error (residual) allowed to decide that a
   * state is solved, or to decide that the value function of the root state of
   * the search has converged (in the latter case: the root state's Bellman
   * error is averaged over the residual_moving_average_window)
   * @param discount Value function's discount factor
   * @param action_choice_noise Bernoulli probability of choosing an agent's
   * random applicable action instead of the best current one when trying to
   * generate a feasible joint applicable action from another agent's viewpoint
   * @param dead_end_cost Cost of a joint dead-end state (note that the
   * transition cost function which is independently decomposed over the agents
   * cannot easily model such joint dead-end state costs, which is why we allow
   * for setting this global dead-end cost in this constructor)
   * @param online_node_garbage Boolean indicating whether the search graph
   * which is no more reachable from the root solving state should be
   * deleted (true) or not (false)
   * @param debug_logs Boolean indicating whether debugging messages should be
   * logged (true) or not (false)
   * @param callback Functor called at the end of each RTDP trial (rollout),
   * taking as arguments the solver and the domain from which it
   * is called, and returning true if the solver must be stopped
   */
  MARTDPSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, std::size_t time_budget = 3600000,
      std::size_t rollout_budget = 100000, std::size_t max_depth = 1000,
      std::size_t max_feasibility_trials = 0, // will then choose nb_agents
      double graph_expansion_rate = 0.1,
      std::size_t residual_moving_average_window = 100,
      double epsilon = 0.0, // not a stopping criterion by default
      double discount = 1.0, double action_choice_noise = 0.1,
      const double &dead_end_cost = 10e4, bool online_node_garbage = false,
      bool debug_logs = false,
      const CallbackFunctor &callback = [](const MARTDPSolver &, Domain &) {
        return false;
      });

  /**
   * @brief Clears the search graph, thus preventing from reusing previous
   * search results)
   *
   */
  void clear();

  /**
   * @brief Call the MA-RTDP algorithm
   *
   * @param s Root joint state of the search from which MA-RTDP trials are
   * launched
   */
  void solve(const State &s);

  /**
   * @brief Indicates whether the solution policy is defined for a given joint
   * state
   *
   * @param s Joint state for which an entry is searched in the policy graph
   * @return true If the state has been explored and an action is defined in
   * this state
   * @return false If the state has not been explored or no action is defined in
   * this state
   */
  bool is_solution_defined_for(const State &s) const;

  /**
   * @brief Get the best computed joint action in terms of best Q-value in a
   * given joint state (throws a runtime error exception if no action is defined
   * in the given state, which is why it is advised to call
   * MARTDPSolver::is_solution_defined_for before). The search
   * subgraph which is no more reachable after executing the returned action is
   * also deleted if node garbage was set to true in the MARTDPSolver instance's
   * constructor.
   *
   * @param s Joint state for which the best action is requested
   * @return const Action& Best computed joint action
   */
  const Action &get_best_action(const State &s);

  /**
   * @brief Get the best Q-value in a given joint state (throws a runtime
   * error exception if no action is defined in the given state, which is why it
   * is advised to call MARTDPSolver::is_solution_defined_for before)
   *
   * @param s Joint state from which the best Q-value is requested
   * @return double Maximum Q-value of the given joint state over the applicable
   * joint actions in this state
   */
  Value get_best_value(const State &s) const;

  /**
   * @brief Get the number of states present in the search graph (which can be
   * lower than the number of actually explored states if node garbage was
   * set to true in the MARTDPSolver instance's constructor)
   *
   * @return std::size_t Number of states present in the search graph
   */
  std::size_t get_nb_explored_states() const;

  /**
   * @brief Get the number of joint applicable actions generated so far in the
   * given joint state (throws a runtime error exception if the given state is
   * not present in the search graph, which can happen for instance when node
   * garbage is set to true in the MARTDPSolver instance's constructor and the
   * non-reachable part of the search graph has been erased when calling the
   * MARTDPSolver::get_best_action method)
   *
   * @param s Joint state from which the number of generated applicable actions
   * is requested
   * @return const std::size_t& Number of generated applicable joint actions in
   * the given state
   */
  const std::size_t &get_state_nb_actions(const State &s) const;

  /**
   * @brief Get the number of rollouts since the beginning of the search from
   * the root solving state
   *
   * @return std::size_t Number of rollouts (RTDP trials)
   */
  std::size_t get_nb_rollouts() const;

  /**
   * @brief Get the average Bellman error (residual) at the root state of the
   * search, or an infinite value if the number of computed residuals is lower
   * than the epsilon moving average window set in the MARTDPSolver instance's
   * constructor
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
  std::size_t get_solving_time() const;

  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map policy() const;

private:
  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  std::size_t _time_budget;
  std::size_t _rollout_budget;
  std::size_t _max_depth;
  std::size_t _max_feasibility_trials;
  double _graph_expansion_rate;
  std::size_t _residual_moving_average_window;
  double _epsilon;
  double _discount;
  double _dead_end_cost;
  bool _online_node_garbage;
  bool _debug_logs;
  CallbackFunctor _callback;
  std::unique_ptr<std::mt19937> _gen;

  double _residual_moving_average;
  std::list<double> _residuals;

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
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
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
  void trial(StateNode *s);
  void compute_reachable_subgraph(StateNode *node,
                                  std::unordered_set<StateNode *> &subgraph);
  void remove_subgraph(std::unordered_set<StateNode *> &root_subgraph,
                       std::unordered_set<StateNode *> &child_subgraph);
  void update_residual_moving_average(const StateNode &node,
                                      const double &node_record_value);
};

} // namespace skdecide

#endif // SKDECIDE_MA_RTDP_HH
