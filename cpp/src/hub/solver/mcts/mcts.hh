/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * This is the scikit-decide implementation of MCTS and UCT from
 * "A Survey of Monte Carlo Tree Search Methods" by Browne et al
 * (IEEE Transactions on Computational Intelligence  and AI in games,
 * 2012). We additionally implement a heuristic value estimate as in
 * "Monte-Carlo tree search and rapid action value estimation in
 * computer Go" by Gelly and Silver (Artificial Intelligence, 2011)
 * except that the heuristic estimate is called on states but not
 * on state-action pairs to be more in line with heuristic search
 * algorithms in the literature and other implementations of
 * heuristic search algorithms in scikit-decide.
 */
#ifndef SKDECIDE_MCTS_HH
#define SKDECIDE_MCTS_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <list>
#include <chrono>
#include <random>

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace skdecide {

/** Use Environment domain knowledge for transitions */
template <typename Tsolver> struct StepTransitionMode {
  void init_rollout(Tsolver &solver, const std::size_t *thread_id) const;

  typename Tsolver::Domain::EnvironmentOutcome
  random_next_outcome(Tsolver &solver, const std::size_t *thread_id,
                      const typename Tsolver::Domain::State &state,
                      const typename Tsolver::Domain::Action &action) const;

  typename Tsolver::StateNode *
  random_next_node(Tsolver &solver, const std::size_t *thread_id,
                   typename Tsolver::ActionNode &action) const;
};

/** Use Simulation domain knowledge for transitions */
template <typename Tsolver> struct SampleTransitionMode {
  void init_rollout(Tsolver &solver, const std::size_t *thread_id) const;

  typename Tsolver::Domain::EnvironmentOutcome
  random_next_outcome(Tsolver &solver, const std::size_t *thread_id,
                      const typename Tsolver::Domain::State &state,
                      const typename Tsolver::Domain::Action &action) const;

  typename Tsolver::StateNode *
  random_next_node(Tsolver &solver, const std::size_t *thread_id,
                   typename Tsolver::ActionNode &action) const;
};

/** Use uncertain transitions domain knowledge for transitions */
template <typename Tsolver> struct DistributionTransitionMode {
  void init_rollout(Tsolver &solver, const std::size_t *thread_id) const;

  typename Tsolver::Domain::EnvironmentOutcome
  random_next_outcome(Tsolver &solver, const std::size_t *thread_id,
                      const typename Tsolver::Domain::State &state,
                      const typename Tsolver::Domain::Action &action) const;

  typename Tsolver::StateNode *
  random_next_node(Tsolver &solver, const std::size_t *thread_id,
                   typename Tsolver::ActionNode &action) const;
};

/** Default tree policy as used in UCT */
template <typename Tsolver> class DefaultTreePolicy {
public:
  typename Tsolver::StateNode *operator()(
      Tsolver &solver,
      const std::size_t *thread_id, // for parallelisation
      const typename Tsolver::Expander &expander,
      const typename Tsolver::ActionSelectorOptimization &action_selector,
      typename Tsolver::StateNode &n, std::size_t &d) const;
};

/** Test if a given node needs to be expanded by assuming that applicable
 * actions and next states can be enumerated. Returns nullptr if all actions and
 * outcomes have already been tried, otherwise a sampled unvisited outcome
 * according to its probability (among only unvisited outcomes). REQUIREMENTS:
 * returns nullptr if all actions have been already tried, and set the terminal
 *  flag of the returned next state
 */
template <typename Tsolver> class FullExpand {
public:
  typedef std::function<std::pair<typename Tsolver::Domain::Value, std::size_t>(
      typename Tsolver::Domain &, const typename Tsolver::Domain::State &,
      const std::size_t *)>
      HeuristicFunctor;

  FullExpand(
      const HeuristicFunctor &heuristic =
          [](typename Tsolver::Domain &domain,
             const typename Tsolver::Domain::State &state,
             const std::size_t *thread_id) {
            // MSVC versions earlier than 2019 cannot catch
            // Tsolver::Domain::Value inside the lambda function but can only
            // catch types through FullExpand<Tsolver> thus we must get the
            // state value type from FullExpand<Tsolver>::HeuristicFunctor
            typedef typename FullExpand<
                Tsolver>::HeuristicFunctor::result_type::first_type StateValue;
            return std::make_pair(StateValue(), 0);
          });

  FullExpand(const FullExpand &other);

  virtual ~FullExpand(); // required to prevent implicit deletion of
                         // _action_expander before defining
                         // ExpandActionImplementation

  typename Tsolver::StateNode *operator()(Tsolver &solver,
                                          const std::size_t *thread_id,
                                          typename Tsolver::StateNode &n) const;

private:
  HeuristicFunctor _heuristic;

  typename Tsolver::StateNode *
  expand_action(Tsolver &solver, const std::size_t *thread_id,
                typename Tsolver::StateNode &state,
                typename Tsolver::ActionNode &action) const;
  struct ExpandActionImplementation;
  std::unique_ptr<ExpandActionImplementation> _action_expander;
};

/** Test if a given node needs to be expanded by sampling applicable actions and
 * next states. Tries to sample new outcomes with a probability proportional to
 * the number of actual expansions. Returns nullptr if we cannot sample new
 * outcomes, otherwise a sampled unvisited outcome according to its probability
 * (among only unvisited outcomes). REQUIREMENTS: returns nullptr if all actions
 * have been already tried, and set the terminal flag of the returned next state
 */
template <typename Tsolver> class PartialExpand {
public:
  typedef std::function<std::pair<typename Tsolver::Domain::Value, std::size_t>(
      typename Tsolver::Domain &, const typename Tsolver::Domain::State &,
      const std::size_t *)>
      HeuristicFunctor;

  PartialExpand(
      const double &state_expansion_rate = 0.1,
      const double &action_expansion_rate = 0.1,
      const HeuristicFunctor &heuristic =
          [](typename Tsolver::Domain &domain,
             const typename Tsolver::Domain::State &state,
             const std::size_t *thread_id) {
            // MSVC cannot catch Tsolver::Domain::Value inside the lambda
            // function but can only catch types through PartialExpand<Tsolver>
            // thus we must get the state value type from
            // PartialExpand<Tsolver>::HeuristicFunctor
            typedef typename PartialExpand<
                Tsolver>::HeuristicFunctor::result_type::first_type StateValue;
            return std::make_pair(StateValue(), 0);
          });

  PartialExpand(const PartialExpand &other);

  typename Tsolver::StateNode *operator()(Tsolver &solver,
                                          const std::size_t *thread_id,
                                          typename Tsolver::StateNode &n) const;

private:
  HeuristicFunctor _heuristic;
  double _state_expansion_rate;
  double _action_expansion_rate;
};

/** UCB1 Best Child */
template <typename Tsolver> class UCB1ActionSelector {
public:
  // 1/sqrt(2) is a good compromise for rewards in [0;1]
  UCB1ActionSelector(double ucb_constant = 1.0 / std::sqrt(2.0));

  UCB1ActionSelector(const UCB1ActionSelector &other);

  typename Tsolver::ActionNode *
  operator()(Tsolver &solver, const std::size_t *thread_id,
             const typename Tsolver::StateNode &n) const;

private:
  typename Tsolver::ExecutionPolicy::template atomic<double> _ucb_constant;
};

/** Select action with maximum Q-value */
template <typename Tsolver> class BestQValueActionSelector {
public:
  typename Tsolver::ActionNode *
  operator()(Tsolver &solver, const std::size_t *thread_id,
             const typename Tsolver::StateNode &n) const;
};

/** Default rollout policy */
template <typename Tsolver> class DefaultRolloutPolicy {
public:
  typedef std::function<typename Tsolver::Domain::Action(
      typename Tsolver::Domain &, const typename Tsolver::Domain::State &,
      const std::size_t *)>
      PolicyFunctor;

  DefaultRolloutPolicy(
      const PolicyFunctor &policy =
          [](typename Tsolver::Domain &domain,
             const typename Tsolver::Domain::State &state,
             const std::size_t *thread_id) {
            return domain.get_applicable_actions(state, thread_id).sample();
          });

  void operator()(Tsolver &solver, const std::size_t *thread_id,
                  typename Tsolver::StateNode &n, std::size_t d) const;

private:
  PolicyFunctor _policy;
};

/** Void rollout policy */
template <typename Tsolver> class VoidRolloutPolicy {
public:
  VoidRolloutPolicy() {}
  void operator()(Tsolver &solver, const std::size_t *thread_id,
                  typename Tsolver::StateNode &n, std::size_t d) const {}
};

/** Graph backup: update Q values using the graph ancestors (rather than
 * only the trajectory leading to n) */
template <typename Tsolver> struct GraphBackup {
  void operator()(Tsolver &solver, const std::size_t *thread_id,
                  typename Tsolver::StateNode &n) const;

  static void update_frontier(
      Tsolver &solver,
      std::unordered_set<typename Tsolver::StateNode *> &new_frontier,
      typename Tsolver::StateNode *f);
  struct UpdateFrontierImplementation;
};

/**
 * @brief This is the skdecide implementation of MCTS and UCT from
 * "A Survey of Monte Carlo Tree Search Methods" by Browne et al
 * (IEEE Transactions on Computational Intelligence  and AI in games,
 * 2012). We additionally implement a heuristic value estimate as in
 * "Monte-Carlo tree search and rapid action value estimation in
 * computer Go" by Gelly and Silver (Artificial Intelligence, 2011)
 * except that the heuristic estimate is called on states but not
 * on state-action pairs to be more in line with heuristic search
 * algorithms in the literature and other implementations of
 * heuristic search algorithms in skdecide.
 *
 * @tparam Tdomain Type of the domain class
 * @tparam TexecutionPolicy Type of the execution policy (one of
 * 'SequentialExecution' to execute rollouts in sequence, or 'ParallelExecution'
 * to execute rollouts in parallel on different threads)
 * @tparam TtransitionMode Type of transition mode (one of 'StepTransitionMode',
 * 'SampleTransitionMode' or 'DistributionTransitionMode' to progress the
 * trajectories with, respectively, the 'step' or 'sample' or
 * 'get_next_state_distribution' method of the domain depending on the
 * domain's dynamics capabilities)
 * @tparam TtreePolicy Type of the tree policy class (currently only
 * 'DefaultTreePolicy' which rollouts a random trajectory from the current root
 * solving state until reaching a non-expanded state node of the tree)
 * @tparam Texpander Type of the expander class when a state needs to be
 * expanded (one of: 'FullExpand' if applicable actions and next states
 * should be all enumerated for each transition function, or 'PartialExpand' if
 * they should be sampled with a probability which exponentially decreases as
 * the number of already discovered applicable actions and next states
 * increases)
 * @tparam TactionSelectorOptimization Type of the action selector class used to
 * select actions in the tree policy's trajectory simulations (one of:
 * 'UCB1ActionSelector' to select the action based on the UCB criterion, or
 * 'BestQValueActionSelector' to select the action with maximum Q-Value in the
 * current state node)
 * @tparam TactionSelectorExecution Type of the action selector class used to
 * select actions at execution time when the 'get_best_action' method of the
 * class is invoked in a given execution state (one of: 'UCB1ActionSelector' to
 * select the action based on the UCB criterion, or 'BestQValueActionSelector'
 * to select the action with maximum Q-Value in the current state node)
 * @tparam TrolloutPolicy Type of the rollout policy class (one of:
 * 'DefaultRolloutPolicy' to simulate trajectories starting in a non-expanded
 * state node of the tree by applying actions from a given policy or by sampling
 * random applicable actions in each visited state if no policy is provided, or
 * 'VoidRolloutPolicy' to deactivate the simulation of trajectories from
 * non-expanded state nodes, in which latter case it is advised to provide a
 * heuristic function in the constructor of the expander instance to initialize
 * non-expanded state nodes' values)
 * @tparam TbackPropagator Type of the back propagator class (currently only
 * 'GraphBackup' which back-propagates empirical Q-values from non-expanded
 * state nodes up to the root node of the tree along the tree policy's sampled
 * trajectories)
 */
template <typename Tdomain, typename TexecutionPolicy = SequentialExecution,
          template <typename Tsolver> class TtransitionMode =
              DistributionTransitionMode,
          template <typename Tsolver> class TtreePolicy = DefaultTreePolicy,
          template <typename Tsolver> class Texpander = FullExpand,
          template <typename Tsolver> class TactionSelectorOptimization =
              UCB1ActionSelector,
          template <typename Tsolver> class TactionSelectorExecution =
              BestQValueActionSelector,
          template <typename Tsolver> class TrolloutPolicy =
              DefaultRolloutPolicy,
          template <typename Tsolver> class TbackPropagator = GraphBackup>
class MCTSSolver {
public:
  typedef MCTSSolver<Tdomain, TexecutionPolicy, TtransitionMode, TtreePolicy,
                     Texpander, TactionSelectorOptimization,
                     TactionSelectorExecution, TrolloutPolicy, TbackPropagator>
      Solver;

  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef TexecutionPolicy ExecutionPolicy;
  typedef TtransitionMode<Solver> TransitionMode;
  typedef TtreePolicy<Solver> TreePolicy;
  typedef Texpander<Solver> Expander;
  typedef TactionSelectorOptimization<Solver> ActionSelectorOptimization;
  typedef TactionSelectorExecution<Solver> ActionSelectorExecution;
  typedef TrolloutPolicy<Solver> RolloutPolicy;
  typedef TbackPropagator<Solver> BackPropagator;

  typedef typename ExecutionPolicy::template atomic<std::size_t> atomic_size_t;
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;
  typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

  struct StateNode;

  struct ActionNode {
    Action action;
    typedef std::unordered_map<StateNode *, std::pair<double, std::size_t>>
        OutcomeMap; // next state nodes owned by _graph
    OutcomeMap outcomes;
    std::vector<typename OutcomeMap::iterator> dist_to_outcome;
    std::discrete_distribution<> dist;
    atomic_size_t expansions_count; // used only for partial expansion mode
    atomic_double value;
    atomic_size_t visits_count;
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
    atomic_bool terminal;
    atomic_bool expanded;           // used only for full expansion mode
    atomic_size_t expansions_count; // used only for partial expansion mode
    ActionSet actions;
    atomic_double value;
    atomic_size_t visits_count;
    std::unordered_set<ActionNode *> parents;
    mutable typename ExecutionPolicy::Mutex mutex;

    StateNode(const State &s);
    StateNode(const StateNode &s);

    struct Key {
      const State &operator()(const StateNode &sn) const;
    };
  };

  typedef typename SetTypeDeducer<StateNode, State>::Set Graph;
  typedef std::function<bool(const MCTSSolver &, Domain &, const std::size_t *)>
      CallbackFunctor;

  /**
   * @brief Constructs a new MCTSSolver object
   *
   * @param domain The domain instance
   * @param time_budget Maximum solving time in milliseconds
   * @param rollout_budget Maximum number of rollouts (deactivated when
   * use_labels is true)
   * @param max_depth Maximum depth of each MCTS rollout
   * @param residual_moving_average_window Number of latest computed residual
   * values to memorize in order to compute the average Bellman error (residual)
   * at the root state of the search
   * @param epsilon Maximum Bellman error (residual) allowed to decide that a
   * state is solved, or to decide when no labels are used that the value
   * function of the root state of the search has converged (in the latter case:
   * the root state's Bellman error is averaged over the
   * residual_moving_average_window)
   * @param discount Value function's discount factor
   * @param online_node_garbage Boolean indicating whether the search graph
   * which is no more reachable from the root solving state should be
   * deleted (true) or not (false)
   * @param callback Functor called at the end of each MCTS trial rollout,
   * taking as arguments the solver, the domain and the thread ID from which it
   * is called, and returning true if the solver must be stopped
   * @param verbose Boolean indicating whether verbose messages should be
   * logged (true) or not (false)
   * @param tree_policy tree policy instance in charge of simulating
   * trajectories from the root solving state down to some non-expanded state
   * node
   * @param expander expander instance in charge of expanding non-expanded state
   * nodes
   * @param action_selector_optimization action selector instance in charge of
   * selecting actions in the tree policy's trajectory simulations
   * @param action_selector_execution action selector instance in charge of
   * selecting actions at execution time when the 'get_best_action' method of
   * the class is invoked in a given execution state
   * @param rollout_policy rollout policy instance in charge of simulating
   * trajectories from a non-expanded state node to some terminal state
   * @param back_propagator back-propagator instance in charge of
   * back-propagating non-expanded state node values up to the root solving
   * state node along the tree policy's sampled trajectories
   */
  MCTSSolver(
      Domain &domain, std::size_t time_budget = 3600000,
      std::size_t rollout_budget = 100000, std::size_t max_depth = 1000,
      std::size_t residual_moving_average_window = 100,
      double epsilon = 0.0, // not a stopping criterion by default
      double discount = 1.0, bool online_node_garbage = false,
      const CallbackFunctor &callback =
          [](const MCTSSolver &, Domain &, const std::size_t *) {
            return false;
          },
      bool verbose = false,
      std::unique_ptr<TreePolicy> tree_policy = std::make_unique<TreePolicy>(),
      std::unique_ptr<Expander> expander = std::make_unique<Expander>(),
      std::unique_ptr<ActionSelectorOptimization> action_selector_optimization =
          std::make_unique<ActionSelectorOptimization>(),
      std::unique_ptr<ActionSelectorExecution> action_selector_execution =
          std::make_unique<ActionSelectorExecution>(),
      std::unique_ptr<RolloutPolicy> rollout_policy =
          std::make_unique<RolloutPolicy>(),
      std::unique_ptr<BackPropagator> back_propagator =
          std::make_unique<BackPropagator>());

  /**
   * @brief Clears the search graph, thus preventing from reusing previous
   * search results)
   *
   */
  void clear();

  /**
   * @brief Call the MCTS algorithm
   *
   * @param s Root state of the search from which MCTS rollouts are launched
   */
  void solve(const State &s);

  /**
   * @brief Indicates whether the solution policy is defined for a given state
   *
   * @param s State for which an entry is searched in the policy graph
   * @return true If the state has been explored and an action can be obtained
   * from the execution action selector
   * @return false If the state has not been explored or no action can be
   * obtained from the execution action selector
   */
  bool is_solution_defined_for(const State &s);

  /**
   * @brief Get the best action to execute in a given state according to the
   * execution action selector (throws a runtime error exception if no action
   * can be obtained from the execution action selector, which is why it is
   * advised to call MCTSSolver::is_solution_defined_for before). The search
   * subgraph which is no more reachable after executing the returned action is
   * also deleted if node garbage was set to true in the MCTSSolver instance's
   * constructor.
   *
   * @param s State for which the best action is requested
   * @return const Action& Best action to execute according to the execution
   * action selector
   */
  Action get_best_action(const State &s);

  /**
   * @brief Get the best value in a given state according to the
   * execution action selector (throws a runtime error exception if no action is
   * defined in the given state, which is why it is advised to call
   * MCTSSolver::is_solution_defined_for before)
   *
   * @param s State from which the best value is requested
   * @return double Value of the action returned by the execution action
   * selector
   */
  Value get_best_value(const State &s);

  /**
   * @brief Get the number of states present in the search graph (which can be
   * lower than the number of actually explored states if node garbage was
   * set to true in the MCTSSolver instance's constructor)
   *
   * @return std::size_t Number of states present in the search graph
   */
  std::size_t get_nb_explored_states();

  /**
   * @brief Get the number of rollouts since the beginning of the search from
   * the root solving state
   *
   * @return std::size_t Number of MCTS rollouts
   */
  std::size_t get_nb_rollouts() const;

  /**
   * @brief Get the average Bellman error (residual)
   * at the root state of the search, or an infinite value if the number of
   * computed residuals is lower than the epsilon moving average window set in
   * the MCTSSolver instance's constructor
   *
   * @return double Bellman error at the root state of the search averaged over
   * the epsilon moving average window
   */
  double get_residual_moving_average();

  /**
   * @brief Get the solving time in milliseconds since the beginning of the
   * search from the root solving state
   *
   * @return std::size_t Solving time in milliseconds
   */
  std::size_t get_solving_time();

  /**
   * @brief Get the (partial) solution policy defined for the states for which
   * the best value according to the execution action selector has been updated
   * at least once (which is optimal if the algorithm has converged and labels
   * are used); warning: only defined over the states reachable from the last
   * root solving state when node garbage was set to True in the MCTSSolver
   * instance's constructor
   *
   * @return Mapping from states to pairs of action and best value according to
   * the execution action selector
   */
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map get_policy();

  /**
   * @brief Get the list of actions returned by the solver so far after each
   * call to the MCTSSolver::get_best_action method (mostly internal use in
   * order to rebuild the sequence of visited states until reaching the current
   * solving state, when using the 'StepTransitionMode' for which we can
   * only progress the transition function with steps that hide the current
   * state of the domain)
   *
   * @return std::list<Action> List of actions executed by the solver so far
   * after each call to the MCTSSolver::get_best_action method
   */
  const std::list<Action> &action_prefix() const;

  Domain &domain();
  std::size_t time_budget() const;
  std::size_t rollout_budget() const;
  std::size_t max_depth() const;
  double discount() const;

  ExecutionPolicy &execution_policy();
  const TransitionMode &transition_mode();
  const TreePolicy &tree_policy();
  const Expander &expander();
  const ActionSelectorOptimization &action_selector_optimization();
  const ActionSelectorExecution &action_selector_execution();
  const RolloutPolicy &rollout_policy();
  const BackPropagator &back_propagator();

  Graph &graph();
  std::mt19937 &gen();
  typename ExecutionPolicy::Mutex &gen_mutex();
  bool verbose() const;

private:
  Domain &_domain;
  atomic_size_t _time_budget;
  atomic_size_t _rollout_budget;
  atomic_size_t _max_depth;
  atomic_size_t _residual_moving_average_window;
  atomic_double _epsilon;
  atomic_double _discount;
  atomic_size_t _nb_rollouts;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
  bool _online_node_garbage;
  CallbackFunctor _callback;
  atomic_bool _verbose;

  std::unique_ptr<ExecutionPolicy> _execution_policy;
  std::unique_ptr<TransitionMode> _transition_mode;

  std::unique_ptr<TreePolicy> _tree_policy;
  std::unique_ptr<Expander> _expander;
  std::unique_ptr<ActionSelectorOptimization> _action_selector_optimization;
  std::unique_ptr<ActionSelectorExecution> _action_selector_execution;
  std::unique_ptr<RolloutPolicy> _rollout_policy;
  std::unique_ptr<BackPropagator> _back_propagator;

  Graph _graph;
  StateNode *_current_state;
  std::list<Action> _action_prefix;

  std::unique_ptr<std::mt19937> _gen;
  typename ExecutionPolicy::Mutex _gen_mutex;
  typename ExecutionPolicy::Mutex _time_mutex;
  typename ExecutionPolicy::Mutex _residuals_protect;

  atomic_double _residual_moving_average;
  std::list<double> _residuals;

  void compute_reachable_subgraph(StateNode *node,
                                  std::unordered_set<StateNode *> &subgraph);
  void remove_subgraph(std::unordered_set<StateNode *> &root_subgraph,
                       std::unordered_set<StateNode *> &child_subgraph);
  void update_residual_moving_average(const StateNode &node,
                                      const double &node_record_value);
}; // MCTSSolver class

/** UCT as described in the paper " Bandit Based Monte-Carlo Planning" by
 * Levente Kocsis and Csaba Szepesvari (ECML 2006) is a famous variant of MCTS
 * with some specific options including the famous UCB1 action selector to
 * perform tree exploration */
template <typename Tdomain, typename Texecution_policy,
          template <typename Tsolver> typename TtransitionMode>
using UCTSolver =
    MCTSSolver<Tdomain, Texecution_policy, TtransitionMode, DefaultTreePolicy,
               FullExpand, UCB1ActionSelector, BestQValueActionSelector,
               DefaultRolloutPolicy, GraphBackup>;

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/mcts_impl.hh"
#include "impl/mcts_step_transition_mode_impl.hh"
#include "impl/mcts_sample_transition_mode_impl.hh"
#include "impl/mcts_distribution_transition_mode_impl.hh"
#include "impl/mcts_default_tree_policy_impl.hh"
#include "impl/mcts_full_expand_impl.hh"
#include "impl/mcts_partial_expand_impl.hh"
#include "impl/mcts_ucb1_action_selector_impl.hh"
#include "impl/mcts_best_qvalue_action_selector_impl.hh"
#include "impl/mcts_default_rollout_policy_impl.hh"
#include "impl/mcts_graph_backup_impl.hh"
#endif

#endif // SKDECIDE_MCTS_HH
