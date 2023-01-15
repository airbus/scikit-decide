/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * This is the scikit-decide implementation of MCTS and UCT from
 * "A Survey of Monte Carlo Tree Search Methods" by Browne et al
 * (IEEE Transactions on Computational Intelligence  and AI in games,
 * 2012). We additionnally implement a heuristic value estimate as in
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
  typedef std::function<bool(const std::size_t &, const std::size_t &,
                             const double &, const double &)>
      WatchdogFunctor;

  MCTSSolver(
      Domain &domain, std::size_t time_budget = 3600000,
      std::size_t rollout_budget = 100000, std::size_t max_depth = 1000,
      std::size_t epsilon_moving_average_window = 100,
      double epsilon = 0.0, // not a stopping criterion by default
      double discount = 1.0, bool online_node_garbage = false,
      bool debug_logs = false,
      const WatchdogFunctor &watchdog = [](const std::size_t &,
                                           const std::size_t &, const double &,
                                           const double &) { return true; },
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

  // clears the solver (clears the search graph, thus preventing from reusing
  // previous search results)
  void clear();

  // solves from state s
  void solve(const State &s);

  bool is_solution_defined_for(const State &s);
  Action get_best_action(const State &s);
  double get_best_value(const State &s);
  std::size_t nb_of_explored_states() const;
  std::size_t nb_rollouts() const;
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map policy();

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
  const std::list<Action> &action_prefix() const;
  std::mt19937 &gen();
  typename ExecutionPolicy::Mutex &gen_mutex();
  bool debug_logs() const;

private:
  Domain &_domain;
  atomic_size_t _time_budget;
  atomic_size_t _rollout_budget;
  atomic_size_t _max_depth;
  atomic_size_t _epsilon_moving_average_window;
  atomic_double _epsilon;
  atomic_double _discount;
  atomic_size_t _nb_rollouts;
  bool _online_node_garbage;
  atomic_bool _debug_logs;
  WatchdogFunctor _watchdog;

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
  typename ExecutionPolicy::Mutex _epsilons_protect;

  atomic_double _epsilon_moving_average;
  std::list<double> _epsilons;

  void compute_reachable_subgraph(StateNode *node,
                                  std::unordered_set<StateNode *> &subgraph);
  void remove_subgraph(std::unordered_set<StateNode *> &root_subgraph,
                       std::unordered_set<StateNode *> &child_subgraph);
  std::size_t update_epsilon_moving_average(const StateNode &node,
                                            const double &node_record_value);
  std::size_t
  elapsed_time(const std::chrono::time_point<std::chrono::high_resolution_clock>
                   &start_time);
}; // MCTSSolver class

/** UCT is MCTS with the default template options */
template <typename Tdomain, typename Texecution_policy,
          template <typename Tsolver> typename TtransitionMode,
          template <typename Tsolver> typename... T>
using UCTSolver = MCTSSolver<Tdomain, Texecution_policy, TtransitionMode, T...>;

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
