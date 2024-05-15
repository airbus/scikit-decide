/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_RIW_HH
#define SKDECIDE_RIW_HH

#include <functional>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <list>
#include <chrono>
#include <random>

#include <boost/container_hash/hash.hpp>

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace skdecide {

/** Use default hasher provided with domain's states */
template <typename Tdomain, typename Tfeature_vector> struct DomainStateHash {
  typedef typename Tdomain::State Key;

  template <typename Tnode> static const Key &get_key(const Tnode &n);

  struct Hash {
    std::size_t operator()(const Key &k) const;
  };

  struct Equal {
    bool operator()(const Key &k1, const Key &k2) const;
  };
};

/** Use state binary feature vector to hash states */
template <typename Tdomain, typename Tfeature_vector> struct StateFeatureHash {
  typedef Tfeature_vector Key;

  template <typename Tnode> static const Key &get_key(const Tnode &n);

  struct Hash {
    std::size_t operator()(const Key &k) const;
  };

  struct Equal {
    bool operator()(const Key &k1, const Key &k2) const;
  };
};

/** Use Environment domain knowledge for rollouts */
template <typename Tdomain> struct EnvironmentRollout {
  std::list<typename Tdomain::Action> _action_prefix;

  void init_rollout(Tdomain &domain, const std::size_t *thread_id);

  typename Tdomain::EnvironmentOutcome
  progress(Tdomain &domain, const typename Tdomain::State &state,
           const typename Tdomain::Action &action,
           const std::size_t *thread_id);

  void advance(Tdomain &domain, const typename Tdomain::State &state,
               const typename Tdomain::Action &action, bool record_action,
               const std::size_t *thread_id);

  std::list<typename Tdomain::Action> action_prefix() const;
};

/** Use Simulation domain knowledge for rollouts */
template <typename Tdomain> struct SimulationRollout {
  void init_rollout([[maybe_unused]] Tdomain &domain,
                    [[maybe_unused]] const std::size_t *thread_id);

  typename Tdomain::EnvironmentOutcome
  progress(Tdomain &domain, const typename Tdomain::State &state,
           const typename Tdomain::Action &action,
           const std::size_t *thread_id);

  void advance([[maybe_unused]] Tdomain &domain,
               [[maybe_unused]] const typename Tdomain::State &state,
               [[maybe_unused]] const typename Tdomain::Action &action,
               [[maybe_unused]] bool record_action,
               [[maybe_unused]] const std::size_t *thread_id);

  std::list<typename Tdomain::Action> action_prefix() const;
};

/**
 * @brief This is the skdecide implementation of "Planning with Pixels in
 * (Almost) Real Time" by Wilmer Bandres, Blai Bonet, Hector Geffner (AAAI 2018)
 *
 * @tparam Tdomain Type of the domain class
 * @tparam Tfeature_vector Type of of the state feature vector used to compute
 * the novelty measure
 * @tparam Thashing_policy Type of the hashing class used to hash states (one of
 * 'DomainStateHash' to use the state hash function, or 'StateFeatureHash' to
 * hash states based on their features)
 * @tparam Trollout_policy Type of the rollout policy (one of
 * 'EnvironmentRollout' to progress the trajectories with the 'step' domain
 * method, or 'SimulationRollout' to progress the trajectories with the 'sample'
 * domain method depending on the domain's dynamics capabilities)
 * @tparam Texecution_policy Type of the execution policy (one of
 * 'SequentialExecution' to execute rollouts in sequence, or 'ParallelExecution'
 * to execute rollouts in parallel on different threads)
 */
template <typename Tdomain, typename Tfeature_vector,
          template <typename...> class Thashing_policy = DomainStateHash,
          template <typename...> class Trollout_policy = EnvironmentRollout,
          typename Texecution_policy = SequentialExecution>
class RIWSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef Tfeature_vector FeatureVector;
  typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
  typedef Trollout_policy<Domain> RolloutPolicy;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<std::unique_ptr<FeatureVector>(
      Domain &d, const State &s, const std::size_t *thread_id)>
      StateFeatureFunctor;
  typedef std::function<bool(const RIWSolver &, Domain &, const std::size_t *)>
      CallbackFunctor;

  /**
   * @brief Construct a new RIWSolver object
   *
   * @param domain The domain instance
   * @param state_features State feature vector used to compute the novelty
   * measure
   * @param time_budget Maximum solving time in milliseconds
   * @param rollout_budget Maximum number of rollouts (deactivated when
   * use_labels is true)
   * @param max_depth Maximum depth of each LRTDP trial (rollout)
   * @param exploration Probability of choosing a non-solved child of a given
   * node (more precisely, a first-time explored child is chosen with a
   * probability 'exploration', and a already-explored but non-solved child is
   * chosen with a probability of '1 - exploration' divided by its novelty
   * measure; probabilities among children are then normalized)
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
   * @param debug_logs Boolean indicating whether debugging messages should be
   * logged (true) or not (false)
   * @param callback Functor called at the end of each RIW rollout,
   * taking as arguments the solver, the domain and the thread ID from which it
   * is called, and returning true if the solver must be stopped
   */
  RIWSolver(
      Domain &domain, const StateFeatureFunctor &state_features,
      std::size_t time_budget = 3600000, std::size_t rollout_budget = 100000,
      std::size_t max_depth = 1000, double exploration = 0.25,
      std::size_t residual_moving_average_window = 100, double epsilon = 0.001,
      double discount = 1.0, bool online_node_garbage = false,
      bool debug_logs = false,
      const CallbackFunctor &callback = [](const RIWSolver &, Domain &,
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
   * @param s Root state of the search from which RIW rollouts are launched
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
  bool is_solution_defined_for(const State &s);

  /**
   * @brief Get the best computed action in terms of best Q-value in a given
   * state (throws a runtime error exception if no action is defined in the
   * given state, which is why it is advised to call
   * RIWSolver::is_solution_defined_for before).  The search
   * subgraph which is no more reachable after executing the returned action is
   * also deleted if node garbage was set to true in the RIWSolver instance's
   * constructor.
   *
   * @param s State for which the best action is requested
   * @return Action Best computed action
   */
  Action get_best_action(const State &s);

  /**
   * @brief Get the best Q-value in a given state (throws a runtime
   * error exception if no action is defined in the given state, which is why it
   * is advised to call RIWSolver::is_solution_defined_for before)
   *
   * @param s State from which the best Q-value is requested
   * @return double Maximum Q-value of the given state over the applicable
   * actions in this state
   */
  Value get_best_value(const State &s);

  /**
   * @brief Get the number of states present in the search graph (which can be
   * lower than the number of actually explored states if node garbage was
   * set to true in the RIWSolver instance's constructor)
   *
   * @return std::size_t Number of states present in the search graph
   */
  std::size_t get_nb_explored_states();

  /**
   * @brief Get the number of states present in the search graph that have been
   * pruned by the novelty test (which can be lower than the number of actually
   * explored states if node garbage was set to true in the RIWSolver
   * instance's constructor)
   *
   * @return std::size_t Number of states present in the search graph that have
   * been pruned by the novelty test
   */
  std::size_t get_nb_pruned_states();

  /**
   * @brief Get the exploration statistics as number of states present in the
   * search graph and number of such states that have been pruned by the novelty
   * test (both statistics can be lower than the number of actually
   * explored states if node garbage was set to true in the RIWSolver
   * instance's constructor)
   *
   * @return std::pair<std::size_t, std::size_t> Pair of number of states
   * present in the search graph and of number of such states that have been
   * pruned by the novelty test
   */
  std::pair<std::size_t, std::size_t> get_exploration_statistics();

  /**
   * @brief Get the number of rollouts since the beginning of the search from
   * the root solving state
   *
   * @return std::size_t Number of RIW rollouts
   */
  std::size_t get_nb_rollouts() const;

  /**
   * @brief Get the average Bellman error (residual)
   * at the root state of the search, or an infinite value if the number of
   * computed residuals is lower than the epsilon moving average window set in
   * the LRTDP instance's constructor
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
   * the Q-value has been updated at least once (which is optimal if the
   * algorithm has converged and labels are used); warning: only defined over
   * the states reachable from the last root solving state when node garbage was
   * set to true in the RIWSolver instance's constructor
   *
   * @return Mapping from states to pairs of action and best Q-value
   */
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map get_policy();

  /**
   * @brief Get the list of actions returned by the solver so far after each
   * call to the RIWSolver::get_best_action method (mostly internal use in order
   * to rebuild the sequence of visited states until reaching the current
   * solving state, when using the 'EnvironmentRollout' policy for which we can
   * only progress the transition function with steps that hide the current
   * state of the environment)
   *
   * @return std::list<Action> List of actions executed by the solver so far
   * after each call to the RIWSolver::get_best_action method
   */
  std::list<Action> action_prefix() const;

private:
  typedef typename ExecutionPolicy::template atomic<std::size_t> atomic_size_t;
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;
  typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

  Domain &_domain;
  StateFeatureFunctor _state_features;
  atomic_size_t _time_budget;
  atomic_size_t _rollout_budget;
  atomic_size_t _max_depth;
  atomic_double _exploration;
  atomic_size_t _residual_moving_average_window;
  atomic_double _epsilon;
  atomic_double _discount;
  bool _online_node_garbage;
  atomic_double _min_reward;
  atomic_size_t _nb_rollouts;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
  RolloutPolicy _rollout_policy;
  ExecutionPolicy _execution_policy;
  atomic_bool _debug_logs;
  CallbackFunctor _callback;

  std::unique_ptr<std::mt19937> _gen;
  typename ExecutionPolicy::Mutex _gen_mutex;
  typename ExecutionPolicy::Mutex _time_mutex;
  typename ExecutionPolicy::Mutex _residuals_protect;

  atomic_double _residual_moving_average;
  std::list<double> _residuals;

  struct Node {
    State state;
    std::unique_ptr<FeatureVector> features;
    std::vector<std::tuple<Action, double, Node *>> children;
    std::unordered_set<Node *> parents;
    atomic_double value;
    atomic_size_t depth;
    atomic_size_t novelty;
    Action *best_action;
    atomic_bool
        terminal; // true if seen terminal from the simulator's perspective
    atomic_bool pruned; // true if pruned from novelty measure perspective
    atomic_bool solved; // from this node: true if all reached states are either
                        // max_depth, or terminal or pruned
    mutable typename ExecutionPolicy::Mutex mutex;

    Node(const State &s, Domain &d, const StateFeatureFunctor &state_features,
         const std::size_t *thread_id);

    Node(const Node &n);

    struct Key {
      const typename HashingPolicy::Key &operator()(const Node &n) const;
    };
  };

  typedef typename SetTypeDeducer<Node, HashingPolicy>::Set Graph;
  Graph _graph;

  typedef std::vector<
      std::pair<std::size_t, typename FeatureVector::value_type>>
      TupleType; // pair of var id and var value
  typedef std::vector<
      std::unordered_map<TupleType, std::size_t, boost::hash<TupleType>>>
      TupleVector; // mapped to min reached depth

  class WidthSolver { // known as IW(i), i.e. the fixed-width solver
                      // sequentially run by IW
  public:
    typedef Tdomain Domain;
    typedef typename Domain::State State;
    typedef typename Domain::Action Action;
    typedef Tfeature_vector FeatureVector;
    typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
    typedef Trollout_policy<Domain> RolloutPolicy;
    typedef Texecution_policy ExecutionPolicy;

    typedef typename ExecutionPolicy::template atomic<std::size_t>
        atomic_size_t;
    typedef typename ExecutionPolicy::template atomic<double> atomic_double;
    typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

  public:
    WidthSolver(
        RIWSolver &parent_solver, Domain &domain,
        const StateFeatureFunctor &state_features,
        const atomic_size_t &time_budget, const atomic_size_t &rollout_budget,
        const atomic_size_t &max_depth, const atomic_double &exploration,
        const atomic_size_t &residual_moving_average_window,
        const atomic_double &epsilon, atomic_double &residual_moving_average,
        std::list<double> &residuals, const atomic_double &discount,
        atomic_double &min_reward, const atomic_size_t &width, Graph &graph,
        RolloutPolicy &rollout_policy, ExecutionPolicy &execution_policy,
        std::mt19937 &gen, typename ExecutionPolicy::Mutex &gen_mutex,
        typename ExecutionPolicy::Mutex &time_mutex,
        typename ExecutionPolicy::Mutex &residuals_protect,
        const atomic_bool &debug_logs, const CallbackFunctor &callback);

    // solves from state s
    // return true iff no state has been pruned or time or rollout budgets are
    // consumed
    bool solve(const State &s, atomic_size_t &nb_rollouts,
               TupleVector &feature_tuples);

  private:
    RIWSolver &_parent_solver;
    Domain &_domain;
    const StateFeatureFunctor &_state_features;
    const atomic_size_t &_time_budget;
    const atomic_size_t &_rollout_budget;
    const atomic_size_t &_max_depth;
    const atomic_double &_exploration;
    const atomic_size_t &_residual_moving_average_window;
    const atomic_double &_epsilon;
    atomic_double &_residual_moving_average;
    std::list<double> &_residuals;
    const atomic_double &_discount;
    atomic_double &_min_reward;
    atomic_bool _min_reward_changed;
    const atomic_size_t &_width;
    Graph &_graph;
    RolloutPolicy &_rollout_policy;
    ExecutionPolicy &_execution_policy;
    std::mt19937 &_gen;
    typename ExecutionPolicy::Mutex &_gen_mutex;
    typename ExecutionPolicy::Mutex &_time_mutex;
    typename ExecutionPolicy::Mutex &_residuals_protect;
    const atomic_bool &_debug_logs;
    const CallbackFunctor &_callback;

    void rollout(Node &root_node, TupleVector &feature_tuples,
                 atomic_size_t &nb_rollouts, atomic_bool &states_pruned,
                 atomic_bool &reached_end_of_trajectory_once,
                 const std::size_t *thread_id);

    // Input: feature tuple vector, node for which to compute novelty depth,
    // boolean indicating whether this node is new or not Returns true if at
    // least one tuple is new or is reached with lower depth
    bool novelty(TupleVector &feature_tuples, Node &n, bool nn) const;

    // Generates all combinations of size k from [0 ... (n-1)]
    void generate_tuples(const std::size_t &k, const std::size_t &n,
                         const std::function<void(TupleType &)> &f) const;

    // Get the state reachable by calling the simulator from given node by
    // applying given action number Sets given node to the next one and returns
    // whether the next one is terminal or not
    bool fill_child_node(Node *&node, std::size_t action_number, bool &new_node,
                         const std::size_t *thread_id);

    void update_node(Node &node, bool solved);

    void update_residual_moving_average(const Node &node,
                                        const double &node_record_value);
  }; // WidthSolver class

  void compute_reachable_subgraph(Node &node,
                                  std::unordered_set<Node *> &subgraph);

  // Prune the nodes that are no more reachable from the root's chosen child and
  // reduce the depth of the nodes reachable from the child node by 1
  void update_graph(std::unordered_set<Node *> &root_subgraph,
                    std::unordered_set<Node *> &child_subgraph);

  // Backup values from tip solved nodes to their parents in graph
  void backup_values(std::unordered_set<Node *> &frontier);

  void update_frontier(std::unordered_set<Node *> &new_frontier, Node *n);
  template <typename TTexecution_policy = ExecutionPolicy,
            typename Enable = void>
  struct UpdateFrontierImplementation {};
}; // RIWSolver class

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/riw_impl.hh"
#endif

#endif // SKDECIDE_RIW_HH
