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

template <typename Tdomain, typename Tfeature_vector,
          template <typename...> class Thashing_policy = DomainStateHash,
          template <typename...> class Trollout_policy = EnvironmentRollout,
          typename Texecution_policy = SequentialExecution>
class RIWSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef Tfeature_vector FeatureVector;
  typedef Thashing_policy<Domain, FeatureVector> HashingPolicy;
  typedef Trollout_policy<Domain> RolloutPolicy;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<std::unique_ptr<FeatureVector>(
      Domain &d, const State &s, const std::size_t *thread_id)>
      StateFeatureFunctor;
  typedef std::function<bool(const std::size_t &, const std::size_t &,
                             const double &, const double &)>
      WatchdogFunctor;

  RIWSolver(
      Domain &domain, const StateFeatureFunctor &state_features,
      std::size_t time_budget = 3600000, std::size_t rollout_budget = 100000,
      std::size_t max_depth = 1000, double exploration = 0.25,
      std::size_t epsilon_moving_average_window = 100, double epsilon = 0.001,
      double discount = 1.0, bool online_node_garbage = false,
      bool debug_logs = false,
      const WatchdogFunctor &watchdog = [](const std::size_t &,
                                           const std::size_t &, const double &,
                                           const double &) { return true; });

  // clears the solver (clears the search graph, thus preventing from reusing
  // previous search results)
  void clear();

  // solves from state s
  void solve(const State &s);

  bool is_solution_defined_for(const State &s) const;
  Action get_best_action(const State &s);
  double get_best_value(const State &s) const;
  std::size_t get_nb_of_explored_states() const;
  std::size_t get_nb_of_pruned_states() const;
  std::pair<std::size_t, std::size_t> get_exploration_statistics() const;
  std::size_t get_nb_rollouts() const;
  std::list<Action> action_prefix() const;
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map policy();

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
  atomic_size_t _epsilon_moving_average_window;
  atomic_double _epsilon;
  atomic_double _discount;
  bool _online_node_garbage;
  atomic_double _min_reward;
  atomic_size_t _nb_rollouts;
  RolloutPolicy _rollout_policy;
  ExecutionPolicy _execution_policy;
  atomic_bool _debug_logs;
  WatchdogFunctor _watchdog;

  std::unique_ptr<std::mt19937> _gen;
  typename ExecutionPolicy::Mutex _gen_mutex;
  typename ExecutionPolicy::Mutex _time_mutex;
  typename ExecutionPolicy::Mutex _epsilons_protect;

  atomic_double _epsilon_moving_average;
  std::list<double> _epsilons;

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
        const atomic_size_t &epsilon_moving_average_window,
        const atomic_double &epsilon, atomic_double &epsilon_moving_average,
        std::list<double> &epsilons, const atomic_double &discount,
        atomic_double &min_reward, const atomic_size_t &width, Graph &graph,
        RolloutPolicy &rollout_policy, ExecutionPolicy &execution_policy,
        std::mt19937 &gen, typename ExecutionPolicy::Mutex &gen_mutex,
        typename ExecutionPolicy::Mutex &time_mutex,
        typename ExecutionPolicy::Mutex &epsilons_protect,
        const atomic_bool &debug_logs, const WatchdogFunctor &watchdog);

    // solves from state s
    // return true iff no state has been pruned or time or rollout budgets are
    // consumed
    bool solve(const State &s,
               const std::chrono::time_point<std::chrono::high_resolution_clock>
                   &start_time,
               atomic_size_t &nb_rollouts, TupleVector &feature_tuples);

  private:
    RIWSolver &_parent_solver;
    Domain &_domain;
    const StateFeatureFunctor &_state_features;
    const atomic_size_t &_time_budget;
    const atomic_size_t &_rollout_budget;
    const atomic_size_t &_max_depth;
    const atomic_double &_exploration;
    const atomic_size_t &_epsilon_moving_average_window;
    const atomic_double &_epsilon;
    atomic_double &_epsilon_moving_average;
    std::list<double> &_epsilons;
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
    typename ExecutionPolicy::Mutex &_epsilons_protect;
    const atomic_bool &_debug_logs;
    const WatchdogFunctor &_watchdog;

    void
    rollout(Node &root_node, TupleVector &feature_tuples,
            atomic_size_t &nb_rollouts, atomic_bool &states_pruned,
            atomic_bool &reached_end_of_trajectory_once,
            const std::chrono::time_point<std::chrono::high_resolution_clock>
                &start_time,
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

    std::size_t update_epsilon_moving_average(const Node &node,
                                              const double &node_record_value);
    std::size_t elapsed_time(
        const std::chrono::time_point<std::chrono::high_resolution_clock>
            &start_time);
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
