/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_SARSOP_HH
#define SKDECIDE_SARSOP_HH

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

/**
 * @brief SARSOP solver for general POMDPs (reward maximization).
 *
 * From Kurniawati, Hsu & Lee, "SARSOP: Efficient Point-Based POMDP
 * Planning by Approximating Optimally Reachable Belief Spaces", RSS 2008.
 *
 * SARSOP maintains dual bounds:
 * - Lower bound: set of alpha-vectors, V(b) = max_alpha (alpha . b)
 * - Upper bound: sawtooth approximation from MDP corners + interior points
 *
 * It focuses exploration on beliefs reachable under the (unknown) optimal
 * policy via guided belief tree sampling.
 *
 * @tparam Tdomain Type of the domain class (must be PartiallyObservable)
 * @tparam Texecution_policy Type of the execution policy
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class SARSOPSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Observation Observation;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::unordered_map<std::size_t, double> Belief;

  typedef std::function<bool(const SARSOPSolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Construct a new SARSOPSolver.
   *
   * @param domain The domain instance to solve.
   * @param epsilon Convergence threshold for the gap
   *   V_upper(b0) - V_lower(b0). Defaults to 0.001.
   * @param discount Discount factor gamma. Must be in (0, 1).
   *   Defaults to 0.95.
   * @param time_budget Maximum solving time in milliseconds.
   *   Defaults to 300000 (5 minutes).
   * @param max_beliefs Maximum number of belief tree nodes to explore.
   *   Defaults to 100000.
   * @param pruning_delta Delta parameter for alpha-vector dominance
   *   pruning. Defaults to 1e-6.
   * @param max_vi_iterations Maximum iterations for bound initialization
   *   value iteration. Defaults to 1000.
   * @param vi_convergence_factor Convergence factor for initialization VI.
   *   The VI threshold is epsilon * vi_convergence_factor. Defaults to 0.01.
   * @param max_sample_depth Maximum depth for belief tree sampling.
   *   Defaults to 100.
   * @param prob_epsilon Near-zero probability threshold below which
   *   transition probabilities are ignored. Defaults to 1e-15.
   * @param ub_improvement_epsilon Minimum upper-bound improvement required
   *   to record an interior point. Defaults to 1e-10.
   * @param pruning_interval Number of iterations between alpha-vector
   *   pruning passes. Set to 0 to disable. Defaults to 10.
   * @param logging_interval Number of iterations between verbose log
   *   messages. Set to 0 to disable. Defaults to 50.
   * @param callback Functor called at the end of each iteration. Returns
   *   true to stop solving. Defaults to never stop.
   * @param verbose Whether to log verbose messages. Defaults to false.
   */
  SARSOPSolver(
      Domain &domain, double epsilon = 0.001, double discount = 0.95,
      std::size_t time_budget = 300000, std::size_t max_beliefs = 100000,
      double pruning_delta = 1e-6, std::size_t max_vi_iterations = 1000,
      double vi_convergence_factor = 0.01, std::size_t max_sample_depth = 100,
      double prob_epsilon = 1e-15, double ub_improvement_epsilon = 1e-10,
      std::size_t pruning_interval = 10, std::size_t logging_interval = 50,
      const CallbackFunctor &callback = [](const SARSOPSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();

  void solve(const std::vector<std::pair<State, double>> &initial_distribution);

  // Observation-based interface (tracks belief internally)
  const Action &get_best_action(const Observation &obs);
  Value get_best_value(const Observation &obs);
  bool is_solution_defined_for(const Observation &obs);
  void reset_belief();

  // Belief-based interface
  const Action &get_best_action_from_belief(const Belief &b);
  Value get_best_value_from_belief(const Belief &b);
  bool is_solution_defined_for_from_belief(const Belief &b);

  // Statistics
  std::size_t get_nb_alpha_vectors() const;
  std::size_t get_nb_explored_beliefs() const;
  std::size_t get_solving_time() const;
  double get_initial_lower_bound() const;
  double get_initial_upper_bound() const;
  double get_gap() const;

  // For pybind layer
  std::size_t get_state_index(const State &s);
  const std::unordered_map<std::size_t, State> &get_index_to_state() const;

private:
  struct AlphaVector {
    std::vector<double> values;
    Action action;
    std::size_t id;

    AlphaVector() : id(0) {}
    AlphaVector(std::size_t num_states, const Action &a, std::size_t vid)
        : values(num_states, 0.0), action(a), id(vid) {}
  };

  struct BeliefTreeNode {
    Belief belief;
    double lower_bound;
    double upper_bound;
    std::size_t best_alpha_idx;

    struct ActionEdge {
      Action action;
      double q_lower;
      double q_upper;
      double expected_reward;
      std::unordered_map<std::size_t, std::shared_ptr<BeliefTreeNode>> children;
      std::unordered_map<std::size_t, double> obs_probs;

      ActionEdge(const Action &a)
          : action(a), q_lower(0.0), q_upper(0.0), expected_reward(0.0) {}
    };

    std::vector<ActionEdge> action_edges;
    BeliefTreeNode *parent;
    std::size_t depth;
    bool pruned;
    bool expanded;

    BeliefTreeNode()
        : lower_bound(-std::numeric_limits<double>::infinity()),
          upper_bound(std::numeric_limits<double>::infinity()),
          best_alpha_idx(0), parent(nullptr), depth(0), pruned(false),
          expanded(false) {}
  };

  struct UpperBoundPoint {
    Belief belief;
    double value;
  };

  Domain &_domain;
  double _epsilon;
  double _discount;
  std::size_t _time_budget;
  std::size_t _max_beliefs;
  double _pruning_delta;
  std::size_t _max_vi_iterations;
  double _vi_convergence_factor;
  std::size_t _max_sample_depth;
  double _prob_epsilon;
  double _ub_improvement_epsilon;
  std::size_t _pruning_interval;
  std::size_t _logging_interval;
  CallbackFunctor _callback;
  bool _verbose;
  ExecutionPolicy _execution_policy;

  // State enumeration
  std::vector<State> _states;
  std::unordered_map<std::size_t, std::size_t> _state_hash_to_idx;
  std::unordered_map<std::size_t, State> _index_to_state;

  // Pre-cached model
  std::vector<std::vector<std::vector<std::pair<double, std::size_t>>>>
      _transitions;
  std::vector<std::vector<std::unordered_map<std::size_t, double>>> _obs_prob;
  std::unordered_map<std::size_t, Observation> _obs_objects;
  std::vector<std::vector<double>> _rewards;
  std::vector<Action> _actions;
  std::unordered_map<std::size_t, std::size_t> _action_hash_to_idx;
  // All unique observation hashes reachable per action
  std::vector<std::vector<std::size_t>> _action_obs_hashes;

  // Alpha-vector set (lower bound)
  std::vector<AlphaVector> _alpha_vectors;
  std::size_t _next_alpha_id;

  // Upper bound
  std::vector<double> _mdp_values;
  std::vector<UpperBoundPoint> _ub_points;

  // Belief tree
  std::unique_ptr<BeliefTreeNode> _root;
  std::size_t _nb_beliefs;

  // Belief tracking for observation-based interface
  Belief _current_belief;
  std::unique_ptr<Action> _last_action;
  bool _has_solution;

  // Timing
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  // State enumeration
  void enumerate_states(const Belief &b0);

  // Model pre-caching
  void pre_cache_model();

  // Bound initialization
  void initialize_lower_bound();
  void initialize_upper_bound();

  // Alpha-vector operations
  double dot_product(const AlphaVector &alpha, const Belief &b) const;
  double evaluate_lower(const Belief &b) const;
  std::size_t best_alpha_index(const Belief &b) const;

  // Upper bound
  double evaluate_upper(const Belief &b) const;
  double evaluate_upper_corner(const Belief &b) const;
  void update_upper_bound(BeliefTreeNode *node);

  // Belief operations
  Belief compute_posterior(const Belief &b, std::size_t action_idx,
                           std::size_t obs_hash) const;

  // Belief tree
  void initialize_belief_node(BeliefTreeNode *node);
  void expand_node(BeliefTreeNode *node);

  // SARSOP core
  std::vector<BeliefTreeNode *> sample();
  AlphaVector backup_belief(BeliefTreeNode *node);
  void backup(const std::vector<BeliefTreeNode *> &path);
  void prune();

  // Belief tracking
  void update_current_belief(const Observation &obs);

  std::size_t elapsed_ms() const;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/sarsop_impl.hh"
#endif

#endif // SKDECIDE_SARSOP_HH
