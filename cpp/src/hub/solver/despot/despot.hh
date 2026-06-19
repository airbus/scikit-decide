/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_DESPOT_HH
#define SKDECIDE_DESPOT_HH

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include <boost/range/irange.hpp>

#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

/**
 * @brief DESPOT solver for POMDPs (online, anytime, reward maximization).
 *
 * From Ye, Somani, Hsu & Lee, "DESPOT: Online POMDP Planning with
 * Regularization", JAIR 2017.
 *
 * DESPOT builds a sparse AND-OR tree using K determinized scenarios
 * sampled from the current belief. Regularization (RWDU) prevents
 * overfitting to the sampled scenarios.
 *
 * @tparam Tdomain Type of the domain class (must be PartiallyObservable)
 * @tparam Texecution_policy Type of the execution policy
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class DespotSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Observation Observation;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::unordered_map<std::size_t, double> Belief;

  typedef std::function<Value(Domain &, const State &, const std::size_t *)>
      DefaultPolicyFunctor;
  typedef std::function<Value(Domain &, const State &, const std::size_t *)>
      UpperBoundFunctor;
  typedef std::function<bool(const DespotSolver &, Domain &,
                             const std::size_t *)>
      CallbackFunctor;

  struct Scenario {
    State state;
    std::size_t id;
    double reward_sum;

    Scenario(const State &s, std::size_t i)
        : state(s), id(i), reward_sum(0.0) {}
  };

  struct VNode;

  struct QNode {
    Action action;
    std::unordered_map<std::size_t, std::unique_ptr<VNode>> children;
    VNode *parent;
    double upper_bound;
    double lower_bound;
    double step_reward;

    QNode(const Action &a, VNode *p)
        : action(a), parent(p), upper_bound(0.0), lower_bound(0.0),
          step_reward(0.0) {}
  };

  struct VNode {
    std::vector<Scenario> scenarios;
    double upper_bound;
    double lower_bound;
    double default_value;
    int depth;
    bool is_expanded;
    bool is_default;
    std::vector<std::unique_ptr<QNode>> children;
    QNode *parent;

    VNode()
        : upper_bound(std::numeric_limits<double>::infinity()),
          lower_bound(-std::numeric_limits<double>::infinity()),
          default_value(-std::numeric_limits<double>::infinity()), depth(0),
          is_expanded(false), is_default(false), parent(nullptr) {}
  };

  DespotSolver(
      Domain &domain, std::size_t num_scenarios = 500,
      std::size_t max_depth = 90, double regularization_constant = 0.0,
      double gap_reduction_rate = 0.95, double target_gap = 0.0,
      std::size_t time_budget = 1000, double discount = 0.95,
      std::size_t max_rollout_depth = 90,
      std::size_t num_particles_belief_update = 500,
      const DefaultPolicyFunctor &default_policy = nullptr,
      const UpperBoundFunctor &upper_bound_heuristic = nullptr,
      const CallbackFunctor &callback =
          [](const DespotSolver &, Domain &, const std::size_t *) {
            return false;
          },
      bool verbose = false);

  void clear();

  void solve(const std::vector<std::pair<State, double>> &initial_distribution);

  const Action &get_best_action(const Observation &obs);
  Value get_best_value(const Observation &obs);
  bool is_solution_defined_for(const Observation &obs);
  void reset_belief();

  const Action &get_best_action_from_belief(const Belief &b);
  Value get_best_value_from_belief(const Belief &b);
  bool is_solution_defined_for_from_belief(const Belief &b);

  std::size_t get_nb_tree_nodes() const;
  std::size_t get_solving_time() const;
  double get_gap() const;

  std::size_t get_state_index(const State &s);
  const std::unordered_map<std::size_t, State> &get_index_to_state() const;

private:
  void build_despot(VNode *root);
  VNode *explore(VNode *v);
  void backup(VNode *v);
  void prune(VNode *v);
  void make_default(VNode *v);
  double excess_uncertainty(VNode *v, double target_gap) const;

  void expand(VNode *v, const std::size_t *thread_id = nullptr);
  void init_bounds(VNode *v);
  void init_bounds_scenario(VNode *v, std::size_t scenario_idx,
                            std::vector<double> &ub_vals,
                            std::vector<double> &lb_vals,
                            const std::size_t *thread_id);
  double default_rollout(const State &s, int depth,
                         const std::size_t *thread_id);
  double upper_bound_state(const State &s, const std::size_t *thread_id);

  void plan_from_belief(const Belief &b);
  void update_belief_particles(const Observation &obs);
  Belief particles_to_belief() const;

  std::size_t elapsed_ms() const;

  Domain &_domain;
  std::size_t _num_scenarios;
  std::size_t _max_depth;
  double _regularization_constant;
  double _gap_reduction_rate;
  double _target_gap;
  std::size_t _time_budget;
  double _discount;
  std::size_t _max_rollout_depth;
  std::size_t _num_particles_belief;
  DefaultPolicyFunctor _default_policy;
  UpperBoundFunctor _upper_bound_heuristic;
  CallbackFunctor _callback;
  bool _verbose;

  ExecutionPolicy _execution_policy;
  typename ExecutionPolicy::Mutex _gen_mutex;
  typename ExecutionPolicy::Mutex _state_index_mutex;

  std::unordered_map<std::size_t, State> _index_to_state;

  std::mt19937 _rng;

  std::vector<std::pair<State, double>> _belief_particles;
  std::unique_ptr<Action> _last_action;
  bool _has_solution = false;

  std::unique_ptr<VNode> _current_tree;
  Action _best_action_cache;
  double _best_value_cache = 0.0;
  double _gap_cache = 0.0;
  std::size_t _nb_tree_nodes = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/despot_impl.hh"
#endif

#endif // SKDECIDE_DESPOT_HH
