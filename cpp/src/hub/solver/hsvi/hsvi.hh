/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_HSVI_HH
#define SKDECIDE_HSVI_HH

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

/**
 * @brief HSVI solver for discounted reward POMDPs.
 *
 * From Smith & Simmons, "Heuristic Search Value Iteration for POMDPs",
 * UAI 2004.
 *
 * HSVI maintains dual bounds on the value function:
 * - Lower bound: set of alpha-vectors, V(b) = max_alpha (alpha . b)
 * - Upper bound: sawtooth interpolation from MDP corner values
 *
 * It performs heuristic search in belief space, selecting actions via
 * the upper bound (optimistic) and observations via excess uncertainty.
 * Converges when the gap at the initial belief falls below epsilon.
 *
 * @tparam Tdomain Type of the domain class (must be PartiallyObservable)
 * @tparam Texecution_policy Type of the execution policy
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class HSVISolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Observation Observation;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::unordered_map<std::size_t, double> Belief;

  typedef std::function<bool(const HSVISolver &, Domain &)> CallbackFunctor;
  typedef std::function<bool(Domain &, const State &)> GoalCheckerFunctor;

  /**
   * @brief Construct a new HSVISolver.
   *
   * @param domain The domain instance to solve.
   * @param epsilon Convergence threshold for the gap
   *   V_upper(b0) - V_lower(b0). Defaults to 0.001.
   * @param discount Discount factor gamma. Must be strictly less than 1
   *   for reward-maximizing HSVI. Defaults to 0.95.
   * @param time_budget Maximum solving time in milliseconds.
   *   Defaults to 300000 (5 minutes).
   * @param max_sample_depth Maximum depth for heuristic exploration in
   *   belief space. Defaults to 100.
   * @param use_closed_list Whether to skip beliefs already explored at the
   *   same depth. Defaults to false.
   * @param depth_bound_eta Parameter eta for depth bound computation in
   *   the Goal-HSVI variant. Unused in reward HSVI. Defaults to 0.1.
   * @param max_vi_iterations Maximum iterations for bound initialization
   *   value iteration. Defaults to 1000.
   * @param vi_convergence_factor Convergence factor for initialization VI.
   *   The VI threshold is epsilon * vi_convergence_factor. Defaults to 0.01.
   * @param prob_epsilon Near-zero probability threshold below which
   *   transition probabilities are ignored. Defaults to 1e-15.
   * @param belief_hash_resolution Discretization factor for belief hashing.
   *   Probabilities are multiplied by this value and rounded to integers
   *   for hash computation. Defaults to 1000.0.
   * @param callback Functor called at each exploration iteration. Returns
   *   true to stop solving. Defaults to never stop.
   * @param verbose Whether to log verbose messages. Defaults to false.
   */
  HSVISolver(
      Domain &domain, double epsilon = 0.001, double discount = 0.95,
      std::size_t time_budget = 300000, std::size_t max_sample_depth = 100,
      bool use_closed_list = false, double depth_bound_eta = 0.1,
      std::size_t max_vi_iterations = 1000, double vi_convergence_factor = 0.01,
      double prob_epsilon = 1e-15, double belief_hash_resolution = 1000.0,
      const CallbackFunctor &callback = [](const HSVISolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  virtual ~HSVISolver() = default;

  virtual void clear();

  void solve(const std::vector<std::pair<State, double>> &initial_distribution);

  const Action &get_best_action(const Observation &obs);
  Value get_best_value(const Observation &obs);
  bool is_solution_defined_for(const Observation &obs);
  void reset_belief();

  const Action &get_best_action_from_belief(const Belief &b) const;
  Value get_best_value_from_belief(const Belief &b) const;
  bool is_solution_defined_for_from_belief(const Belief &b) const;

  std::size_t get_nb_alpha_vectors() const;
  std::size_t get_nb_bound_points() const;
  std::size_t get_solving_time() const;
  double get_gap() const;

  std::size_t get_state_index(const State &s);
  const std::unordered_map<std::size_t, State> &get_index_to_state() const;
  const std::unordered_map<std::size_t, std::size_t> &
  get_state_hash_to_idx() const;
  const std::vector<State> &get_states() const;

  /**
   * @brief Get the ordered list of (belief, action) pairs visited during
   * the last HSVI exploration.
   *
   * Returns the trajectory (path) explored during the most recent explore()
   * call. Each element is a pair of (belief, action) where the belief is
   * represented as a std::unordered_map<std::size_t, double> mapping state
   * indices to probabilities, and action is the greedy action (optimistic,
   * via upper bound) selected at that belief. The trajectory begins with the
   * root belief and ends at the deepest belief explored.
   *
   * Note: HSVI operates on continuous belief spaces via heuristic search.
   * Beliefs are returned as mappings from state indices to probabilities.
   *
   * Returns an empty list if solve() has not been called yet.
   */
  std::vector<std::pair<Belief, Action>> get_last_trajectory() const;

  struct AlphaVector {
    std::vector<double> values;
    Action action;
    std::size_t id;

    AlphaVector() : id(0) {}
    AlphaVector(std::size_t num_states, const Action &a, std::size_t vid)
        : values(num_states, 0.0), action(a), id(vid) {}
  };

  const std::vector<AlphaVector> &get_alpha_vectors() const;

protected:
  struct BoundPoint {
    Belief belief;
    double value;
  };

  virtual double _better(double a, double b) const { return std::max(a, b); }

  virtual double _worse(double a, double b) const { return std::min(a, b); }

  virtual double _best_init() const {
    return -std::numeric_limits<double>::infinity();
  }

  virtual double _worst_init() const {
    return std::numeric_limits<double>::infinity();
  }

  virtual bool _is_better(double a, double b) const { return a > b; }

  virtual double _get_value(const Value &v) const { return v.reward(); }

  virtual void make_value_obj(double v, Value &out) const { out.reward(v); }

  virtual double convergence_threshold(std::size_t depth) const {
    return _epsilon * std::pow(_discount, -static_cast<double>(depth));
  }

  virtual double get_terminal_state_value(std::size_t /*si*/) const {
    return 0.0;
  }

  virtual void compute_depth_bound() { _depth_bound = _max_sample_depth; }

  virtual void on_states_enumerated() {}
  virtual void on_model_cached() {}

  void enumerate_states(const Belief &b0);
  void pre_cache_model();
  virtual void initialize_alpha_bound();
  virtual void initialize_point_bound();

  void create_blind_policy_alphas();

  void explore(const Belief &b, std::size_t depth,
               std::unordered_set<std::size_t> &closed_list,
               std::vector<Belief> *belief_path = nullptr);

  void alpha_backup(const Belief &b);
  void point_update(const Belief &b);

  double evaluate_alpha(const Belief &b) const;
  double evaluate_sawtooth(const Belief &b) const;
  double evaluate_sawtooth_corner(const Belief &b) const;

  virtual double evaluate_upper(const Belief &b) const {
    return evaluate_sawtooth(b);
  }

  virtual double evaluate_lower(const Belief &b) const {
    return evaluate_alpha(b);
  }

  double dot_product(const AlphaVector &alpha, const Belief &b) const;
  std::size_t best_alpha_index(const Belief &b) const;

  Belief compute_posterior(const Belief &b, std::size_t action_idx,
                           std::size_t obs_hash) const;

  double compute_obs_probability(const Belief &b, std::size_t action_idx,
                                 std::size_t obs_hash) const;

  std::size_t belief_hash(const Belief &b) const;

  void update_current_belief(const Observation &obs);

  std::size_t elapsed_ms() const;

  Domain &_domain;
  double _epsilon;
  double _discount;
  std::size_t _time_budget;
  std::size_t _max_sample_depth;
  bool _use_closed_list;
  double _depth_bound_eta;
  std::size_t _max_vi_iterations;
  double _vi_convergence_factor;
  double _prob_epsilon;
  double _belief_hash_resolution;
  CallbackFunctor _callback;
  bool _verbose;

  ExecutionPolicy _execution_policy;

  std::vector<State> _states;
  std::unordered_map<std::size_t, std::size_t> _state_hash_to_idx;
  std::unordered_map<std::size_t, State> _index_to_state;
  std::vector<Action> _actions;
  std::unordered_map<std::size_t, std::size_t> _action_hash_to_idx;
  std::vector<std::vector<std::vector<std::pair<double, std::size_t>>>>
      _transitions;
  std::vector<std::vector<std::unordered_map<std::size_t, double>>> _obs_prob;
  std::vector<std::vector<double>> _values;
  std::unordered_map<std::size_t, Observation> _obs_objects;
  std::vector<std::vector<std::size_t>> _action_obs_hashes;
  std::vector<std::size_t> _state_idx_to_hash;
  std::vector<bool> _is_terminal_cache;

  std::vector<AlphaVector> _alpha_vectors;
  std::size_t _next_alpha_id = 0;

  std::vector<double> _mdp_values;
  std::vector<BoundPoint> _bound_points;

  Belief _initial_belief;
  Belief _current_belief;
  Action _last_action;
  bool _has_last_action = false;

  std::size_t _depth_bound = 0;
  double _gap = std::numeric_limits<double>::infinity();

  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  // Trajectory tracking (lazy computation on-demand)
  std::vector<Belief> _last_belief_path;
  typename ExecutionPolicy::Mutex _trajectory_mutex;
};

/**
 * @brief Goal-HSVI solver for undiscounted cost Goal-POMDPs.
 *
 * From Horak, Bosansky, Chatterjee, "Goal-HSVI: Heuristic Search
 * Value Iteration for Goal-POMDPs", IJCAI 2018.
 *
 * Goal-HSVI extends HSVI to handle undiscounted Goal-POMDPs with
 * cost minimization. Bounds are swapped relative to reward HSVI:
 * alpha-vectors form the upper bound and sawtooth interpolation
 * forms the lower bound. Uses a bounded search depth T based on
 * cost bounds and a closed list to avoid re-exploring beliefs.
 *
 * @tparam Tdomain Type of the domain class (must be PartiallyObservable)
 * @tparam Texecution_policy Type of the execution policy
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class GoalHSVISolver : public HSVISolver<Tdomain, Texecution_policy> {
public:
  using Base = HSVISolver<Tdomain, Texecution_policy>;
  using typename Base::Action;
  using typename Base::Belief;
  using typename Base::CallbackFunctor;
  using typename Base::Domain;
  using typename Base::ExecutionPolicy;
  using typename Base::GoalCheckerFunctor;
  using typename Base::Observation;
  using typename Base::State;
  using typename Base::Value;

  /**
   * @brief Construct a new GoalHSVISolver.
   *
   * @param domain The domain instance to solve.
   * @param goal_checker Functor (domain, state) -> bool returning true if
   *   the physical state is a goal state.
   * @param epsilon Convergence threshold for the gap
   *   V_upper(b0) - V_lower(b0). Defaults to 0.001.
   * @param discount Discount factor gamma. Typically 1.0 for undiscounted
   *   Goal-POMDPs. Defaults to 1.0.
   * @param time_budget Maximum solving time in milliseconds.
   *   Defaults to 300000 (5 minutes).
   * @param max_sample_depth Maximum exploration depth in belief space. Also
   *   used as fallback when the depth bound cannot be computed.
   *   Defaults to 100.
   * @param use_closed_list Whether to skip beliefs already explored at the
   *   same depth. Defaults to true.
   * @param depth_bound_eta Parameter eta for depth bound computation:
   *   T = ceil(C_max/c_min * (C_max - eta*eps) / ((1-eta)*eps)).
   *   Defaults to 0.1.
   * @param max_vi_iterations Maximum iterations for bound initialization
   *   value iteration. Defaults to 1000.
   * @param vi_convergence_factor Convergence factor for initialization VI.
   *   Defaults to 0.01.
   * @param prob_epsilon Near-zero probability threshold below which
   *   transition probabilities are ignored. Defaults to 1e-15.
   * @param belief_hash_resolution Discretization factor for belief hashing.
   *   Defaults to 1000.0.
   * @param callback Functor called at each exploration iteration. Returns
   *   true to stop solving. Defaults to never stop.
   * @param verbose Whether to log verbose messages. Defaults to false.
   * @param dead_end_cost Cost assigned to non-goal terminal states (dead
   *   ends). If nullopt, automatically computed from transition costs and
   *   depth/discount. Defaults to nullopt.
   */
  GoalHSVISolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      double epsilon = 0.001, double discount = 1.0,
      std::size_t time_budget = 300000, std::size_t max_sample_depth = 100,
      bool use_closed_list = true, double depth_bound_eta = 0.1,
      std::size_t max_vi_iterations = 1000, double vi_convergence_factor = 0.01,
      double prob_epsilon = 1e-15, double belief_hash_resolution = 1000.0,
      const CallbackFunctor &callback = [](const Base &,
                                           Domain &) { return false; },
      bool verbose = false, std::optional<double> dead_end_cost = std::nullopt);

  void clear() override;

protected:
  double _better(double a, double b) const override { return std::min(a, b); }
  double _worse(double a, double b) const override { return std::max(a, b); }
  double _best_init() const override {
    return std::numeric_limits<double>::infinity();
  }
  double _worst_init() const override {
    return -std::numeric_limits<double>::infinity();
  }
  bool _is_better(double a, double b) const override { return a < b; }
  double _get_value(const Value &v) const override { return v.cost(); }
  void make_value_obj(double v, Value &out) const override { out.cost(v); }

  double evaluate_upper(const Belief &b) const override {
    return this->evaluate_alpha(b);
  }
  double evaluate_lower(const Belief &b) const override {
    return this->evaluate_sawtooth(b);
  }

  double convergence_threshold(std::size_t /*depth*/) const override {
    return this->_depth_bound_eta * this->_epsilon;
  }

  double get_terminal_state_value(std::size_t si) const override;

  void initialize_alpha_bound() override;
  void initialize_point_bound() override;
  void compute_depth_bound() override;
  void on_states_enumerated() override;
  void on_model_cached() override;

private:
  GoalCheckerFunctor _goal_checker;
  std::optional<double> _user_dead_end_cost;
  std::vector<bool> _is_goal_cache;
  double _dead_end_cost = 0.0;
};

} // namespace skdecide

#endif // SKDECIDE_HSVI_HH
