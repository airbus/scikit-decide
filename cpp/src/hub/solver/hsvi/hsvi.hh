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

  typedef std::function<Value(const State &)> TerminalValueFunctor;
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
   * @param terminal_value Functor taking a state and returning its terminal
   *   value (for terminal states). Defaults to 0.0.
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
      const TerminalValueFunctor &terminal_value =
          [](const State &) { return Value(0.0, false); },
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

  virtual double _better(double a, double b) const;
  virtual double _worse(double a, double b) const;
  virtual double _best_init() const;
  virtual double _worst_init() const;
  virtual bool _is_better(double a, double b) const;
  virtual double _get_value(const Value &v) const;
  virtual void make_value_obj(double v, Value &out) const;
  virtual double convergence_threshold(std::size_t depth) const;
  virtual double get_terminal_state_value(std::size_t si) const;
  virtual void compute_depth_bound();
  virtual void on_states_enumerated();
  virtual void on_model_cached();

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
  virtual double evaluate_sawtooth_corner(const Belief &b) const;

  virtual double evaluate_upper(const Belief &b) const;
  virtual double evaluate_lower(const Belief &b) const;

  // Per-state clamping applied after the Bellman backup (safety net for
  // numerical drift). With the two-phase backup that uses a fallback alpha for
  // observations unreachable from the backup belief, bounds are naturally
  // admissible and this clamp is redundant — but it is kept as a guard.
  // GoalHSVI overrides to a no-op (backup is naturally >= LB there).
  virtual void apply_alpha_clamp(double &val, std::size_t si) const;

  // Return the index of the "worst" alpha to use as a fallback for
  // observations that are unreachable from the backup belief (empty
  // posterior). Using this alpha for those observations ensures the backup
  // remains globally admissible at states outside the belief's support.
  //
  // Default (reward-max): worst LB = alpha with minimum total value.
  // Ensures the new alpha doesn't exceed the UB at those states.
  virtual std::size_t fallback_alpha_index_for_empty_posterior() const;

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
  TerminalValueFunctor _terminal_value;
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
   * @param terminal_value Functor taking a state and returning its terminal
   *   value (for terminal states). Overrides goal/dead-end logic if provided.
   *   Defaults to nullptr (use goal_checker + dead_end_cost logic).
   * @param dead_end_cost Cost assigned to non-goal terminal states (dead
   *   ends). If nullopt, automatically computed from transition costs and
   *   depth/discount. Ignored if terminal_value is provided. Defaults to
   * nullopt.
   */
  GoalHSVISolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      double epsilon = 0.001, double discount = 1.0,
      std::size_t time_budget = 300000, std::size_t max_sample_depth = 100,
      bool use_closed_list = true, double depth_bound_eta = 0.1,
      std::size_t max_vi_iterations = 1000, double vi_convergence_factor = 0.01,
      double prob_epsilon = 1e-15, double belief_hash_resolution = 1000.0,
      const typename Base::TerminalValueFunctor &terminal_value = nullptr,
      const CallbackFunctor &callback = [](const Base &,
                                           Domain &) { return false; },
      bool verbose = false, std::optional<double> dead_end_cost = std::nullopt);

  void clear() override;

protected:
  double _better(double a, double b) const override;
  double _worse(double a, double b) const override;
  double _best_init() const override;
  double _worst_init() const override;
  bool _is_better(double a, double b) const override;
  double _get_value(const Value &v) const override;
  void make_value_obj(double v, Value &out) const override;
  double evaluate_upper(const Belief &b) const override;
  double evaluate_lower(const Belief &b) const override;

  // No per-state clamping for cost minimization. In GoalHSVI the alpha
  // vectors are the UPPER bound; clamping them up to _mdp_values (the LB)
  // collapses UB = LB → gap = 0 after one iteration. The two-phase backup
  // (fallback alpha for empty-posterior observations) maintains admissibility
  // naturally, so no clamping is needed.
  void apply_alpha_clamp(double &val, std::size_t si) const override;

  // Worst UB alpha = the one with maximum total value. Using it as the
  // fallback for observations unreachable from the backup belief ensures
  // g_a[si] >= MDP[si] for all states, maintaining UB admissibility.
  std::size_t fallback_alpha_index_for_empty_posterior() const override;

  double evaluate_sawtooth_corner(const Belief &b) const override;
  double convergence_threshold(std::size_t depth) const override;
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
  bool _use_custom_terminal_value = false;
};

} // namespace skdecide

#endif // SKDECIDE_HSVI_HH
