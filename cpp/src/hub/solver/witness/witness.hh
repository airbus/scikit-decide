/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_WITNESS_HH
#define SKDECIDE_WITNESS_HH

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

/**
 * @brief Exact POMDP solver using the Witness algorithm.
 *
 * From Littman, "The Witness Algorithm: Solving Partially Observable
 * Markov Decision Processes", Brown University TR, 1994.
 *
 * Performs exact value iteration with piecewise-linear convex value
 * functions represented as sets of alpha-vectors. Uses LP-based
 * witness point finding to discover all non-dominated alpha-vectors
 * and Monahan LP pruning to remove dominated ones.
 *
 * Intended for verifying correctness of approximate POMDP solvers
 * on tiny test problems. Not suitable for large state/action spaces.
 *
 * @tparam Tdomain Type of the domain class (must be PartiallyObservable)
 * @tparam Texecution_policy Type of the execution policy
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class WitnessSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Observation Observation;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  enum class LPCallbackEvent {
    SolverIteration,
    LPProgress,
  };

  typedef std::unordered_map<std::size_t, double> Belief;

  template <typename T> struct VectorHash {
    std::size_t operator()(const std::vector<T> &v) const {
      std::size_t seed = v.size();
      for (const auto &x : v) {
        seed ^= std::hash<T>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };

  typedef std::function<Value(const State &)> TerminalValueFunctor;
  typedef std::function<bool(const WitnessSolver &, Domain &)> CallbackFunctor;

  /**
   * @brief Construct a new WitnessSolver.
   *
   * @param domain The domain instance to solve.
   * @param epsilon Convergence threshold for value function change at
   *   corner beliefs between iterations. Defaults to 0.001.
   * @param discount Discount factor gamma. Must be in (0, 1).
   *   Defaults to 0.95.
   * @param max_iterations Maximum number of value iteration steps.
   *   Defaults to 100.
   * @param lp_infinity Upper bound used for LP variable bounds with
   *   HiGHS. Defaults to 1e20.
   * @param lp_tolerance Numerical tolerance for LP feasibility checks
   *   and alpha-vector comparisons. Defaults to 1e-10.
   * @param terminal_value Functor taking a state and returning its terminal
   *   value (for non-goal terminal states). Defaults to reward=0.
   * @param callback Functor called at the end of each value iteration
   *   step. Returns true to stop solving. Defaults to never stop.
   * @param verbose Whether to log verbose messages. Defaults to false.
   */
  WitnessSolver(
      Domain &domain, double epsilon = 0.001, double discount = 0.95,
      std::size_t max_iterations = 100, double lp_infinity = 1e20,
      double lp_tolerance = 1e-10,
      const TerminalValueFunctor &terminal_value =
          [](const State &) { return Value(0.0, true); },
      const CallbackFunctor &callback = [](const WitnessSolver &,
                                           Domain &) { return false; },
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

  std::size_t get_nb_alpha_vectors() const;
  std::size_t get_nb_iterations() const;
  std::size_t get_solving_time() const;
  LPCallbackEvent get_callback_event() const {
    return LPCallbackEvent::SolverIteration;
  }

  std::size_t get_state_index(const State &s);
  const std::unordered_map<std::size_t, State> &get_index_to_state() const;
  const std::unordered_map<std::size_t, std::size_t> &
  get_state_hash_to_idx() const;
  const std::vector<State> &get_states() const;

  struct AlphaVector {
    std::vector<double> values;
    std::size_t action_idx;
    std::vector<std::size_t> obs_choices;

    AlphaVector() : action_idx(0) {}
    AlphaVector(std::size_t num_states, std::size_t a_idx)
        : values(num_states, 0.0), action_idx(a_idx) {}
  };

  const std::vector<AlphaVector> &get_alpha_vectors() const;
  const std::vector<Action> &get_action_list() const;

private:
  Domain &_domain;
  double _epsilon;
  double _discount;
  std::size_t _max_iterations;
  double _lp_infinity;
  double _lp_tolerance;
  TerminalValueFunctor _terminal_value;
  CallbackFunctor _callback;
  bool _verbose;
  ExecutionPolicy _execution_policy;

  std::vector<State> _states;
  std::unordered_map<std::size_t, std::size_t> _state_hash_to_idx;
  std::unordered_map<std::size_t, State> _index_to_state;

  std::vector<std::vector<std::vector<std::pair<double, std::size_t>>>>
      _transitions;
  std::vector<std::vector<std::unordered_map<std::size_t, double>>> _obs_prob;
  std::unordered_map<std::size_t, Observation> _obs_objects;
  std::vector<std::vector<double>> _rewards;
  std::vector<Action> _actions;
  std::unordered_map<std::size_t, std::size_t> _action_hash_to_idx;
  std::vector<std::vector<std::size_t>> _action_obs_hashes;
  std::vector<std::unordered_map<std::size_t, std::size_t>> _obs_hash_to_pos;

  std::vector<AlphaVector> _alpha_vectors;

  Belief _initial_belief;
  Belief _current_belief;
  std::unique_ptr<Action> _last_action;
  bool _has_solution;

  std::size_t _nb_iterations;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
  std::size_t _solving_time;

  void enumerate_states(const Belief &b0);
  void pre_cache_model();

  double dot_product(const AlphaVector &alpha, const Belief &b) const;
  double dot_product_dense(const std::vector<double> &alpha,
                           const std::vector<double> &b) const;
  double evaluate(const Belief &b) const;
  std::size_t best_alpha_index(const Belief &b) const;

  std::vector<double> compute_back(const std::vector<double> &alpha_values,
                                   std::size_t action_idx,
                                   std::size_t obs_hash) const;

  std::vector<std::vector<std::vector<double>>>
  precompute_back_vectors(const std::vector<AlphaVector> &v_prev,
                          std::size_t action_idx) const;

  AlphaVector besttree(
      const std::vector<double> &b_dense, std::size_t action_idx,
      const std::vector<AlphaVector> &v_prev,
      const std::vector<std::vector<std::vector<double>>> &back_vecs) const;

  std::vector<double>
  findb(std::size_t action_idx, const std::vector<AlphaVector> &v_prev,
        const std::vector<AlphaVector> &q_hat,
        const std::vector<std::vector<std::vector<double>>> &back_vecs,
        std::unordered_set<std::vector<std::size_t>, VectorHash<std::size_t>>
            &checked_candidates) const;

  std::vector<AlphaVector>
  witness_action(const std::vector<AlphaVector> &v_prev,
                 std::size_t action_idx) const;

  std::vector<AlphaVector> purge(const std::vector<AlphaVector> &v) const;

  bool check_convergence(const std::vector<AlphaVector> &v_prev,
                         const std::vector<AlphaVector> &v_next) const;

  Belief compute_posterior(const Belief &b, std::size_t action_idx,
                           std::size_t obs_hash) const;
  void update_current_belief(const Observation &obs);

  std::size_t elapsed_ms() const;
};

} // namespace skdecide

#endif // SKDECIDE_WITNESS_HH
