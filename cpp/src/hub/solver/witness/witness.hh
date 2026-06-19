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

template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class WitnessSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Observation Observation;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

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

  typedef std::function<bool(const WitnessSolver &, Domain &)> CallbackFunctor;

  WitnessSolver(
      Domain &domain, double epsilon = 0.001, double discount = 0.95,
      std::size_t max_iterations = 100, double lp_infinity = 1e20,
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

  std::size_t get_state_index(const State &s);
  const std::unordered_map<std::size_t, State> &get_index_to_state() const;

private:
  struct AlphaVector {
    std::vector<double> values;
    std::size_t action_idx;
    std::vector<std::size_t> obs_choices;

    AlphaVector() : action_idx(0) {}
    AlphaVector(std::size_t num_states, std::size_t a_idx)
        : values(num_states, 0.0), action_idx(a_idx) {}
  };

  Domain &_domain;
  double _epsilon;
  double _discount;
  std::size_t _max_iterations;
  double _lp_infinity;
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
