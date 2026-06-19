/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_POMCP_HH
#define SKDECIDE_POMCP_HH

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

/**
 * @brief POMCP solver for POMDPs (online, reward maximization).
 *
 * From Silver & Veness, "Monte-Carlo Planning in Large POMDPs",
 * NIPS 2010.
 *
 * POMCP applies UCT to a history tree with particle-based belief
 * tracking. Planning happens online at each step: the solver samples
 * particles from the current belief, runs Monte Carlo simulations
 * through the history tree, and selects actions via UCB1.
 *
 * @tparam Tdomain Type of the domain class (must be PartiallyObservable)
 * @tparam Texecution_policy Type of the execution policy
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class POMCPSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Observation Observation;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef typename ExecutionPolicy::template atomic<std::size_t> atomic_size_t;
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;

  typedef std::unordered_map<std::size_t, double> Belief;

  typedef std::function<bool(const POMCPSolver &, Domain &)> CallbackFunctor;

  struct ActionNode;

  struct HistoryNode {
    atomic_size_t visits_count = 0;
    std::vector<State> particles;
    typename MapTypeDeducer<Action, std::unique_ptr<ActionNode>>::Map
        action_children;
    ActionNode *parent = nullptr;
    mutable typename ExecutionPolicy::Mutex mutex;
  };

  struct ActionNode {
    Action action;
    atomic_size_t visits_count = 0;
    atomic_double value = 0.0;
    std::unordered_map<std::size_t, std::unique_ptr<HistoryNode>>
        observation_children;
    HistoryNode *parent = nullptr;
    mutable typename ExecutionPolicy::Mutex mutex;

    ActionNode() {}
    ActionNode(const Action &a, HistoryNode *p) : action(a), parent(p) {}
  };

  /**
   * @brief Construct a new POMCPSolver.
   *
   * @param domain The domain instance to solve.
   * @param exploration_constant UCB1 exploration constant (c). Controls
   *   the exploration-exploitation trade-off. Defaults to 1/sqrt(2).
   * @param discount Discount factor gamma. Must be in (0, 1].
   *   Defaults to 0.95.
   * @param num_simulations Number of Monte Carlo simulations per planning
   *   step. Defaults to 1000.
   * @param max_depth Maximum search and rollout depth. Defaults to 100.
   * @param epsilon Discount-depth cutoff threshold. A simulation stops
   *   when gamma^depth < epsilon. Defaults to 0.001.
   * @param time_budget Maximum planning time per step in milliseconds.
   *   0 means no time limit (only num_simulations is used).
   *   Defaults to 0.
   * @param num_particles_belief_update Number of particles for belief
   *   update via particle filter. Defaults to 500.
   * @param ess_threshold_ratio Effective sample size threshold ratio for
   *   resampling. Resampling occurs when ESS < N / ratio. Defaults to 2.0.
   * @param callback Functor called at each simulation iteration. Returns
   *   true to stop planning. Defaults to never stop.
   * @param verbose Whether to log verbose messages. Defaults to false.
   */
  POMCPSolver(
      Domain &domain, double exploration_constant = 1.0 / std::sqrt(2.0),
      double discount = 0.95, std::size_t num_simulations = 1000,
      std::size_t max_depth = 100, double epsilon = 0.001,
      std::size_t time_budget = 0,
      std::size_t num_particles_belief_update = 500,
      double ess_threshold_ratio = 2.0,
      const CallbackFunctor &callback = [](const POMCPSolver &,
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

  std::size_t get_nb_tree_nodes() const;
  std::size_t get_solving_time() const;

  std::size_t get_state_index(const State &s);
  const std::unordered_map<std::size_t, State> &get_index_to_state() const;

private:
  void search(HistoryNode *root);
  double simulate(const State &s, HistoryNode *h, std::size_t depth,
                  const std::size_t *thread_id);
  double rollout(const State &s, std::size_t depth,
                 const std::size_t *thread_id);
  ActionNode *select_action_ucb1(HistoryNode *h);
  void expand(HistoryNode *h, const State &s, const std::size_t *thread_id);

  struct SimulationResult {
    State next_state;
    Observation observation;
    double reward;
    bool terminal;
  };
  SimulationResult simulate_transition(const State &s, const Action &a,
                                       const std::size_t *thread_id);

  void update_belief_particles(const Observation &obs);
  void plan_from_belief(const Belief &b);
  Belief particles_to_belief() const;

  std::size_t elapsed_ms() const;

  Domain &_domain;
  double _exploration_constant;
  double _discount;
  std::size_t _num_simulations;
  std::size_t _max_depth;
  double _epsilon;
  std::size_t _time_budget;
  std::size_t _num_particles_belief;
  double _ess_threshold_ratio;
  CallbackFunctor _callback;
  bool _verbose;

  ExecutionPolicy _execution_policy;
  typename ExecutionPolicy::Mutex _gen_mutex;
  typename ExecutionPolicy::Mutex _state_index_mutex;
  typename ExecutionPolicy::Mutex _time_mutex;

  std::unordered_map<std::size_t, State> _index_to_state;

  std::mt19937 _rng;

  std::vector<std::pair<State, double>> _belief_particles;
  std::unique_ptr<Action> _last_action;
  bool _has_solution = false;

  std::unique_ptr<HistoryNode> _current_tree;
  Action _best_action_cache;
  double _best_value_cache = 0.0;
  atomic_size_t _nb_tree_nodes = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/pomcp_impl.hh"
#endif

#endif // SKDECIDE_POMCP_HH
