/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_RTDP_BEL_HH
#define SKDECIDE_RTDP_BEL_HH

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <chrono>
#include <random>
#include <cmath>

#include <boost/range/irange.hpp>

#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

/**
 * @brief RTDP-Bel solver for Goal POMDPs.
 *
 * From Bonet & Geffner, "Solving POMDPs: RTDP-Bel vs. Point-based
 * Algorithms", IJCAI 2009. This implements the RTDP-Bel algorithm from
 * Figure 2 of the paper.
 *
 * RTDP-Bel is a straightforward adaptation of RTDP to Goal POMDPs where
 * physical states are replaced by belief states. Beliefs are discretized
 * for hash table access using d(b(s)) = ceil(D * b(s)) where D is a
 * positive integer (the discretization parameter).
 *
 * The default interface works with observations (consistent with
 * scikit-decide's POMDP API). The solver internally maintains and updates
 * beliefs from the observation history. Additional _from_belief methods
 * provide direct belief-space access for advanced users.
 *
 * @tparam Tdomain Type of the domain class (must be PartiallyObservable)
 * @tparam Texecution_policy Type of the execution policy
 */
template <typename Tdomain, typename Texecution_policy = SequentialExecution>
class RTDPBelSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Observation Observation;
  typedef typename Domain::Value Value;
  typedef typename Domain::Predicate Predicate;
  typedef Texecution_policy ExecutionPolicy;

  // Internal belief type: sparse map from state hash to probability
  typedef std::unordered_map<std::size_t, double> Belief;

  typedef std::function<Predicate(Domain &, const State &, const std::size_t *)>
      GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &, const std::size_t *)>
      HeuristicFunctor;
  typedef std::function<bool(const RTDPBelSolver &, Domain &,
                             const std::size_t *)>
      CallbackFunctor;

  /**
   * @brief Construct a new RTDPBelSolver
   *
   * @param domain The domain instance
   * @param goal_checker Functor returning true if a physical state is a goal
   * @param heuristic Functor returning the heuristic cost for a physical state
   *   (belief heuristic is h(b) = Σ_s b(s)*h(s))
   * @param discretization Discretization parameter D for belief hashing
   * @param time_budget Maximum solving time in milliseconds
   * @param rollout_budget Maximum number of trials
   * @param max_depth Maximum depth of each trial
   * @param epsilon Maximum Bellman residual for convergence
   * @param discount Value function's discount factor
   * @param callback Functor called at the end of each trial
   * @param verbose Whether to log verbose messages
   */
  RTDPBelSolver(
      Domain &domain, const GoalCheckerFunctor &goal_checker,
      const HeuristicFunctor &heuristic, std::size_t discretization = 10,
      std::size_t time_budget = 3600000, std::size_t rollout_budget = 100000,
      std::size_t max_depth = 1000, double epsilon = 0.001,
      double discount = 1.0,
      const CallbackFunctor &callback =
          [](const RTDPBelSolver &, Domain &, const std::size_t *) {
            return false;
          },
      bool verbose = false);

  void clear();

  /**
   * @brief Run RTDP-Bel from an initial belief state, specified as a
   * distribution over physical states (list of state-probability pairs).
   */
  void solve(const std::vector<std::pair<State, double>> &initial_distribution);

  // ===== Default interface: observation-based =====
  // All methods work with observations. The solver internally maintains
  // and updates the current belief using Bayes rule.

  /**
   * @brief Get the best action given a new observation. Updates the
   * internal belief with (last_action, observation) via Bayes rule.
   * On the first call after solve(), uses the initial belief.
   */
  const Action &get_best_action(const Observation &obs);

  /**
   * @brief Get the best value for the current belief after observing obs.
   */
  Value get_best_value(const Observation &obs);

  /**
   * @brief Check if a solution is defined for the current belief.
   */
  bool is_solution_defined_for(const Observation &obs);

  /**
   * @brief Get (action, value) for the current tracked belief.
   */
  std::pair<Action, double> get_policy(const Observation &obs);

  /**
   * @brief Reset the tracked belief to the initial belief from solve().
   */
  void reset_belief();

  // ===== Belief-state interface =====
  // Direct belief-space access for advanced users.

  /**
   * @brief Get the best action for an explicit belief state.
   * @param b Belief as a sparse map from state hash to probability
   */
  const Action &get_best_action_from_belief(const Belief &b);

  /**
   * @brief Get the best value for an explicit belief state.
   */
  Value get_best_value_from_belief(const Belief &b);

  /**
   * @brief Check if a solution is defined for an explicit belief state.
   */
  bool is_solution_defined_for_from_belief(const Belief &b);

  // ===== Statistics =====

  std::size_t get_nb_explored_beliefs() const;
  std::size_t get_nb_rollouts() const;
  std::size_t get_solving_time() const;

private:
  typedef typename ExecutionPolicy::template atomic<double> atomic_double;
  typedef typename ExecutionPolicy::template atomic<bool> atomic_bool;

  // A discretized belief is used as hash table key
  // d(b(s)) = ceil(D * b(s))
  typedef std::unordered_map<std::size_t, std::size_t> DiscretizedBelief;

  struct DiscretizedBeliefHash {
    std::size_t operator()(const DiscretizedBelief &db) const;
  };

  struct DiscretizedBeliefEqual {
    bool operator()(const DiscretizedBelief &a,
                    const DiscretizedBelief &b) const;
  };

  struct ActionNode;

  struct BeliefNode {
    Belief belief;
    DiscretizedBelief discretized;
    std::list<std::unique_ptr<ActionNode>> actions;
    ActionNode *best_action;
    double best_value;
    bool goal;
    bool solved;
    typename ExecutionPolicy::Mutex mutex;

    BeliefNode(const Belief &b, const DiscretizedBelief &db);
  };

  struct ActionNode {
    Action action;
    std::vector<std::tuple<double, BeliefNode *>> outcomes;
    double value;

    ActionNode(const Action &a);
  };

  typedef std::unordered_map<DiscretizedBelief, std::unique_ptr<BeliefNode>,
                             DiscretizedBeliefHash, DiscretizedBeliefEqual>
      BeliefGraph;

  Domain &_domain;
  GoalCheckerFunctor _goal_checker;
  HeuristicFunctor _heuristic;
  std::size_t _discretization;
  std::size_t _time_budget;
  std::size_t _rollout_budget;
  std::size_t _max_depth;
  double _epsilon;
  double _discount;
  CallbackFunctor _callback;
  bool _verbose;

  ExecutionPolicy _execution_policy;
  typename ExecutionPolicy::Mutex _gen_mutex;
  typename ExecutionPolicy::Mutex _state_index_mutex;
  typename ExecutionPolicy::Mutex _graph_mutex;

  std::unique_ptr<std::mt19937> _gen;
  BeliefGraph _belief_graph;
  typename ExecutionPolicy::template atomic<std::size_t> _nb_rollouts;
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

  // Belief tracking for observation-based interaction
  BeliefNode *_initial_belief_node;
  BeliefNode *_current_belief_node;
  Action *_last_action;

  // State index mapping: State hash -> State object
  std::unordered_map<std::size_t, State> _index_to_state;
  std::size_t _next_state_index;

  DiscretizedBelief discretize(const Belief &b) const;
  double heuristic_value(const Belief &b, const std::size_t *thread_id) const;
  bool is_goal_belief(const Belief &b, const std::size_t *thread_id) const;

  BeliefNode *get_or_create_belief_node(const Belief &b,
                                        const std::size_t *thread_id);
  void expand(BeliefNode *bn, const std::size_t *thread_id);
  double q_value(ActionNode *a);
  ActionNode *greedy_action(BeliefNode *bn, const std::size_t *thread_id);
  void update(BeliefNode *bn, const std::size_t *thread_id);
  void trial(BeliefNode *bn, const std::size_t *thread_id);

  void update_current_belief(const Observation &obs);

  Belief compute_posterior_belief(const Belief &b, const Action &a,
                                  const Observation &o,
                                  const std::size_t *thread_id) const;
  State sample_state_from_belief(const Belief &b, const std::size_t *thread_id);

public:
  // Accessors for pybind layer
  std::size_t get_state_index(const State &s);
  const BeliefGraph &get_belief_graph() const;
  const std::unordered_map<std::size_t, State> &get_index_to_state() const;
  std::size_t get_discretization() const;

  /**
   * @brief Get the full explored belief-space policy as a map from
   * discretized beliefs to (action, value) pairs.
   */
  std::unordered_map<DiscretizedBelief, std::pair<Action, double>,
                     DiscretizedBeliefHash, DiscretizedBeliefEqual>
  get_belief_policy() const;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/rtdp_bel_impl.hh"
#endif

#endif // SKDECIDE_RTDP_BEL_HH
