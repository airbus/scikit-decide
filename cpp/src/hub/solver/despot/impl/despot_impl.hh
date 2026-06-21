/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Implementation of DESPOT from:
 * Ye, Somani, Hsu & Lee, "DESPOT: Online POMDP Planning with
 * Regularization", JAIR 2017.
 */
#ifndef SKDECIDE_DESPOT_IMPL_HH
#define SKDECIDE_DESPOT_IMPL_HH

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>

#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

#define SK_DESPOT_TEMPLATE_DECL                                                \
  template <typename Tdomain, typename Texecution_policy>

#define SK_DESPOT_CLASS DespotSolver<Tdomain, Texecution_policy>

// --- Constructor ---

SK_DESPOT_TEMPLATE_DECL
SK_DESPOT_CLASS::DespotSolver(
    Domain &domain, std::size_t num_scenarios, std::size_t max_depth,
    double regularization_constant, double gap_reduction_rate,
    double target_gap, std::size_t time_budget, double discount,
    std::size_t max_rollout_depth, std::size_t num_particles_belief_update,
    double ess_threshold_ratio, const DefaultPolicyFunctor &default_policy,
    const UpperBoundFunctor &upper_bound_heuristic,
    const TerminalValueFunctor &terminal_value, const CallbackFunctor &callback,
    bool verbose)
    : _domain(domain), _num_scenarios(num_scenarios), _max_depth(max_depth),
      _regularization_constant(regularization_constant),
      _gap_reduction_rate(gap_reduction_rate), _target_gap(target_gap),
      _time_budget(time_budget), _discount(discount),
      _max_rollout_depth(max_rollout_depth),
      _num_particles_belief(num_particles_belief_update),
      _ess_threshold_ratio(ess_threshold_ratio),
      _default_policy(default_policy),
      _upper_bound_heuristic(upper_bound_heuristic),
      _terminal_value(terminal_value), _callback(callback), _verbose(verbose),
      _rng(std::random_device{}()) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm DESPOT");
  }
}

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::clear() {
  _index_to_state.clear();
  _belief_particles.clear();
  _last_action.reset();
  _has_solution = false;
  _current_tree.reset();
  _nb_tree_nodes = 0;
  _gap_cache = 0.0;
  _best_value_cache = 0.0;
}

SK_DESPOT_TEMPLATE_DECL
std::size_t SK_DESPOT_CLASS::get_state_index(const State &s) {
  std::size_t h = typename State::Hash()(s);
  _execution_policy.protect(
      [this, &h, &s]() {
        if (_index_to_state.find(h) == _index_to_state.end()) {
          _index_to_state[h] = s;
        }
      },
      _state_index_mutex);
  return h;
}

SK_DESPOT_TEMPLATE_DECL
const std::unordered_map<std::size_t, typename SK_DESPOT_CLASS::State> &
SK_DESPOT_CLASS::get_index_to_state() const {
  return _index_to_state;
}

SK_DESPOT_TEMPLATE_DECL
std::size_t SK_DESPOT_CLASS::elapsed_ms() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now() - _start_time)
      .count();
}

// --- solve(): Initialize belief from distribution ---

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::solve(
    const std::vector<std::pair<State, double>> &initial_distribution) {
  clear();

  for (const auto &p : initial_distribution) {
    get_state_index(p.first);
  }

  _belief_particles = initial_distribution;
  _last_action.reset();
  _has_solution = true;

  if (_verbose)
    Logger::debug("DESPOT: initialized with " +
                  std::to_string(_belief_particles.size()) +
                  " belief particles");
}

// --- Online planning from current belief ---

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::plan_from_belief(const Belief &b) {
  Logger::info("Running DESPOT solver");
  _start_time = std::chrono::high_resolution_clock::now();

  // Build probability vector for sampling scenarios
  std::vector<State> states;
  std::vector<double> probs;
  for (const auto &bp : b) {
    auto it = _index_to_state.find(bp.first);
    if (it != _index_to_state.end()) {
      states.push_back(it->second);
      probs.push_back(bp.second);
    }
  }

  if (states.empty()) {
    throw std::runtime_error("DESPOT: empty belief, cannot plan");
  }

  // Create root VNode with K scenarios sampled from belief
  auto root = std::make_unique<VNode>();
  root->depth = 0;

  std::discrete_distribution<std::size_t> belief_dist(probs.begin(),
                                                      probs.end());
  for (std::size_t k = 0; k < _num_scenarios; ++k) {
    std::size_t idx = belief_dist(_rng);
    root->scenarios.emplace_back(states[idx], k);
  }

  _nb_tree_nodes = 1;
  init_bounds(root.get());

  // Run anytime DESPOT construction
  build_despot(root.get());

  // Extract best action: action with highest lower bound
  double best_lb = -std::numeric_limits<double>::infinity();
  QNode *best_q = nullptr;
  for (auto &q : root->children) {
    if (q->lower_bound > best_lb) {
      best_lb = q->lower_bound;
      best_q = q.get();
    }
  }

  if (best_q) {
    _best_action_cache = best_q->action;
    _best_value_cache = best_lb;
  } else if (!root->scenarios.empty()) {
    auto elements =
        _domain.get_applicable_actions(root->scenarios[0].state, nullptr)
            .get_elements();
    auto it = elements.begin();
    if (it != elements.end()) {
      _best_action_cache = *it;
      _best_value_cache = root->lower_bound;
    }
  }

  _gap_cache = root->upper_bound - root->lower_bound;
  _current_tree = std::move(root);

  Logger::info("DESPOT finished in " +
               StringConverter::from((double)elapsed_ms() / 1e3) +
               " seconds with " + StringConverter::from(_nb_tree_nodes) +
               " tree nodes.");
}

// --- BuildDESPOT: Algorithm 1 ---

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::build_despot(VNode *root) {
  double initial_gap = root->upper_bound - root->lower_bound;
  double current_target = initial_gap;
  std::size_t iteration = 0;

  while (elapsed_ms() < _time_budget) {
    // Explore: find leaf to expand (skip trajectory building for performance)
    VNode *leaf = explore(root, nullptr);

    // Save the leaf for on-demand trajectory reconstruction
    _execution_policy.protect([this, leaf]() { _last_explored_leaf = leaf; },
                              _trajectory_mutex);

    if (!leaf)
      break;

    // Backup: propagate bounds from leaf to root
    VNode *v = leaf;
    while (v) {
      backup(v);
      if (v->parent)
        v = v->parent->parent;
      else
        break;
    }

    // Prune: RWDU-based regularization
    if (_regularization_constant > 0) {
      prune(root);
    }

    ++iteration;

    double gap = root->upper_bound - root->lower_bound;
    if (gap <= _target_gap)
      break;

    current_target *= _gap_reduction_rate;

    if (_callback(*this, _domain, nullptr))
      break;
  }

  if (_verbose)
    Logger::debug("DESPOT: planned in " + std::to_string(elapsed_ms()) +
                  "ms, " + std::to_string(iteration) + " iterations, " +
                  std::to_string(_nb_tree_nodes) + " tree nodes, gap = " +
                  std::to_string(root->upper_bound - root->lower_bound));
}

// --- Explore: Algorithm 2 — Forward heuristic search ---

SK_DESPOT_TEMPLATE_DECL
typename SK_DESPOT_CLASS::VNode *
SK_DESPOT_CLASS::explore(VNode *v,
                         std::vector<std::pair<State, Action>> *trajectory) {
  if (v->scenarios.empty())
    return nullptr;

  if (static_cast<std::size_t>(v->depth) >= _max_depth)
    return nullptr;

  // Check if all scenarios are terminal
  bool all_terminal = true;
  for (const auto &sc : v->scenarios) {
    if (!_domain.is_terminal(sc.state, nullptr)) {
      all_terminal = false;
      break;
    }
  }
  if (all_terminal)
    return nullptr;

  double gap = v->upper_bound - v->lower_bound;
  if (gap <= _target_gap)
    return nullptr;

  // If not expanded yet, expand and return this node
  if (!v->is_expanded) {
    expand(v, nullptr);
    return v;
  }

  // Find best action (highest Q upper bound)
  QNode *best_q = nullptr;
  double best_q_ub = -std::numeric_limits<double>::infinity();
  for (auto &q : v->children) {
    if (q->upper_bound > best_q_ub) {
      best_q_ub = q->upper_bound;
      best_q = q.get();
    }
  }
  if (!best_q)
    return nullptr;

  // Record trajectory step: representative state and selected action
  if (trajectory && !v->scenarios.empty()) {
    trajectory->push_back(
        std::make_pair(v->scenarios[0].state, best_q->action));
  }

  // Find observation child with highest weighted excess uncertainty
  VNode *best_child = nullptr;
  double best_eu = -std::numeric_limits<double>::infinity();
  double parent_weight =
      static_cast<double>(v->scenarios.size()) / _num_scenarios;

  for (auto &child_pair : best_q->children) {
    VNode *child = child_pair.second.get();
    double child_weight =
        static_cast<double>(child->scenarios.size()) / _num_scenarios;
    double child_gap = child->upper_bound - child->lower_bound;
    double eu = child_weight * child_gap;
    if (eu > best_eu) {
      best_eu = eu;
      best_child = child;
    }
  }

  if (best_child)
    return explore(best_child, trajectory);

  return nullptr;
}

// --- expand(): Generate action and observation children ---

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::expand(VNode *v, const std::size_t *thread_id) {
  if (v->is_expanded)
    return;

  v->is_expanded = true;

  if (v->scenarios.empty())
    return;

  // Get applicable actions from a representative state
  auto actions =
      _domain.get_applicable_actions(v->scenarios[0].state, thread_id)
          .get_elements();

  for (auto action : actions) {
    auto qnode = std::make_unique<QNode>(action, v);
    double total_reward = 0.0;

    // For each scenario, simulate one step
    for (const auto &sc : v->scenarios) {
      if (_domain.is_terminal(sc.state, thread_id))
        continue;

      // Sample next state from transition distribution
      auto next_dist =
          _domain.get_next_state_distribution(sc.state, action, thread_id)
              .get_values();

      std::vector<double> t_probs;
      std::vector<State> t_states;
      for (auto ns_item : next_dist) {
        t_states.push_back(ns_item.state());
        t_probs.push_back(ns_item.probability());
      }

      if (t_states.empty())
        continue;

      std::discrete_distribution<std::size_t> t_dist(t_probs.begin(),
                                                     t_probs.end());
      std::size_t ns_idx = t_dist(_rng);
      State next_state = t_states[ns_idx];
      get_state_index(next_state);

      // Get reward
      double reward =
          _domain.get_transition_value(sc.state, action, next_state, thread_id)
              .reward();
      total_reward += reward;

      // Sample observation from observation distribution
      auto obs_dist =
          _domain.get_observation_distribution(next_state, action, thread_id)
              .get_values();

      std::vector<double> o_probs;
      std::vector<Observation> o_obs;
      for (auto o_item : obs_dist) {
        o_obs.push_back(o_item.observation());
        o_probs.push_back(o_item.probability());
      }

      if (o_obs.empty())
        continue;

      std::discrete_distribution<std::size_t> o_dist(o_probs.begin(),
                                                     o_probs.end());
      std::size_t o_idx = o_dist(_rng);
      Observation obs = o_obs[o_idx];
      std::size_t obs_hash = typename Observation::Hash()(obs);

      // Add scenario to appropriate observation child
      auto &child = qnode->children[obs_hash];
      if (!child) {
        child = std::make_unique<VNode>();
        child->depth = v->depth + 1;
        child->parent = qnode.get();
        ++_nb_tree_nodes;
      }
      child->scenarios.emplace_back(next_state, sc.id);
    }

    // Average step reward
    std::size_t non_terminal = 0;
    for (const auto &sc : v->scenarios) {
      if (!_domain.is_terminal(sc.state, thread_id))
        ++non_terminal;
    }
    qnode->step_reward = non_terminal > 0 ? total_reward / non_terminal : 0.0;

    // Initialize bounds for each observation child
    for (auto &child_pair : qnode->children) {
      init_bounds(child_pair.second.get());
    }

    // Compute Q-node bounds from children
    double q_ub = qnode->step_reward;
    double q_lb = qnode->step_reward;
    double total_scenarios = static_cast<double>(v->scenarios.size());

    for (auto &child_pair : qnode->children) {
      VNode *child = child_pair.second.get();
      double child_frac =
          static_cast<double>(child->scenarios.size()) / total_scenarios;
      q_ub += _discount * child_frac * child->upper_bound;
      q_lb += _discount * child_frac * child->lower_bound;
    }
    qnode->upper_bound = q_ub;
    qnode->lower_bound = q_lb;

    v->children.push_back(std::move(qnode));
  }

  // Update V-node bounds from Q-node bounds
  v->upper_bound = -std::numeric_limits<double>::infinity();
  v->lower_bound = -std::numeric_limits<double>::infinity();
  for (auto &q : v->children) {
    v->upper_bound = std::max(v->upper_bound, q->upper_bound);
    v->lower_bound = std::max(v->lower_bound, q->lower_bound);
  }
}

// --- init_bounds(): Initialize upper and lower bounds for a leaf ---

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::init_bounds(VNode *v) {
  if (v->scenarios.empty()) {
    v->upper_bound = 0.0;
    v->lower_bound = 0.0;
    v->default_value = 0.0;
    return;
  }

  std::size_t n = v->scenarios.size();
  std::vector<double> ub_vals(n, 0.0);
  std::vector<double> lb_vals(n, 0.0);

  // Fan out across parallel domain workers
  std::size_t capacity = _domain.get_parallel_capacity();
  boost::integer_range<std::size_t> worker_range(0, std::min(n, capacity));

  std::for_each(ExecutionPolicy::policy, worker_range.begin(),
                worker_range.end(),
                [this, &v, &ub_vals, &lb_vals, n,
                 capacity](const std::size_t &thread_id) {
                  // Each thread processes scenarios in a strided pattern
                  for (std::size_t i = thread_id; i < n; i += capacity) {
                    init_bounds_scenario(v, i, ub_vals, lb_vals, &thread_id);
                  }
                });

  double ub_sum = 0.0;
  double lb_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    ub_sum += ub_vals[i];
    lb_sum += lb_vals[i];
  }

  v->upper_bound = ub_sum / n;
  v->lower_bound = lb_sum / n;
  v->default_value = v->lower_bound;
}

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::init_bounds_scenario(VNode *v, std::size_t scenario_idx,
                                           std::vector<double> &ub_vals,
                                           std::vector<double> &lb_vals,
                                           const std::size_t *thread_id) {
  const auto &sc = v->scenarios[scenario_idx];
  if (!_domain.is_terminal(sc.state, thread_id)) {
    ub_vals[scenario_idx] = upper_bound_state(sc.state, thread_id);
    lb_vals[scenario_idx] = default_rollout(sc.state, v->depth, thread_id);
  }
}

// --- default_rollout(): Simulate random policy to estimate lower bound ---

SK_DESPOT_TEMPLATE_DECL
double SK_DESPOT_CLASS::default_rollout(const State &s, int start_depth,
                                        const std::size_t *thread_id) {
  if (_default_policy) {
    return _default_policy(_domain, s, thread_id).reward();
  }

  // Create a thread-local RNG seeded from the shared one
  std::size_t seed;
  _execution_policy.protect([this, &seed]() { seed = _rng(); }, _gen_mutex);
  std::mt19937 local_rng(seed);

  // Random rollout
  State current = s;
  double total_reward = 0.0;
  double gamma_power = 1.0;

  for (std::size_t d = start_depth; d < _max_rollout_depth; ++d) {
    if (_domain.is_terminal(current, thread_id)) {
      // Add terminal value for terminal states
      total_reward += gamma_power * _terminal_value(current).reward();
      break;
    }

    auto actions =
        _domain.get_applicable_actions(current, thread_id).get_elements();

    // Random action selection
    std::vector<Action> action_vec;
    for (auto a : actions) {
      action_vec.push_back(a);
    }
    if (action_vec.empty()) {
      // State has no actions but is not marked terminal: treat as terminal
      total_reward += gamma_power * _terminal_value(current).reward();
      break;
    }

    std::uniform_int_distribution<std::size_t> action_dist(
        0, action_vec.size() - 1);
    Action action = action_vec[action_dist(local_rng)];

    // Sample next state
    auto next_dist =
        _domain.get_next_state_distribution(current, action, thread_id)
            .get_values();

    std::vector<double> probs;
    std::vector<State> states;
    for (auto ns_item : next_dist) {
      states.push_back(ns_item.state());
      probs.push_back(ns_item.probability());
    }

    if (states.empty()) {
      // Action has no transitions but state is not marked terminal: treat as
      // terminal
      total_reward += gamma_power * _terminal_value(current).reward();
      break;
    }

    std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
    std::size_t idx = dist(local_rng);
    State next_state = states[idx];

    double reward =
        _domain.get_transition_value(current, action, next_state, thread_id)
            .reward();
    total_reward += gamma_power * reward;
    gamma_power *= _discount;

    current = next_state;
  }

  return total_reward;
}

// --- upper_bound_state(): Per-state upper bound ---

SK_DESPOT_TEMPLATE_DECL
double SK_DESPOT_CLASS::upper_bound_state(const State &s,
                                          const std::size_t *thread_id) {
  if (_upper_bound_heuristic) {
    return _upper_bound_heuristic(_domain, s, thread_id).reward();
  }

  // Check if terminal state
  if (_domain.is_terminal(s, thread_id)) {
    return _terminal_value(s).reward();
  }

  // Uninformed upper bound: R_max / (1 - gamma)
  // Estimate R_max from a few sampled transitions
  auto actions = _domain.get_applicable_actions(s, thread_id).get_elements();
  double max_reward = 0.0;

  for (auto action : actions) {
    auto next_dist =
        _domain.get_next_state_distribution(s, action, thread_id).get_values();
    if (next_dist.begin() == next_dist.end()) {
      // Action has no transitions: use terminal value
      max_reward = std::max(max_reward, std::abs(_terminal_value(s).reward()));
      continue;
    }
    for (auto ns_item : next_dist) {
      double r =
          _domain.get_transition_value(s, action, ns_item.state(), thread_id)
              .reward();
      max_reward = std::max(max_reward, std::abs(r));
    }
  }

  if (_discount < 1.0) {
    return max_reward / (1.0 - _discount);
  }
  return max_reward * _max_depth;
}

// --- backup(): Algorithm 3 — Bellman backup ---

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::backup(VNode *v) {
  if (!v->is_expanded)
    return;

  double best_ub = -std::numeric_limits<double>::infinity();
  double best_lb = -std::numeric_limits<double>::infinity();

  double total_scenarios = static_cast<double>(v->scenarios.size());

  for (auto &q : v->children) {
    // Recompute Q-node bounds from children
    double q_ub = q->step_reward;
    double q_lb = q->step_reward;

    for (auto &child_pair : q->children) {
      VNode *child = child_pair.second.get();
      double child_frac =
          static_cast<double>(child->scenarios.size()) / total_scenarios;
      q_ub += _discount * child_frac * child->upper_bound;
      q_lb += _discount * child_frac * child->lower_bound;
    }

    q->upper_bound = q_ub;
    q->lower_bound = q_lb;

    best_ub = std::max(best_ub, q_ub);
    best_lb = std::max(best_lb, q_lb);
  }

  v->upper_bound = best_ub;
  // Lower bound: max of children's lower bound and default policy value
  v->lower_bound = std::max(best_lb, v->default_value);
}

// --- prune(): Algorithm 4 — RWDU-based regularization ---

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::prune(VNode *v) {
  if (!v->is_expanded)
    return;

  // Recursively prune children first
  for (auto &q : v->children) {
    for (auto &child_pair : q->children) {
      prune(child_pair.second.get());
    }
  }

  // Compute RWDU for current best policy vs default policy
  double weight = static_cast<double>(v->scenarios.size()) / _num_scenarios;
  double gamma_d = std::pow(_discount, v->depth);
  double default_rwdu =
      weight * gamma_d * v->default_value - _regularization_constant;

  // RWDU for the expanded tree: weight * gamma^d * V_lower - lambda * |pi|
  int policy_nodes = 0;
  for (auto &q : v->children) {
    for (auto &child_pair : q->children) {
      policy_nodes +=
          child_pair.second->is_default
              ? 1
              : static_cast<int>(child_pair.second->scenarios.size());
    }
  }

  double tree_rwdu = weight * gamma_d * v->lower_bound -
                     _regularization_constant * policy_nodes;

  if (default_rwdu >= tree_rwdu) {
    make_default(v);
  }
}

// --- make_default(): Algorithm 5 — Collapse to default policy ---

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::make_default(VNode *v) {
  v->children.clear();
  v->is_expanded = false;
  v->is_default = true;
  v->lower_bound = v->default_value;
  v->upper_bound = v->default_value;
}

// --- excess_uncertainty() ---

SK_DESPOT_TEMPLATE_DECL
double SK_DESPOT_CLASS::excess_uncertainty(VNode *v, double target_gap) const {
  double weight = static_cast<double>(v->scenarios.size()) / _num_scenarios;
  double gamma_d = std::pow(_discount, v->depth);
  return weight * gamma_d * (v->upper_bound - v->lower_bound) - target_gap;
}

// --- Belief tracking for observation-based interface ---

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::update_belief_particles(const Observation &obs) {
  if (!_last_action || _belief_particles.empty())
    return;

  const Action &action = *_last_action;
  std::size_t obs_hash = typename Observation::Hash()(obs);

  // Particle filter update
  std::vector<std::pair<State, double>> new_particles;
  double total_weight = 0.0;

  for (const auto &particle : _belief_particles) {
    if (_domain.is_terminal(particle.first, nullptr))
      continue;

    // Sample next state from transition distribution
    auto next_dist =
        _domain.get_next_state_distribution(particle.first, action, nullptr)
            .get_values();

    std::vector<double> t_probs;
    std::vector<State> t_states;
    for (auto ns_item : next_dist) {
      t_states.push_back(ns_item.state());
      t_probs.push_back(ns_item.probability());
    }

    if (t_states.empty())
      continue;

    std::discrete_distribution<std::size_t> t_dist(t_probs.begin(),
                                                   t_probs.end());
    std::size_t ns_idx = t_dist(_rng);
    State next_state = t_states[ns_idx];
    get_state_index(next_state);

    // Compute observation likelihood P(obs | next_state, action)
    auto obs_dist =
        _domain.get_observation_distribution(next_state, action, nullptr)
            .get_values();

    double obs_likelihood = 0.0;
    for (auto o_item : obs_dist) {
      std::size_t oh = typename Observation::Hash()(o_item.observation());
      if (oh == obs_hash) {
        obs_likelihood = o_item.probability();
        break;
      }
    }

    if (obs_likelihood > 0.0) {
      double w = particle.second * obs_likelihood;
      new_particles.emplace_back(next_state, w);
      total_weight += w;
    }
  }

  // Normalize weights
  if (total_weight > 0.0) {
    for (auto &p : new_particles) {
      p.second /= total_weight;
    }
    _belief_particles = std::move(new_particles);
  }

  // Resample if effective sample size is low
  if (_belief_particles.size() > 1) {
    double ess = 0.0;
    for (const auto &p : _belief_particles) {
      ess += p.second * p.second;
    }
    ess = 1.0 / ess;

    if (ess < _belief_particles.size() / _ess_threshold_ratio) {
      std::vector<double> weights;
      for (const auto &p : _belief_particles) {
        weights.push_back(p.second);
      }
      std::discrete_distribution<std::size_t> resample_dist(weights.begin(),
                                                            weights.end());
      std::vector<std::pair<State, double>> resampled;
      double uniform_weight = 1.0 / _num_particles_belief;
      for (std::size_t i = 0; i < _num_particles_belief; ++i) {
        std::size_t idx = resample_dist(_rng);
        resampled.emplace_back(_belief_particles[idx].first, uniform_weight);
      }
      _belief_particles = std::move(resampled);
    }
  }
}

SK_DESPOT_TEMPLATE_DECL
typename SK_DESPOT_CLASS::Belief SK_DESPOT_CLASS::particles_to_belief() const {
  Belief b;
  for (const auto &p : _belief_particles) {
    std::size_t h = typename State::Hash()(p.first);
    b[h] += p.second;
  }
  return b;
}

// --- Observation-based interface ---

SK_DESPOT_TEMPLATE_DECL
const typename SK_DESPOT_CLASS::Action &
SK_DESPOT_CLASS::get_best_action(const Observation &obs) {
  update_belief_particles(obs);
  Belief b = particles_to_belief();
  plan_from_belief(b);
  _last_action = std::make_unique<Action>(_best_action_cache);
  return _best_action_cache;
}

SK_DESPOT_TEMPLATE_DECL
typename SK_DESPOT_CLASS::Value
SK_DESPOT_CLASS::get_best_value(const Observation &obs) {
  update_belief_particles(obs);
  Belief b = particles_to_belief();
  plan_from_belief(b);
  return Value(_best_value_cache, true);
}

SK_DESPOT_TEMPLATE_DECL
bool SK_DESPOT_CLASS::is_solution_defined_for(const Observation &obs) {
  return _has_solution;
}

SK_DESPOT_TEMPLATE_DECL
void SK_DESPOT_CLASS::reset_belief() {
  _last_action.reset();
  _current_tree.reset();
}

// --- Belief-based interface ---

SK_DESPOT_TEMPLATE_DECL
const typename SK_DESPOT_CLASS::Action &
SK_DESPOT_CLASS::get_best_action_from_belief(const Belief &b) {
  plan_from_belief(b);
  return _best_action_cache;
}

SK_DESPOT_TEMPLATE_DECL
typename SK_DESPOT_CLASS::Value
SK_DESPOT_CLASS::get_best_value_from_belief(const Belief &b) {
  plan_from_belief(b);
  return Value(_best_value_cache, true);
}

SK_DESPOT_TEMPLATE_DECL
bool SK_DESPOT_CLASS::is_solution_defined_for_from_belief(const Belief &b) {
  return _has_solution;
}

// --- Statistics ---

SK_DESPOT_TEMPLATE_DECL
std::size_t SK_DESPOT_CLASS::get_nb_tree_nodes() const {
  return _nb_tree_nodes;
}

SK_DESPOT_TEMPLATE_DECL
std::size_t SK_DESPOT_CLASS::get_solving_time() const { return elapsed_ms(); }

SK_DESPOT_TEMPLATE_DECL
double SK_DESPOT_CLASS::get_gap() const { return _gap_cache; }

SK_DESPOT_TEMPLATE_DECL
std::vector<typename SK_DESPOT_CLASS::BeliefNode>
SK_DESPOT_CLASS::get_explored_beliefs() const {
  std::vector<BeliefNode> beliefs;

  if (!_current_tree) {
    return beliefs;
  }

  // BFS traversal of the tree
  std::queue<VNode *> queue;
  queue.push(_current_tree.get());

  while (!queue.empty()) {
    VNode *v = queue.front();
    queue.pop();

    BeliefNode node;

    // Extract particles (states only)
    for (const auto &scenario : v->scenarios) {
      node.particles.push_back(scenario.state);
    }

    node.lower_bound = v->lower_bound;
    node.upper_bound = v->upper_bound;
    node.default_value = v->default_value;
    node.depth = v->depth;

    // Find best action (action with highest lower bound)
    if (v->is_expanded && !v->children.empty()) {
      auto best_child = std::max_element(
          v->children.begin(), v->children.end(),
          [](const std::unique_ptr<QNode> &a, const std::unique_ptr<QNode> &b) {
            return a->lower_bound < b->lower_bound;
          });
      node.best_action = (*best_child)->action;
    }

    beliefs.push_back(std::move(node));

    // Add children to queue
    for (const auto &qchild : v->children) {
      for (const auto &[obs_hash, vchild] : qchild->children) {
        queue.push(vchild.get());
      }
    }
  }

  return beliefs;
}

SK_DESPOT_TEMPLATE_DECL
std::vector<std::pair<typename SK_DESPOT_CLASS::State,
                      typename SK_DESPOT_CLASS::Action>>
SK_DESPOT_CLASS::get_last_trajectory() const {
  std::vector<std::pair<State, Action>> trajectory;
  const_cast<DespotSolver *>(this)->_execution_policy.protect(
      [&]() {
        // Reconstruct trajectory from leaf to root on-demand
        if (!_last_explored_leaf)
          return;

        // Walk from leaf to root collecting (state, action) pairs
        std::vector<std::pair<State, Action>> path_reversed;
        VNode *v = _last_explored_leaf;

        while (v && v->parent) {
          QNode *q = v->parent;
          VNode *parent_v = q->parent;
          if (parent_v && !parent_v->scenarios.empty()) {
            // Use representative state from parent node
            path_reversed.push_back(
                std::make_pair(parent_v->scenarios[0].state, q->action));
          }
          v = parent_v;
        }

        // Reverse to get root-to-leaf order
        trajectory.assign(path_reversed.rbegin(), path_reversed.rend());
      },
      const_cast<typename ExecutionPolicy::Mutex &>(_trajectory_mutex));
  return trajectory;
}

} // namespace skdecide

#endif // SKDECIDE_DESPOT_IMPL_HH
