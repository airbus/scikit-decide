/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Implementation of SARSOP from:
 * Kurniawati, Hsu & Lee, "SARSOP: Efficient Point-Based POMDP Planning
 * by Approximating Optimally Reachable Belief Spaces", RSS 2008.
 */
#ifndef SKDECIDE_SARSOP_IMPL_HH
#define SKDECIDE_SARSOP_IMPL_HH

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

#define SK_SARSOP_TEMPLATE_DECL                                                \
  template <typename Tdomain, typename Texecution_policy>

#define SK_SARSOP_CLASS SARSOPSolver<Tdomain, Texecution_policy>

// --- Constructor ---

SK_SARSOP_TEMPLATE_DECL
SK_SARSOP_CLASS::SARSOPSolver(
    Domain &domain, double epsilon, double discount, std::size_t time_budget,
    std::size_t max_beliefs, double pruning_delta,
    std::size_t max_vi_iterations, double vi_convergence_factor,
    std::size_t max_sample_depth, double prob_epsilon,
    double ub_improvement_epsilon, std::size_t pruning_interval,
    std::size_t logging_interval, const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _epsilon(epsilon), _discount(discount),
      _time_budget(time_budget), _max_beliefs(max_beliefs),
      _pruning_delta(pruning_delta), _max_vi_iterations(max_vi_iterations),
      _vi_convergence_factor(vi_convergence_factor),
      _max_sample_depth(max_sample_depth), _prob_epsilon(prob_epsilon),
      _ub_improvement_epsilon(ub_improvement_epsilon),
      _pruning_interval(pruning_interval), _logging_interval(logging_interval),
      _callback(callback), _verbose(verbose), _next_alpha_id(0), _nb_beliefs(0),
      _has_solution(false) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm SARSOP");
  }
}

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::clear() {
  _states.clear();
  _state_hash_to_idx.clear();
  _index_to_state.clear();
  _transitions.clear();
  _obs_prob.clear();
  _obs_objects.clear();
  _rewards.clear();
  _actions.clear();
  _action_hash_to_idx.clear();
  _action_obs_hashes.clear();
  _alpha_vectors.clear();
  _next_alpha_id = 0;
  _mdp_values.clear();
  _ub_points.clear();
  _root.reset();
  _nb_beliefs = 0;
  _current_belief.clear();
  _last_action.reset();
  _has_solution = false;
}

// --- State indexing ---

SK_SARSOP_TEMPLATE_DECL
std::size_t SK_SARSOP_CLASS::get_state_index(const State &s) {
  std::size_t h = typename State::Hash()(s);
  if (_index_to_state.find(h) == _index_to_state.end()) {
    _index_to_state[h] = s;
  }
  return h;
}

SK_SARSOP_TEMPLATE_DECL
const std::unordered_map<std::size_t, typename SK_SARSOP_CLASS::State> &
SK_SARSOP_CLASS::get_index_to_state() const {
  return _index_to_state;
}

// --- Timing ---

SK_SARSOP_TEMPLATE_DECL
std::size_t SK_SARSOP_CLASS::elapsed_ms() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now() - _start_time)
      .count();
}

// --- State enumeration (BFS from belief support) ---

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::enumerate_states(const Belief &b0) {
  if (_verbose)
    Logger::debug("SARSOP: enumerating reachable states");

  std::queue<std::size_t> frontier;
  std::unordered_set<std::size_t> visited;

  // Seed with b0 support
  for (const auto &p : b0) {
    if (visited.insert(p.first).second) {
      frontier.push(p.first);
    }
  }

  // Collect all unique actions from any state
  std::unordered_set<std::size_t> action_hashes_seen;

  while (!frontier.empty()) {
    std::size_t sh = frontier.front();
    frontier.pop();

    const State &s = _index_to_state.at(sh);

    if (_domain.is_terminal(s))
      continue;

    auto applicable = _domain.get_applicable_actions(s).get_elements();
    std::for_each(
        ExecutionPolicy::policy, applicable.begin(), applicable.end(),
        [this, &s, &frontier, &visited, &action_hashes_seen](auto a) {
          std::size_t ah = typename Action::Hash()(a);

          auto next_dist =
              _domain.get_next_state_distribution(s, a).get_values();

          _execution_policy.protect([this, &a, ah, &action_hashes_seen] {
            if (action_hashes_seen.insert(ah).second) {
              _actions.push_back(a);
              _action_hash_to_idx[ah] = _actions.size() - 1;
            }
          });

          for (auto ns : next_dist) {
            _execution_policy.protect([this, &ns, &frontier, &visited] {
              std::size_t nsh = get_state_index(ns.state());
              if (visited.insert(nsh).second) {
                frontier.push(nsh);
              }
            });
          }
        });
  }

  // Build position-indexed state vector
  _states.clear();
  _state_hash_to_idx.clear();
  for (const auto &p : _index_to_state) {
    _state_hash_to_idx[p.first] = _states.size();
    _states.push_back(p.second);
  }

  if (_verbose)
    Logger::debug("SARSOP: found " + std::to_string(_states.size()) +
                  " states and " + std::to_string(_actions.size()) +
                  " actions");
}

// --- Model pre-caching ---

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::pre_cache_model() {
  std::size_t ns = _states.size();
  std::size_t na = _actions.size();

  _transitions.resize(ns);
  _obs_prob.resize(ns);
  _rewards.resize(ns, std::vector<double>(na, 0.0));
  _action_obs_hashes.resize(na);

  // Track which observations we've seen per action
  std::vector<std::unordered_set<std::size_t>> action_obs_sets(na);

  for (std::size_t i = 0; i < ns; ++i) {
    _transitions[i].resize(na);
    _obs_prob[i].resize(na);
  }

  std::vector<std::size_t> state_indices(ns);
  std::iota(state_indices.begin(), state_indices.end(), 0);

  std::for_each(
      ExecutionPolicy::policy, state_indices.begin(), state_indices.end(),
      [this, na, &action_obs_sets](std::size_t si) {
        const State &s = _states[si];
        if (_domain.is_terminal(s))
          return;

        for (std::size_t ai = 0; ai < na; ++ai) {
          auto next_dist =
              _domain.get_next_state_distribution(s, _actions[ai]).get_values();

          double weighted_reward = 0.0;
          for (auto ns_item : next_dist) {
            std::size_t nsh = typename State::Hash()(ns_item.state());
            auto it = _state_hash_to_idx.find(nsh);
            if (it == _state_hash_to_idx.end())
              continue;
            std::size_t ns_idx = it->second;
            double prob = ns_item.probability();

            _transitions[si][ai].push_back({prob, ns_idx});

            double r =
                _domain.get_transition_value(s, _actions[ai], _states[ns_idx])
                    .reward();
            weighted_reward += prob * r;

            auto obs_dist =
                _domain
                    .get_observation_distribution(_states[ns_idx], _actions[ai])
                    .get_values();

            _execution_policy.protect([this, &obs_dist, ns_idx, ai,
                                       &action_obs_sets] {
              for (auto od : obs_dist) {
                std::size_t oh = typename Observation::Hash()(od.observation());
                _obs_prob[ns_idx][ai][oh] = od.probability();
                if (_obs_objects.find(oh) == _obs_objects.end()) {
                  _obs_objects[oh] = od.observation();
                }
                action_obs_sets[ai].insert(oh);
              }
            });
          }

          _rewards[si][ai] = weighted_reward;
        }
      });

  // Flatten observation sets to vectors for iteration
  for (std::size_t ai = 0; ai < na; ++ai) {
    _action_obs_hashes[ai].assign(action_obs_sets[ai].begin(),
                                  action_obs_sets[ai].end());
  }

  if (_verbose)
    Logger::debug("SARSOP: model pre-cached");
}

// --- Alpha-vector operations ---

SK_SARSOP_TEMPLATE_DECL
double SK_SARSOP_CLASS::dot_product(const AlphaVector &alpha,
                                    const Belief &b) const {
  double result = 0.0;
  for (const auto &p : b) {
    auto it = _state_hash_to_idx.find(p.first);
    if (it != _state_hash_to_idx.end()) {
      result += alpha.values[it->second] * p.second;
    }
  }
  return result;
}

SK_SARSOP_TEMPLATE_DECL
double SK_SARSOP_CLASS::evaluate_lower(const Belief &b) const {
  if (_alpha_vectors.empty())
    return -std::numeric_limits<double>::infinity();
  double best = -std::numeric_limits<double>::infinity();
  for (const auto &alpha : _alpha_vectors) {
    double v = dot_product(alpha, b);
    if (v > best)
      best = v;
  }
  return best;
}

SK_SARSOP_TEMPLATE_DECL
std::size_t SK_SARSOP_CLASS::best_alpha_index(const Belief &b) const {
  double best = -std::numeric_limits<double>::infinity();
  std::size_t best_idx = 0;
  for (std::size_t i = 0; i < _alpha_vectors.size(); ++i) {
    double v = dot_product(_alpha_vectors[i], b);
    if (v > best) {
      best = v;
      best_idx = i;
    }
  }
  return best_idx;
}

// --- Upper bound ---

SK_SARSOP_TEMPLATE_DECL
double SK_SARSOP_CLASS::evaluate_upper_corner(const Belief &b) const {
  double v = 0.0;
  for (const auto &p : b) {
    auto it = _state_hash_to_idx.find(p.first);
    if (it != _state_hash_to_idx.end()) {
      v += p.second * _mdp_values[it->second];
    }
  }
  return v;
}

SK_SARSOP_TEMPLATE_DECL
double SK_SARSOP_CLASS::evaluate_upper(const Belief &b) const {
  double v_corner = evaluate_upper_corner(b);

  for (const auto &pt : _ub_points) {
    // Compute c = min_{s in support(b_i)} b(s) / b_i(s)
    double c = std::numeric_limits<double>::infinity();
    bool valid = true;
    for (const auto &p : pt.belief) {
      if (p.second <= 0.0)
        continue;
      auto it = b.find(p.first);
      double b_s = (it != b.end()) ? it->second : 0.0;
      double ratio = b_s / p.second;
      if (ratio <= 0.0) {
        valid = false;
        break;
      }
      c = std::min(c, ratio);
    }
    if (!valid || c <= 0.0 || std::isinf(c))
      continue;

    // Compute residual belief and its corner value
    Belief b_res;
    double one_minus_c = 1.0 - c;
    if (one_minus_c <= _prob_epsilon) {
      // c >= 1: the point fully covers the belief
      v_corner = std::min(v_corner, pt.value);
      continue;
    }
    double v_res = 0.0;
    for (const auto &p : b) {
      auto pt_it = pt.belief.find(p.first);
      double pt_s = (pt_it != pt.belief.end()) ? pt_it->second : 0.0;
      double res = (p.second - c * pt_s) / one_minus_c;
      if (res > 0.0) {
        auto idx_it = _state_hash_to_idx.find(p.first);
        if (idx_it != _state_hash_to_idx.end()) {
          v_res += res * _mdp_values[idx_it->second];
        }
      }
    }

    double candidate = c * pt.value + one_minus_c * v_res;
    v_corner = std::min(v_corner, candidate);
  }

  return v_corner;
}

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::update_upper_bound(BeliefTreeNode *node) {
  if (!node->expanded)
    return;

  double old_ub = node->upper_bound;

  // Recompute Q_upper for each action from children
  double best_q_upper = -std::numeric_limits<double>::infinity();
  for (auto &ae : node->action_edges) {
    double q = ae.expected_reward;
    for (const auto &child_entry : ae.children) {
      auto obs_it = ae.obs_probs.find(child_entry.first);
      double obs_p = (obs_it != ae.obs_probs.end()) ? obs_it->second : 0.0;
      q += _discount * obs_p * child_entry.second->upper_bound;
    }
    ae.q_upper = q;
    best_q_upper = std::max(best_q_upper, q);
  }

  // Also consider the sawtooth evaluation
  double sawtooth_ub = evaluate_upper(node->belief);
  node->upper_bound = std::min(best_q_upper, sawtooth_ub);

  // Add interior point only if the upper bound actually improved
  if (node->upper_bound < old_ub - _ub_improvement_epsilon) {
    _ub_points.push_back({node->belief, node->upper_bound});
  }
}

// --- Bound initialization ---

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::initialize_lower_bound() {
  std::size_t ns = _states.size();
  std::size_t na = _actions.size();

  // Blind policy: for each action a, compute the fixed-action value
  // alpha_a(s) = R(s,a) + gamma * sum_{s'} T(s'|s,a) * alpha_a(s')
  for (std::size_t ai = 0; ai < na; ++ai) {
    AlphaVector alpha(ns, _actions[ai], _next_alpha_id++);

    // Initialize with R(s,a) / (1-gamma) if discount < 1
    if (_discount < 1.0) {
      for (std::size_t si = 0; si < ns; ++si) {
        alpha.values[si] = _rewards[si][ai] / (1.0 - _discount);
      }
    }

    // Iterate to convergence (parallel Jacobi-style sweeps)
    std::vector<std::size_t> lb_state_indices(ns);
    std::iota(lb_state_indices.begin(), lb_state_indices.end(), 0);

    for (int iter = 0; iter < _max_vi_iterations; ++iter) {
      double max_change = 0.0;
      std::vector<double> new_values(ns);
      std::for_each(
          ExecutionPolicy::policy, lb_state_indices.begin(),
          lb_state_indices.end(),
          [this, ai, &alpha, &new_values, &max_change](std::size_t si) {
            double v = _rewards[si][ai];
            for (const auto &tr : _transitions[si][ai]) {
              v += _discount * tr.first * alpha.values[tr.second];
            }
            new_values[si] = v;
            double change = std::abs(v - alpha.values[si]);
            _execution_policy.protect([&max_change, change] {
              max_change = std::max(max_change, change);
            });
          });
      alpha.values = std::move(new_values);
      if (max_change < _epsilon * _vi_convergence_factor)
        break;
    }

    _alpha_vectors.push_back(std::move(alpha));
  }

  if (_verbose)
    Logger::debug("SARSOP: initialized " +
                  std::to_string(_alpha_vectors.size()) +
                  " blind policy alpha-vectors");
}

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::initialize_upper_bound() {
  std::size_t ns = _states.size();
  std::size_t na = _actions.size();

  // MDP value iteration (fully observable upper bound)
  _mdp_values.resize(ns, 0.0);

  // Initialize with max_a R(s,a) / (1-gamma)
  if (_discount < 1.0) {
    for (std::size_t si = 0; si < ns; ++si) {
      double best_r = -std::numeric_limits<double>::infinity();
      for (std::size_t ai = 0; ai < na; ++ai) {
        best_r = std::max(best_r, _rewards[si][ai]);
      }
      _mdp_values[si] = (best_r > -std::numeric_limits<double>::infinity())
                            ? best_r / (1.0 - _discount)
                            : 0.0;
    }
  }

  // Iterate (parallel Jacobi-style sweeps)
  std::vector<std::size_t> ub_state_indices(ns);
  std::iota(ub_state_indices.begin(), ub_state_indices.end(), 0);

  for (int iter = 0; iter < _max_vi_iterations; ++iter) {
    double max_change = 0.0;
    std::vector<double> new_values(ns);
    std::for_each(ExecutionPolicy::policy, ub_state_indices.begin(),
                  ub_state_indices.end(),
                  [this, na, &new_values, &max_change](std::size_t si) {
                    double best_v = -std::numeric_limits<double>::infinity();
                    for (std::size_t ai = 0; ai < na; ++ai) {
                      double v = _rewards[si][ai];
                      for (const auto &tr : _transitions[si][ai]) {
                        v += _discount * tr.first * _mdp_values[tr.second];
                      }
                      best_v = std::max(best_v, v);
                    }
                    if (best_v <= -std::numeric_limits<double>::infinity())
                      best_v = 0.0;
                    new_values[si] = best_v;
                    double change = std::abs(best_v - _mdp_values[si]);
                    _execution_policy.protect([&max_change, change] {
                      max_change = std::max(max_change, change);
                    });
                  });
    _mdp_values = std::move(new_values);
    if (max_change < _epsilon * _vi_convergence_factor)
      break;
  }

  if (_verbose)
    Logger::debug("SARSOP: MDP upper bound initialized");
}

// --- Belief operations ---

SK_SARSOP_TEMPLATE_DECL
typename SK_SARSOP_CLASS::Belief
SK_SARSOP_CLASS::compute_posterior(const Belief &b, std::size_t action_idx,
                                   std::size_t obs_hash) const {
  // Prediction: b_a(s') = sum_s T(s'|s,a) * b(s)
  Belief b_a;
  for (const auto &p : b) {
    auto idx_it = _state_hash_to_idx.find(p.first);
    if (idx_it == _state_hash_to_idx.end())
      continue;
    std::size_t si = idx_it->second;
    for (const auto &tr : _transitions[si][action_idx]) {
      std::size_t ns_hash = typename State::Hash()(_states[tr.second]);
      b_a[ns_hash] += p.second * tr.first;
    }
  }

  // Posterior: b^o_a(s') = Z(o|s',a) * b_a(s') / P(o|b,a)
  Belief posterior;
  double normalizer = 0.0;
  for (const auto &p : b_a) {
    auto idx_it = _state_hash_to_idx.find(p.first);
    if (idx_it == _state_hash_to_idx.end())
      continue;
    std::size_t sp_idx = idx_it->second;
    auto obs_it = _obs_prob[sp_idx][action_idx].find(obs_hash);
    double z =
        (obs_it != _obs_prob[sp_idx][action_idx].end()) ? obs_it->second : 0.0;
    double val = z * p.second;
    if (val > 0.0) {
      posterior[p.first] = val;
      normalizer += val;
    }
  }

  // Normalize
  if (normalizer > 0.0) {
    for (auto &p : posterior) {
      p.second /= normalizer;
    }
  }

  return posterior;
}

// --- Belief tree ---

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::initialize_belief_node(BeliefTreeNode *node) {
  node->lower_bound = evaluate_lower(node->belief);
  node->upper_bound = evaluate_upper(node->belief);
  node->best_alpha_idx = best_alpha_index(node->belief);
}

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::expand_node(BeliefTreeNode *node) {
  if (node->expanded)
    return;

  std::size_t na = _actions.size();
  node->action_edges.reserve(na);

  for (std::size_t ai = 0; ai < na; ++ai) {
    node->action_edges.emplace_back(_actions[ai]);
    auto &ae = node->action_edges.back();

    // Expected immediate reward: R(b,a) = sum_s b(s) * R(s,a)
    ae.expected_reward = 0.0;
    for (const auto &bp : node->belief) {
      auto idx_it = _state_hash_to_idx.find(bp.first);
      if (idx_it != _state_hash_to_idx.end()) {
        ae.expected_reward += bp.second * _rewards[idx_it->second][ai];
      }
    }

    // Compute P(o|b,a) for each observation
    // P(o|b,a) = sum_s sum_{s'} b(s) T(s'|s,a) Z(o|s',a)
    for (const auto &bp : node->belief) {
      auto idx_it = _state_hash_to_idx.find(bp.first);
      if (idx_it == _state_hash_to_idx.end())
        continue;
      std::size_t si = idx_it->second;
      for (const auto &tr : _transitions[si][ai]) {
        for (const auto &op : _obs_prob[tr.second][ai]) {
          ae.obs_probs[op.first] += bp.second * tr.first * op.second;
        }
      }
    }

    // Create children for observations with positive probability
    ae.q_lower = ae.expected_reward;
    ae.q_upper = ae.expected_reward;

    for (const auto &op : ae.obs_probs) {
      if (op.second <= _prob_epsilon)
        continue;

      Belief posterior = compute_posterior(node->belief, ai, op.first);
      if (posterior.empty())
        continue;

      auto child = std::make_shared<BeliefTreeNode>();
      child->belief = std::move(posterior);
      child->parent = node;
      child->depth = node->depth + 1;
      initialize_belief_node(child.get());
      _nb_beliefs++;

      ae.q_lower += _discount * op.second * child->lower_bound;
      ae.q_upper += _discount * op.second * child->upper_bound;

      ae.children[op.first] = std::move(child);
    }
  }

  node->expanded = true;

  // Update node bounds from action edges
  double best_lower = -std::numeric_limits<double>::infinity();
  double best_upper = -std::numeric_limits<double>::infinity();
  for (const auto &ae : node->action_edges) {
    best_lower = std::max(best_lower, ae.q_lower);
    best_upper = std::max(best_upper, ae.q_upper);
  }
  if (best_lower > node->lower_bound)
    node->lower_bound = best_lower;
  if (best_upper < node->upper_bound)
    node->upper_bound = best_upper;
}

// --- SARSOP core: sample ---

SK_SARSOP_TEMPLATE_DECL
std::vector<typename SK_SARSOP_CLASS::BeliefTreeNode *>
SK_SARSOP_CLASS::sample() {
  std::vector<BeliefTreeNode *> path;
  BeliefTreeNode *node = _root.get();
  path.push_back(node);

  std::size_t max_depth = _max_sample_depth;

  while (node->depth < max_depth && !node->pruned) {
    double gap = node->upper_bound - node->lower_bound;
    if (gap < _epsilon)
      break;

    if (!node->expanded) {
      expand_node(node);
      if (node->action_edges.empty())
        break;
    }

    // Pick action with max Q_upper
    double best_q = -std::numeric_limits<double>::infinity();
    std::size_t best_a_idx = 0;
    for (std::size_t ai = 0; ai < node->action_edges.size(); ++ai) {
      if (node->action_edges[ai].q_upper > best_q) {
        best_q = node->action_edges[ai].q_upper;
        best_a_idx = ai;
      }
    }

    auto &ae = node->action_edges[best_a_idx];
    if (ae.children.empty())
      break;

    // Pick observation with largest weighted gap at successor
    double best_gap = -std::numeric_limits<double>::infinity();
    std::size_t best_obs_hash = 0;
    BeliefTreeNode *best_child = nullptr;

    for (auto &child_entry : ae.children) {
      BeliefTreeNode *child = child_entry.second.get();
      auto obs_it = ae.obs_probs.find(child_entry.first);
      double obs_p = (obs_it != ae.obs_probs.end()) ? obs_it->second : 0.0;
      double child_gap = obs_p * (child->upper_bound - child->lower_bound);
      if (child_gap > best_gap) {
        best_gap = child_gap;
        best_obs_hash = child_entry.first;
        best_child = child;
      }
    }

    if (best_child == nullptr)
      break;

    node = best_child;
    path.push_back(node);
  }

  return path;
}

// --- SARSOP core: backup ---

SK_SARSOP_TEMPLATE_DECL
typename SK_SARSOP_CLASS::AlphaVector
SK_SARSOP_CLASS::backup_belief(BeliefTreeNode *node) {
  std::size_t ns = _states.size();
  std::size_t na = _actions.size();

  double best_q = -std::numeric_limits<double>::infinity();
  AlphaVector best_alpha;

  for (std::size_t ai = 0; ai < na; ++ai) {
    AlphaVector g_a(ns, _actions[ai], _next_alpha_id);

    // Start with R(s,a)
    for (std::size_t si = 0; si < ns; ++si) {
      g_a.values[si] = _rewards[si][ai];
    }

    // For each observation, add gamma * g_{a,o}(s)
    for (std::size_t oh : _action_obs_hashes[ai]) {
      // Find best alpha for posterior belief tau(b,a,o)
      Belief posterior = compute_posterior(node->belief, ai, oh);
      if (posterior.empty())
        continue;

      std::size_t best_idx = best_alpha_index(posterior);
      const AlphaVector &alpha_ao = _alpha_vectors[best_idx];

      // g_{a,o}(s) = sum_{s'} T(s,a,s') * Z(o|s',a) * alpha_{a,o}(s')
      for (std::size_t si = 0; si < ns; ++si) {
        double contrib = 0.0;
        for (const auto &tr : _transitions[si][ai]) {
          auto obs_it = _obs_prob[tr.second][ai].find(oh);
          double z =
              (obs_it != _obs_prob[tr.second][ai].end()) ? obs_it->second : 0.0;
          contrib += tr.first * z * alpha_ao.values[tr.second];
        }
        g_a.values[si] += _discount * contrib;
      }
    }

    // Compute Q(b,a) = g_a . b
    double q_val = dot_product(g_a, node->belief);
    if (q_val > best_q) {
      best_q = q_val;
      best_alpha = std::move(g_a);
    }
  }

  best_alpha.id = _next_alpha_id++;
  _alpha_vectors.push_back(best_alpha);

  // Update node bounds
  node->lower_bound = best_q;
  node->best_alpha_idx = _alpha_vectors.size() - 1;

  return best_alpha;
}

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::backup(const std::vector<BeliefTreeNode *> &path) {
  // Bottom-up backup along sampled path
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    BeliefTreeNode *node = *it;
    backup_belief(node);
    update_upper_bound(node);
  }
}

// --- SARSOP core: prune ---

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::prune() {
  if (_alpha_vectors.size() <= 1)
    return;

  // Delta-dominance pruning: remove alpha_i if there exists alpha_j
  // such that alpha_j(s) >= alpha_i(s) - delta for all s
  std::size_t ns = _states.size();
  std::vector<bool> dominated(_alpha_vectors.size(), false);

  for (std::size_t i = 0; i < _alpha_vectors.size(); ++i) {
    if (dominated[i])
      continue;
    for (std::size_t j = 0; j < _alpha_vectors.size(); ++j) {
      if (i == j || dominated[j])
        continue;
      // Check if j dominates i (j >= i - delta for all s)
      bool j_dominates_i = true;
      for (std::size_t s = 0; s < ns; ++s) {
        if (_alpha_vectors[j].values[s] <
            _alpha_vectors[i].values[s] - _pruning_delta) {
          j_dominates_i = false;
          break;
        }
      }
      if (j_dominates_i) {
        dominated[i] = true;
        break;
      }
    }
  }

  // Remove dominated vectors
  std::vector<AlphaVector> survivors;
  survivors.reserve(_alpha_vectors.size());
  for (std::size_t i = 0; i < _alpha_vectors.size(); ++i) {
    if (!dominated[i]) {
      survivors.push_back(std::move(_alpha_vectors[i]));
    }
  }

  if (_verbose && survivors.size() < _alpha_vectors.size()) {
    Logger::debug("SARSOP: pruned " +
                  std::to_string(_alpha_vectors.size() - survivors.size()) +
                  " alpha-vectors, " + std::to_string(survivors.size()) +
                  " remaining");
  }

  _alpha_vectors = std::move(survivors);
}

// --- solve ---

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::solve(
    const std::vector<std::pair<State, double>> &initial_distribution) {
  _start_time = std::chrono::high_resolution_clock::now();
  clear();

  // Build initial belief
  Belief b0;
  for (const auto &p : initial_distribution) {
    std::size_t sh = get_state_index(p.first);
    b0[sh] = p.second;
  }

  // Enumerate states and pre-cache model
  enumerate_states(b0);
  pre_cache_model();

  // Initialize bounds
  initialize_lower_bound();
  initialize_upper_bound();

  // Build root belief tree node
  _root = std::make_unique<BeliefTreeNode>();
  _root->belief = b0;
  initialize_belief_node(_root.get());
  _nb_beliefs = 1;

  if (_verbose) {
    Logger::debug(
        "SARSOP: initial lower bound = " + std::to_string(_root->lower_bound) +
        ", upper bound = " + std::to_string(_root->upper_bound) +
        ", gap = " + std::to_string(_root->upper_bound - _root->lower_bound));
  }

  // Main loop
  std::size_t iteration = 0;
  while (true) {
    double gap = _root->upper_bound - _root->lower_bound;
    if (gap < _epsilon) {
      if (_verbose)
        Logger::debug("SARSOP: converged at gap = " + std::to_string(gap));
      break;
    }
    if (elapsed_ms() >= _time_budget) {
      if (_verbose)
        Logger::debug("SARSOP: time budget reached");
      break;
    }
    if (_nb_beliefs >= _max_beliefs) {
      if (_verbose)
        Logger::debug("SARSOP: belief budget reached");
      break;
    }
    if (_callback(*this, _domain)) {
      if (_verbose)
        Logger::debug("SARSOP: stopped by callback");
      break;
    }

    auto path = sample();
    backup(path);

    // Prune periodically
    if (_pruning_interval > 0 && iteration % _pruning_interval == 0) {
      prune();
    }

    // Update root bounds after backup
    _root->lower_bound =
        std::max(_root->lower_bound, evaluate_lower(_root->belief));
    _root->upper_bound =
        std::min(_root->upper_bound, evaluate_upper(_root->belief));

    ++iteration;

    if (_verbose && _logging_interval > 0 &&
        iteration % _logging_interval == 0) {
      Logger::debug(
          "SARSOP: iteration " + std::to_string(iteration) +
          ", gap = " + std::to_string(_root->upper_bound - _root->lower_bound) +
          ", alpha-vectors = " + std::to_string(_alpha_vectors.size()) +
          ", beliefs = " + std::to_string(_nb_beliefs));
    }
  }

  // Final prune
  prune();

  // Set up belief tracking
  _current_belief = b0;
  _last_action.reset();
  _has_solution = true;

  if (_verbose)
    Logger::debug("SARSOP: solved in " + std::to_string(elapsed_ms()) + "ms, " +
                  std::to_string(iteration) + " iterations, " +
                  std::to_string(_alpha_vectors.size()) + " alpha-vectors, " +
                  std::to_string(_nb_beliefs) + " beliefs, final gap = " +
                  std::to_string(_root->upper_bound - _root->lower_bound));
}

// --- Observation-based interface ---

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::update_current_belief(const Observation &obs) {
  if (!_last_action)
    return;

  // Find the action index
  std::size_t ah = typename Action::Hash()(*_last_action);
  auto ait = _action_hash_to_idx.find(ah);
  if (ait == _action_hash_to_idx.end())
    return;

  std::size_t obs_hash = typename Observation::Hash()(obs);
  Belief posterior = compute_posterior(_current_belief, ait->second, obs_hash);
  if (!posterior.empty()) {
    _current_belief = std::move(posterior);
  }
}

SK_SARSOP_TEMPLATE_DECL
const typename SK_SARSOP_CLASS::Action &
SK_SARSOP_CLASS::get_best_action(const Observation &obs) {
  update_current_belief(obs);
  std::size_t idx = best_alpha_index(_current_belief);
  _last_action = std::make_unique<Action>(_alpha_vectors[idx].action);
  return _alpha_vectors[idx].action;
}

SK_SARSOP_TEMPLATE_DECL
typename SK_SARSOP_CLASS::Value
SK_SARSOP_CLASS::get_best_value(const Observation &obs) {
  update_current_belief(obs);
  double v = evaluate_lower(_current_belief);
  return Value(v, true);
}

SK_SARSOP_TEMPLATE_DECL
bool SK_SARSOP_CLASS::is_solution_defined_for(const Observation &obs) {
  return _has_solution && !_alpha_vectors.empty();
}

SK_SARSOP_TEMPLATE_DECL
void SK_SARSOP_CLASS::reset_belief() {
  if (_root) {
    _current_belief = _root->belief;
  }
  _last_action.reset();
}

// --- Belief-based interface ---

SK_SARSOP_TEMPLATE_DECL
const typename SK_SARSOP_CLASS::Action &
SK_SARSOP_CLASS::get_best_action_from_belief(const Belief &b) {
  std::size_t idx = best_alpha_index(b);
  return _alpha_vectors[idx].action;
}

SK_SARSOP_TEMPLATE_DECL
typename SK_SARSOP_CLASS::Value
SK_SARSOP_CLASS::get_best_value_from_belief(const Belief &b) {
  double v = evaluate_lower(b);
  return Value(v, true);
}

SK_SARSOP_TEMPLATE_DECL
bool SK_SARSOP_CLASS::is_solution_defined_for_from_belief(const Belief &b) {
  return _has_solution && !_alpha_vectors.empty();
}

// --- Statistics ---

SK_SARSOP_TEMPLATE_DECL
std::size_t SK_SARSOP_CLASS::get_nb_alpha_vectors() const {
  return _alpha_vectors.size();
}

SK_SARSOP_TEMPLATE_DECL
std::size_t SK_SARSOP_CLASS::get_nb_explored_beliefs() const {
  return _nb_beliefs;
}

SK_SARSOP_TEMPLATE_DECL
std::size_t SK_SARSOP_CLASS::get_solving_time() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now() - _start_time)
      .count();
}

SK_SARSOP_TEMPLATE_DECL
double SK_SARSOP_CLASS::get_initial_lower_bound() const {
  return _root ? _root->lower_bound : -std::numeric_limits<double>::infinity();
}

SK_SARSOP_TEMPLATE_DECL
double SK_SARSOP_CLASS::get_initial_upper_bound() const {
  return _root ? _root->upper_bound : std::numeric_limits<double>::infinity();
}

SK_SARSOP_TEMPLATE_DECL
double SK_SARSOP_CLASS::get_gap() const {
  if (_root) {
    return _root->upper_bound - _root->lower_bound;
  }
  return std::numeric_limits<double>::infinity();
}

} // namespace skdecide

#endif // SKDECIDE_SARSOP_IMPL_HH
