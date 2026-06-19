/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_HSVI_IMPL_HH
#define SKDECIDE_HSVI_IMPL_HH

#include <algorithm>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_set>

#define SK_HSVI_TEMPLATE_DECL                                                  \
  template <typename Tdomain, typename Texecution_policy>

#define SK_HSVI_CLASS HSVISolver<Tdomain, Texecution_policy>

namespace skdecide {

// =============================================================================
// HSVISolver implementation (reward maximization)
// =============================================================================

SK_HSVI_TEMPLATE_DECL
SK_HSVI_CLASS::HSVISolver(Domain &domain, double epsilon, double discount,
                          std::size_t time_budget, std::size_t max_sample_depth,
                          bool use_closed_list, double depth_bound_eta,
                          std::size_t max_vi_iterations,
                          double vi_convergence_factor, double prob_epsilon,
                          const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _epsilon(epsilon), _discount(discount),
      _time_budget(time_budget), _max_sample_depth(max_sample_depth),
      _use_closed_list(use_closed_list), _depth_bound_eta(depth_bound_eta),
      _max_vi_iterations(max_vi_iterations),
      _vi_convergence_factor(vi_convergence_factor),
      _prob_epsilon(prob_epsilon), _callback(callback), _verbose(verbose) {}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::clear() {
  _states.clear();
  _state_hash_to_idx.clear();
  _index_to_state.clear();
  _actions.clear();
  _action_hash_to_idx.clear();
  _transitions.clear();
  _obs_prob.clear();
  _values.clear();
  _obs_objects.clear();
  _action_obs_hashes.clear();
  _alpha_vectors.clear();
  _next_alpha_id = 0;
  _mdp_values.clear();
  _bound_points.clear();
  _is_terminal_cache.clear();
  _has_last_action = false;
  _depth_bound = 0;
  _gap = std::numeric_limits<double>::infinity();
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::enumerate_states(const Belief &b0) {
  std::queue<std::size_t> frontier;
  std::unordered_set<std::size_t> visited;

  for (const auto &p : b0) {
    if (visited.insert(p.first).second) {
      frontier.push(p.first);
    }
  }

  while (!frontier.empty()) {
    std::size_t sh = frontier.front();
    frontier.pop();

    auto it = _index_to_state.find(sh);
    if (it == _index_to_state.end())
      continue;
    const State &s = it->second;

    if (_domain.is_terminal(s))
      continue;

    auto applicable = _domain.get_applicable_actions(s).get_elements();
    std::for_each(
        ExecutionPolicy::policy, applicable.begin(), applicable.end(),
        [this, &s, &frontier, &visited](auto a) {
          std::size_t ah = typename Action::Hash()(a);

          auto next_dist =
              _domain.get_next_state_distribution(s, a).get_values();

          _execution_policy.protect([this, &a, ah] {
            if (_action_hash_to_idx.find(ah) == _action_hash_to_idx.end()) {
              _action_hash_to_idx[ah] = _actions.size();
              _actions.push_back(a);
            }
          });

          for (auto ns_item : next_dist) {
            _execution_policy.protect([this, &ns_item, &frontier, &visited] {
              std::size_t nsh = typename State::Hash()(ns_item.state());
              if (_index_to_state.find(nsh) == _index_to_state.end()) {
                _index_to_state[nsh] = ns_item.state();
              }
              if (visited.insert(nsh).second) {
                frontier.push(nsh);
              }
            });
          }
        });
  }

  _states.reserve(_index_to_state.size());
  _state_idx_to_hash.reserve(_index_to_state.size());
  for (const auto &entry : _index_to_state) {
    _state_hash_to_idx[entry.first] = _states.size();
    _state_idx_to_hash.push_back(entry.first);
    _states.push_back(entry.second);
  }

  _is_terminal_cache.resize(_states.size());
  for (std::size_t si = 0; si < _states.size(); ++si) {
    _is_terminal_cache[si] = _domain.is_terminal(_states[si]);
  }

  if (_verbose) {
    Logger::info("HSVI: enumerated " + std::to_string(_states.size()) +
                 " states and " + std::to_string(_actions.size()) + " actions");
  }
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::pre_cache_model() {
  std::size_t ns = _states.size();
  std::size_t na = _actions.size();

  _transitions.resize(
      ns, std::vector<std::vector<std::pair<double, std::size_t>>>(na));
  _obs_prob.resize(ns,
                   std::vector<std::unordered_map<std::size_t, double>>(na));
  _values.resize(ns, std::vector<double>(na, 0.0));
  _action_obs_hashes.resize(na);

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
        if (_is_terminal_cache[si])
          return;

        const State &s = _states[si];
        for (std::size_t ai = 0; ai < na; ++ai) {
          auto next_dist =
              _domain.get_next_state_distribution(s, _actions[ai]).get_values();

          double weighted_value = 0.0;

          for (auto ns_item : next_dist) {
            std::size_t nsh = typename State::Hash()(ns_item.state());
            auto ns_it = _state_hash_to_idx.find(nsh);
            if (ns_it == _state_hash_to_idx.end())
              continue;
            std::size_t ns_idx = ns_it->second;
            double prob = ns_item.probability();

            _transitions[si][ai].push_back({prob, ns_idx});

            double v = _get_value(
                _domain.get_transition_value(s, _actions[ai], _states[ns_idx]));
            weighted_value += prob * v;

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

          _values[si][ai] = weighted_value;
        }
      });

  for (std::size_t ai = 0; ai < na; ++ai) {
    _action_obs_hashes[ai].assign(action_obs_sets[ai].begin(),
                                  action_obs_sets[ai].end());
  }

  if (_verbose) {
    Logger::info("HSVI: pre-cached model with " +
                 std::to_string(_obs_objects.size()) + " observations");
  }
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::create_blind_policy_alphas() {
  std::size_t ns = _states.size();
  std::size_t na = _actions.size();

  std::vector<std::size_t> state_indices(ns);
  std::iota(state_indices.begin(), state_indices.end(), 0);

  for (std::size_t ai = 0; ai < na; ++ai) {
    AlphaVector alpha(ns, _actions[ai], _next_alpha_id++);

    for (std::size_t si = 0; si < ns; ++si) {
      if (_is_terminal_cache[si]) {
        alpha.values[si] = get_terminal_state_value(si);
        continue;
      }
      if (_discount < 1.0) {
        alpha.values[si] = _values[si][ai] / (1.0 - _discount);
      } else {
        alpha.values[si] = _values[si][ai] * _max_sample_depth;
      }
    }

    for (std::size_t iter = 0; iter < _max_vi_iterations; ++iter) {
      double max_change = 0.0;
      std::vector<double> new_values(ns);
      std::for_each(
          ExecutionPolicy::policy, state_indices.begin(), state_indices.end(),
          [this, ai, &alpha, &new_values, &max_change](std::size_t si) {
            if (_is_terminal_cache[si]) {
              new_values[si] = get_terminal_state_value(si);
              return;
            }
            double v = _values[si][ai];
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
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::initialize_alpha_bound() {
  create_blind_policy_alphas();

  if (_verbose) {
    Logger::info("HSVI: initialized " + std::to_string(_alpha_vectors.size()) +
                 " alpha-vectors");
  }
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::initialize_point_bound() {
  std::size_t ns = _states.size();
  std::size_t na = _actions.size();

  _mdp_values.resize(ns, 0.0);

  // Reward mode: upper bound = optimal MDP reward (start high)
  if (_discount < 1.0) {
    for (std::size_t si = 0; si < ns; ++si) {
      if (_is_terminal_cache[si]) {
        _mdp_values[si] = get_terminal_state_value(si);
        continue;
      }
      double best_r = -std::numeric_limits<double>::infinity();
      for (std::size_t ai = 0; ai < na; ++ai) {
        best_r = std::max(best_r, _values[si][ai]);
      }
      _mdp_values[si] = (best_r > -std::numeric_limits<double>::infinity())
                            ? best_r / (1.0 - _discount)
                            : 0.0;
    }
  }

  std::vector<std::size_t> state_indices(ns);
  std::iota(state_indices.begin(), state_indices.end(), 0);

  for (std::size_t iter = 0; iter < _max_vi_iterations; ++iter) {
    double max_change = 0.0;
    std::vector<double> new_values(ns, 0.0);

    std::for_each(ExecutionPolicy::policy, state_indices.begin(),
                  state_indices.end(),
                  [this, na, &new_values, &max_change](std::size_t si) {
                    if (_is_terminal_cache[si]) {
                      new_values[si] = get_terminal_state_value(si);
                      return;
                    }

                    double best_v = _best_init();
                    for (std::size_t ai = 0; ai < na; ++ai) {
                      double v = _values[si][ai];
                      for (const auto &tr : _transitions[si][ai]) {
                        v += _discount * tr.first * _mdp_values[tr.second];
                      }
                      best_v = _better(best_v, v);
                    }
                    if (best_v == _best_init())
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

  if (_verbose) {
    Logger::info("HSVI: MDP point bound initialized");
  }
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::solve(
    const std::vector<std::pair<State, double>> &initial_distribution) {
  _start_time = std::chrono::high_resolution_clock::now();

  _initial_belief.clear();
  for (const auto &p : initial_distribution) {
    std::size_t sh = typename State::Hash()(p.first);
    if (_index_to_state.find(sh) == _index_to_state.end()) {
      _index_to_state[sh] = p.first;
    }
    _initial_belief[sh] = p.second;
  }

  _current_belief = _initial_belief;
  _has_last_action = false;

  enumerate_states(_initial_belief);
  on_states_enumerated();
  pre_cache_model();
  on_model_cached();
  initialize_alpha_bound();
  initialize_point_bound();
  compute_depth_bound();

  if (_verbose) {
    Logger::info("HSVI: depth bound = " + std::to_string(_depth_bound));
    double ub = evaluate_upper(_initial_belief);
    double lb = evaluate_lower(_initial_belief);
    Logger::info("HSVI: initial bounds [" + std::to_string(lb) + ", " +
                 std::to_string(ub) + "], gap = " + std::to_string(ub - lb));
  }

  std::size_t iteration = 0;
  while (true) {
    double ub = evaluate_upper(_initial_belief);
    double lb = evaluate_lower(_initial_belief);
    _gap = ub - lb;

    if (_gap <= _epsilon) {
      if (_verbose)
        Logger::info("HSVI: converged at gap = " + std::to_string(_gap) +
                     " after " + std::to_string(iteration) + " iterations");
      break;
    }

    if (elapsed_ms() >= _time_budget) {
      if (_verbose)
        Logger::info("HSVI: time budget reached, gap = " +
                     std::to_string(_gap));
      break;
    }

    if (_callback(*this, _domain)) {
      if (_verbose)
        Logger::info("HSVI: interrupted by callback");
      break;
    }

    std::unordered_set<std::size_t> closed_list;
    explore(_initial_belief, 0, closed_list);

    ++iteration;

    if (_verbose && (iteration % 10 == 0)) {
      ub = evaluate_upper(_initial_belief);
      lb = evaluate_lower(_initial_belief);
      _gap = ub - lb;
      Logger::info("HSVI: iteration " + std::to_string(iteration) +
                   ", bounds [" + std::to_string(lb) + ", " +
                   std::to_string(ub) + "], gap = " + std::to_string(_gap) +
                   ", alphas = " + std::to_string(_alpha_vectors.size()) +
                   ", points = " + std::to_string(_bound_points.size()));
    }
  }
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::explore(const Belief &b, std::size_t depth,
                            std::unordered_set<std::size_t> &closed_list) {
  if (elapsed_ms() >= _time_budget)
    return;

  if (depth >= _depth_bound)
    return;

  double ub = evaluate_upper(b);
  double lb = evaluate_lower(b);
  double width = ub - lb;

  double threshold = convergence_threshold(depth);

  if (width <= threshold)
    return;

  if (_use_closed_list) {
    std::size_t bh = belief_hash(b);
    if (closed_list.count(bh) > 0)
      return;
    closed_list.insert(bh);
  }

  std::size_t na = _actions.size();

  std::vector<double> q_values(na);
  std::vector<std::size_t> action_indices(na);
  std::iota(action_indices.begin(), action_indices.end(), 0);

  std::for_each(ExecutionPolicy::policy, action_indices.begin(),
                action_indices.end(), [this, &b, &q_values](std::size_t ai) {
                  double q = 0.0;
                  for (const auto &p : b) {
                    auto it = _state_hash_to_idx.find(p.first);
                    if (it != _state_hash_to_idx.end()) {
                      q += p.second * _values[it->second][ai];
                    }
                  }

                  for (std::size_t oh : _action_obs_hashes[ai]) {
                    double obs_p = compute_obs_probability(b, ai, oh);
                    if (obs_p <= _prob_epsilon)
                      continue;
                    Belief posterior = compute_posterior(b, ai, oh);
                    if (posterior.empty())
                      continue;
                    double future = evaluate_sawtooth(posterior);
                    q += _discount * obs_p * future;
                  }

                  q_values[ai] = q;
                });

  std::size_t best_ai = 0;
  double best_q = _best_init();
  for (std::size_t ai = 0; ai < na; ++ai) {
    if (_is_better(q_values[ai], best_q)) {
      best_q = q_values[ai];
      best_ai = ai;
    }
  }

  std::size_t best_oh = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  bool found_obs = false;
  Belief best_posterior;

  for (std::size_t oh : _action_obs_hashes[best_ai]) {
    double obs_p = compute_obs_probability(b, best_ai, oh);
    if (obs_p <= _prob_epsilon)
      continue;

    Belief posterior = compute_posterior(b, best_ai, oh);
    if (posterior.empty())
      continue;

    double post_ub = evaluate_upper(posterior);
    double post_lb = evaluate_lower(posterior);
    double excess = post_ub - post_lb - threshold;

    if (excess <= 0)
      continue;

    double score = obs_p * excess;
    if (score > best_score) {
      best_score = score;
      best_oh = oh;
      best_posterior = std::move(posterior);
      found_obs = true;
    }
  }

  if (found_obs) {
    explore(best_posterior, depth + 1, closed_list);
  }

  alpha_backup(b);
  point_update(b);
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::alpha_backup(const Belief &b) {
  std::size_t ns = _states.size();
  std::size_t na = _actions.size();

  struct AlphaCandidate {
    AlphaVector alpha;
    double q_val;
  };
  std::vector<AlphaCandidate> candidates(na);

  std::vector<std::size_t> action_indices(na);
  std::iota(action_indices.begin(), action_indices.end(), 0);

  std::for_each(ExecutionPolicy::policy, action_indices.begin(),
                action_indices.end(),
                [this, ns, &b, &candidates](std::size_t ai) {
                  AlphaVector g_a(ns, _actions[ai], 0);

                  for (std::size_t si = 0; si < ns; ++si) {
                    g_a.values[si] = _values[si][ai];
                  }

                  for (std::size_t oh : _action_obs_hashes[ai]) {
                    Belief posterior = compute_posterior(b, ai, oh);
                    if (posterior.empty())
                      continue;

                    std::size_t best_idx = best_alpha_index(posterior);
                    const AlphaVector &alpha_ao = _alpha_vectors[best_idx];

                    for (std::size_t si = 0; si < ns; ++si) {
                      double contrib = 0.0;
                      for (const auto &tr : _transitions[si][ai]) {
                        auto obs_it = _obs_prob[tr.second][ai].find(oh);
                        double z = (obs_it != _obs_prob[tr.second][ai].end())
                                       ? obs_it->second
                                       : 0.0;
                        contrib += tr.first * z * alpha_ao.values[tr.second];
                      }
                      g_a.values[si] += _discount * contrib;
                    }
                  }

                  for (std::size_t si = 0; si < ns; ++si) {
                    if (_is_terminal_cache[si]) {
                      g_a.values[si] = get_terminal_state_value(si);
                    }
                  }

                  double q_val = dot_product(g_a, b);
                  candidates[ai] = {std::move(g_a), q_val};
                });

  double best_q = _best_init();
  std::size_t best_ai = 0;
  for (std::size_t ai = 0; ai < na; ++ai) {
    if (_is_better(candidates[ai].q_val, best_q)) {
      best_q = candidates[ai].q_val;
      best_ai = ai;
    }
  }

  candidates[best_ai].alpha.id = _next_alpha_id++;
  _alpha_vectors.push_back(std::move(candidates[best_ai].alpha));
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::point_update(const Belief &b) {
  std::size_t na = _actions.size();

  std::vector<double> q_values(na);
  std::vector<std::size_t> action_indices(na);
  std::iota(action_indices.begin(), action_indices.end(), 0);

  std::for_each(ExecutionPolicy::policy, action_indices.begin(),
                action_indices.end(), [this, &b, &q_values](std::size_t ai) {
                  double q = 0.0;
                  for (const auto &p : b) {
                    auto it = _state_hash_to_idx.find(p.first);
                    if (it != _state_hash_to_idx.end()) {
                      q += p.second * _values[it->second][ai];
                    }
                  }

                  for (std::size_t oh : _action_obs_hashes[ai]) {
                    double obs_p = compute_obs_probability(b, ai, oh);
                    if (obs_p <= _prob_epsilon)
                      continue;
                    Belief posterior = compute_posterior(b, ai, oh);
                    if (posterior.empty())
                      continue;
                    q += _discount * obs_p * evaluate_alpha(posterior);
                  }

                  q_values[ai] = q;
                });

  double best_v = _best_init();
  for (std::size_t ai = 0; ai < na; ++ai) {
    best_v = _better(best_v, q_values[ai]);
  }

  if (best_v == _best_init())
    return;

  double current = evaluate_sawtooth(b);
  if (_is_better(best_v, current)) {
    _bound_points.push_back({b, best_v});
  }
}

SK_HSVI_TEMPLATE_DECL
double SK_HSVI_CLASS::evaluate_alpha(const Belief &b) const {
  if (_alpha_vectors.empty())
    return 0.0;

  double best = _best_init();
  for (const auto &alpha : _alpha_vectors) {
    double val = dot_product(alpha, b);
    best = _better(best, val);
  }
  return best;
}

SK_HSVI_TEMPLATE_DECL
double SK_HSVI_CLASS::evaluate_sawtooth_corner(const Belief &b) const {
  double v = 0.0;
  for (const auto &p : b) {
    auto it = _state_hash_to_idx.find(p.first);
    if (it != _state_hash_to_idx.end()) {
      v += p.second * _mdp_values[it->second];
    }
  }
  return v;
}

SK_HSVI_TEMPLATE_DECL
double SK_HSVI_CLASS::evaluate_sawtooth(const Belief &b) const {
  double v_corner = evaluate_sawtooth_corner(b);

  for (const auto &pt : _bound_points) {
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

    double one_minus_c = 1.0 - c;
    if (one_minus_c <= _prob_epsilon) {
      v_corner = _better(v_corner, pt.value);
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
    v_corner = _better(v_corner, candidate);
  }

  return v_corner;
}

SK_HSVI_TEMPLATE_DECL
double SK_HSVI_CLASS::dot_product(const AlphaVector &alpha,
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

SK_HSVI_TEMPLATE_DECL
std::size_t SK_HSVI_CLASS::best_alpha_index(const Belief &b) const {
  std::size_t best_idx = 0;
  double best_val = _best_init();
  for (std::size_t i = 0; i < _alpha_vectors.size(); ++i) {
    double val = dot_product(_alpha_vectors[i], b);
    if (_is_better(val, best_val)) {
      best_val = val;
      best_idx = i;
    }
  }
  return best_idx;
}

SK_HSVI_TEMPLATE_DECL
typename SK_HSVI_CLASS::Belief
SK_HSVI_CLASS::compute_posterior(const Belief &b, std::size_t action_idx,
                                 std::size_t obs_hash) const {
  std::unordered_map<std::size_t, double> b_predicted;
  for (const auto &p : b) {
    auto it = _state_hash_to_idx.find(p.first);
    if (it == _state_hash_to_idx.end())
      continue;
    std::size_t si = it->second;
    for (const auto &tr : _transitions[si][action_idx]) {
      std::size_t ns_hash = _state_idx_to_hash[tr.second];
      b_predicted[ns_hash] += p.second * tr.first;
    }
  }

  Belief posterior;
  double normalizer = 0.0;

  for (const auto &p : b_predicted) {
    auto idx_it = _state_hash_to_idx.find(p.first);
    if (idx_it == _state_hash_to_idx.end())
      continue;
    std::size_t ns_idx = idx_it->second;

    auto obs_it = _obs_prob[ns_idx][action_idx].find(obs_hash);
    double z =
        (obs_it != _obs_prob[ns_idx][action_idx].end()) ? obs_it->second : 0.0;

    double val = z * p.second;
    if (val > 0.0) {
      posterior[p.first] = val;
      normalizer += val;
    }
  }

  if (normalizer > _prob_epsilon) {
    for (auto &p : posterior) {
      p.second /= normalizer;
    }
  } else {
    posterior.clear();
  }

  return posterior;
}

SK_HSVI_TEMPLATE_DECL
double SK_HSVI_CLASS::compute_obs_probability(const Belief &b,
                                              std::size_t action_idx,
                                              std::size_t obs_hash) const {
  double prob = 0.0;
  for (const auto &p : b) {
    auto idx_it = _state_hash_to_idx.find(p.first);
    if (idx_it == _state_hash_to_idx.end())
      continue;
    std::size_t si = idx_it->second;
    for (const auto &tr : _transitions[si][action_idx]) {
      auto obs_it = _obs_prob[tr.second][action_idx].find(obs_hash);
      if (obs_it != _obs_prob[tr.second][action_idx].end()) {
        prob += p.second * tr.first * obs_it->second;
      }
    }
  }
  return prob;
}

SK_HSVI_TEMPLATE_DECL
std::size_t SK_HSVI_CLASS::belief_hash(const Belief &b) const {
  std::size_t seed = 0;
  for (const auto &p : b) {
    std::size_t disc = static_cast<std::size_t>(std::ceil(p.second * 1000.0));
    seed ^= p.first * 2654435761UL + disc;
  }
  return seed;
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::update_current_belief(const Observation &obs) {
  if (!_has_last_action)
    return;

  std::size_t ah = typename Action::Hash()(_last_action);
  auto ai_it = _action_hash_to_idx.find(ah);
  if (ai_it == _action_hash_to_idx.end())
    return;
  std::size_t ai = ai_it->second;

  std::size_t oh = typename Observation::Hash()(obs);

  Belief posterior = compute_posterior(_current_belief, ai, oh);
  if (!posterior.empty()) {
    _current_belief = std::move(posterior);
  }
}

SK_HSVI_TEMPLATE_DECL
const typename SK_HSVI_CLASS::Action &
SK_HSVI_CLASS::get_best_action(const Observation &obs) {
  update_current_belief(obs);
  const Action &a = get_best_action_from_belief(_current_belief);
  _last_action = a;
  _has_last_action = true;
  return a;
}

SK_HSVI_TEMPLATE_DECL
typename SK_HSVI_CLASS::Value
SK_HSVI_CLASS::get_best_value(const Observation &obs) {
  update_current_belief(obs);
  return get_best_value_from_belief(_current_belief);
}

SK_HSVI_TEMPLATE_DECL
bool SK_HSVI_CLASS::is_solution_defined_for(const Observation &obs) {
  return !_alpha_vectors.empty();
}

SK_HSVI_TEMPLATE_DECL
void SK_HSVI_CLASS::reset_belief() {
  _current_belief = _initial_belief;
  _has_last_action = false;
}

SK_HSVI_TEMPLATE_DECL
const typename SK_HSVI_CLASS::Action &
SK_HSVI_CLASS::get_best_action_from_belief(const Belief &b) const {
  std::size_t idx = best_alpha_index(b);
  return _alpha_vectors[idx].action;
}

SK_HSVI_TEMPLATE_DECL
typename SK_HSVI_CLASS::Value
SK_HSVI_CLASS::get_best_value_from_belief(const Belief &b) const {
  double v = evaluate_alpha(b);
  Value val;
  make_value_obj(v, val);
  return val;
}

SK_HSVI_TEMPLATE_DECL
bool SK_HSVI_CLASS::is_solution_defined_for_from_belief(const Belief &b) const {
  return !_alpha_vectors.empty();
}

SK_HSVI_TEMPLATE_DECL
std::size_t SK_HSVI_CLASS::get_nb_alpha_vectors() const {
  return _alpha_vectors.size();
}

SK_HSVI_TEMPLATE_DECL
std::size_t SK_HSVI_CLASS::get_nb_bound_points() const {
  return _bound_points.size();
}

SK_HSVI_TEMPLATE_DECL
std::size_t SK_HSVI_CLASS::get_solving_time() const { return elapsed_ms(); }

SK_HSVI_TEMPLATE_DECL
double SK_HSVI_CLASS::get_gap() const { return _gap; }

SK_HSVI_TEMPLATE_DECL
std::size_t SK_HSVI_CLASS::get_state_index(const State &s) {
  std::size_t sh = typename State::Hash()(s);
  auto it = _state_hash_to_idx.find(sh);
  if (it != _state_hash_to_idx.end())
    return it->second;
  _state_hash_to_idx[sh] = _states.size();
  _index_to_state[sh] = s;
  _states.push_back(s);
  return _states.size() - 1;
}

SK_HSVI_TEMPLATE_DECL
const std::unordered_map<std::size_t, typename SK_HSVI_CLASS::State> &
SK_HSVI_CLASS::get_index_to_state() const {
  return _index_to_state;
}

SK_HSVI_TEMPLATE_DECL
std::size_t SK_HSVI_CLASS::elapsed_ms() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now() - _start_time)
      .count();
}

// =============================================================================
// GoalHSVISolver implementation (cost minimization with goals)
// =============================================================================

#define SK_GOAL_HSVI_TEMPLATE_DECL                                             \
  template <typename Tdomain, typename Texecution_policy>

#define SK_GOAL_HSVI_CLASS GoalHSVISolver<Tdomain, Texecution_policy>

SK_GOAL_HSVI_TEMPLATE_DECL
SK_GOAL_HSVI_CLASS::GoalHSVISolver(
    Domain &domain, const GoalCheckerFunctor &goal_checker, double epsilon,
    double discount, std::size_t time_budget, std::size_t max_sample_depth,
    bool use_closed_list, double depth_bound_eta, std::size_t max_vi_iterations,
    double vi_convergence_factor, double prob_epsilon,
    const CallbackFunctor &callback, bool verbose,
    std::optional<double> dead_end_cost)
    : Base(domain, epsilon, discount, time_budget, max_sample_depth,
           use_closed_list, depth_bound_eta, max_vi_iterations,
           vi_convergence_factor, prob_epsilon, callback, verbose),
      _goal_checker(goal_checker), _user_dead_end_cost(dead_end_cost) {}

SK_GOAL_HSVI_TEMPLATE_DECL
void SK_GOAL_HSVI_CLASS::clear() {
  Base::clear();
  _is_goal_cache.clear();
  _dead_end_cost = 0.0;
}

SK_GOAL_HSVI_TEMPLATE_DECL
void SK_GOAL_HSVI_CLASS::on_states_enumerated() {
  _is_goal_cache.resize(this->_states.size(), false);
  for (std::size_t si = 0; si < this->_states.size(); ++si) {
    _is_goal_cache[si] = _goal_checker(this->_domain, this->_states[si]);
  }

  if (this->_verbose) {
    std::size_t n_goals =
        std::count(_is_goal_cache.begin(), _is_goal_cache.end(), true);
    std::size_t n_terms = std::count(this->_is_terminal_cache.begin(),
                                     this->_is_terminal_cache.end(), true);
    Logger::info("GoalHSVI: " + std::to_string(n_goals) + " goal states, " +
                 std::to_string(n_terms) + " terminal states");
  }
}

SK_GOAL_HSVI_TEMPLATE_DECL
void SK_GOAL_HSVI_CLASS::on_model_cached() {
  if (_user_dead_end_cost.has_value()) {
    _dead_end_cost = _user_dead_end_cost.value();
  } else {
    double worst = 0.0;
    for (std::size_t si = 0; si < this->_states.size(); ++si) {
      for (std::size_t ai = 0; ai < this->_actions.size(); ++ai) {
        worst = std::max(worst, this->_values[si][ai]);
      }
    }
    _dead_end_cost = (this->_discount < 1.0) ? worst / (1.0 - this->_discount)
                                             : worst * this->_max_sample_depth;
  }

  if (this->_verbose) {
    Logger::info("GoalHSVI: dead-end cost = " + std::to_string(_dead_end_cost));
  }
}

SK_GOAL_HSVI_TEMPLATE_DECL
double SK_GOAL_HSVI_CLASS::get_terminal_state_value(std::size_t si) const {
  return _is_goal_cache[si] ? 0.0 : _dead_end_cost;
}

SK_GOAL_HSVI_TEMPLATE_DECL
void SK_GOAL_HSVI_CLASS::initialize_alpha_bound() {
  std::size_t ns = this->_states.size();
  std::size_t na = this->_actions.size();

  std::vector<std::size_t> state_indices(ns);
  std::iota(state_indices.begin(), state_indices.end(), 0);

  // Uniform policy upper bound alpha-vector
  std::vector<double> unif_values(ns, 0.0);

  for (std::size_t si = 0; si < ns; ++si) {
    if (this->_is_terminal_cache[si]) {
      unif_values[si] = get_terminal_state_value(si);
    } else {
      unif_values[si] = _dead_end_cost;
    }
  }

  for (std::size_t iter = 0; iter < this->_max_vi_iterations; ++iter) {
    double max_change = 0.0;
    std::vector<double> new_values(ns, 0.0);
    std::for_each(
        ExecutionPolicy::policy, state_indices.begin(), state_indices.end(),
        [this, na, &unif_values, &new_values, &max_change](std::size_t si) {
          if (this->_is_terminal_cache[si]) {
            new_values[si] = get_terminal_state_value(si);
            return;
          }
          double sum = 0.0;
          for (std::size_t ai = 0; ai < na; ++ai) {
            double v = this->_values[si][ai];
            for (const auto &tr : this->_transitions[si][ai]) {
              v += this->_discount * tr.first * unif_values[tr.second];
            }
            sum += v;
          }
          new_values[si] = sum / static_cast<double>(na);
          double change = std::abs(new_values[si] - unif_values[si]);
          this->_execution_policy.protect([&max_change, change] {
            max_change = std::max(max_change, change);
          });
        });
    unif_values = std::move(new_values);
    if (max_change < this->_epsilon * this->_vi_convergence_factor)
      break;
  }

  typename Base::AlphaVector alpha(ns, this->_actions[0],
                                   this->_next_alpha_id++);
  alpha.values = unif_values;
  this->_alpha_vectors.push_back(std::move(alpha));

  // Blind policy alpha-vectors (shared)
  this->create_blind_policy_alphas();

  if (this->_verbose) {
    Logger::info("GoalHSVI: initialized " +
                 std::to_string(this->_alpha_vectors.size()) +
                 " alpha-vectors");
  }
}

SK_GOAL_HSVI_TEMPLATE_DECL
void SK_GOAL_HSVI_CLASS::initialize_point_bound() {
  std::size_t ns = this->_states.size();
  std::size_t na = this->_actions.size();

  this->_mdp_values.resize(ns, 0.0);

  // Cost mode: lower bound starts at 0 (optimistic for cost minimization)
  // Terminal states get their proper value
  for (std::size_t si = 0; si < ns; ++si) {
    if (this->_is_terminal_cache[si]) {
      this->_mdp_values[si] = get_terminal_state_value(si);
    }
  }

  std::vector<std::size_t> state_indices(ns);
  std::iota(state_indices.begin(), state_indices.end(), 0);

  for (std::size_t iter = 0; iter < this->_max_vi_iterations; ++iter) {
    double max_change = 0.0;
    std::vector<double> new_values(ns, 0.0);

    std::for_each(
        ExecutionPolicy::policy, state_indices.begin(), state_indices.end(),
        [this, na, &new_values, &max_change](std::size_t si) {
          if (this->_is_terminal_cache[si]) {
            new_values[si] = get_terminal_state_value(si);
            return;
          }

          double best_v = this->_best_init();
          for (std::size_t ai = 0; ai < na; ++ai) {
            double v = this->_values[si][ai];
            for (const auto &tr : this->_transitions[si][ai]) {
              v += this->_discount * tr.first * this->_mdp_values[tr.second];
            }
            best_v = this->_better(best_v, v);
          }
          if (best_v == this->_best_init())
            best_v = 0.0;

          new_values[si] = best_v;
          double change = std::abs(best_v - this->_mdp_values[si]);
          this->_execution_policy.protect([&max_change, change] {
            max_change = std::max(max_change, change);
          });
        });

    this->_mdp_values = std::move(new_values);
    if (max_change < this->_epsilon * this->_vi_convergence_factor)
      break;
  }

  if (this->_verbose) {
    Logger::info("GoalHSVI: MDP point bound initialized");
  }
}

SK_GOAL_HSVI_TEMPLATE_DECL
void SK_GOAL_HSVI_CLASS::compute_depth_bound() {
  if (this->_discount >= 1.0) {
    double c_max = 0.0;
    double c_min = std::numeric_limits<double>::infinity();
    for (std::size_t si = 0; si < this->_states.size(); ++si) {
      if (this->_is_terminal_cache[si])
        continue;
      for (std::size_t ai = 0; ai < this->_actions.size(); ++ai) {
        if (this->_values[si][ai] > 0) {
          c_max = std::max(c_max, this->_values[si][ai]);
          c_min = std::min(c_min, this->_values[si][ai]);
        }
      }
    }
    if (c_min > 0 && c_min < std::numeric_limits<double>::infinity() &&
        this->_epsilon > 0) {
      double eta = this->_depth_bound_eta;
      double ratio = c_max / c_min;
      double term =
          (c_max - eta * this->_epsilon) / ((1.0 - eta) * this->_epsilon);
      this->_depth_bound = static_cast<std::size_t>(std::ceil(ratio * term));
      this->_depth_bound =
          std::min(this->_depth_bound, this->_max_sample_depth);
    } else {
      this->_depth_bound = this->_max_sample_depth;
    }
  } else {
    this->_depth_bound = this->_max_sample_depth;
  }
}

} // namespace skdecide

#endif // SKDECIDE_HSVI_IMPL_HH
