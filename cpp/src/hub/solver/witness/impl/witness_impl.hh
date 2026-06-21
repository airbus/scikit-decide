/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Exact POMDP solver implementing the Witness algorithm from:
 * Littman, "The Witness Algorithm: Solving Partially Observable
 * Markov Decision Processes", Brown University TR, 1994.
 */
#ifndef SKDECIDE_WITNESS_IMPL_HH
#define SKDECIDE_WITNESS_IMPL_HH

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#include "Highs.h"
#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

#define SK_WITNESS_TEMPLATE_DECL                                               \
  template <typename Tdomain, typename Texecution_policy>

#define SK_WITNESS_CLASS WitnessSolver<Tdomain, Texecution_policy>

SK_WITNESS_TEMPLATE_DECL
SK_WITNESS_CLASS::WitnessSolver(Domain &domain, double epsilon, double discount,
                                std::size_t max_iterations, double lp_infinity,
                                double lp_tolerance,
                                const TerminalValueFunctor &terminal_value,
                                const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _epsilon(epsilon), _discount(discount),
      _max_iterations(max_iterations), _lp_infinity(lp_infinity),
      _lp_tolerance(lp_tolerance), _terminal_value(terminal_value),
      _callback(callback), _verbose(verbose), _has_solution(false),
      _nb_iterations(0), _solving_time(0) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm Witness");
  }
}

SK_WITNESS_TEMPLATE_DECL
void SK_WITNESS_CLASS::clear() {
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
  _obs_hash_to_pos.clear();
  _alpha_vectors.clear();
  _current_belief.clear();
  _last_action.reset();
  _has_solution = false;
  _nb_iterations = 0;
  _solving_time = 0;
}

// --- State indexing ---

SK_WITNESS_TEMPLATE_DECL
std::size_t SK_WITNESS_CLASS::get_state_index(const State &s) {
  std::size_t h = typename State::Hash()(s);
  if (_index_to_state.find(h) == _index_to_state.end()) {
    _index_to_state[h] = s;
  }
  return h;
}

SK_WITNESS_TEMPLATE_DECL
const std::unordered_map<std::size_t, typename SK_WITNESS_CLASS::State> &
SK_WITNESS_CLASS::get_index_to_state() const {
  return _index_to_state;
}

SK_WITNESS_TEMPLATE_DECL
const std::unordered_map<std::size_t, std::size_t> &
SK_WITNESS_CLASS::get_state_hash_to_idx() const {
  return _state_hash_to_idx;
}

SK_WITNESS_TEMPLATE_DECL
const std::vector<typename SK_WITNESS_CLASS::State> &
SK_WITNESS_CLASS::get_states() const {
  return _states;
}

SK_WITNESS_TEMPLATE_DECL
std::size_t SK_WITNESS_CLASS::elapsed_ms() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now() - _start_time)
      .count();
}

// --- State enumeration (BFS from belief support) ---

SK_WITNESS_TEMPLATE_DECL
void SK_WITNESS_CLASS::enumerate_states(const Belief &b0) {
  if (_verbose)
    Logger::debug("Witness: enumerating reachable states");

  std::queue<std::size_t> frontier;
  std::unordered_set<std::size_t> visited;

  for (const auto &p : b0) {
    if (visited.insert(p.first).second) {
      frontier.push(p.first);
    }
  }

  std::unordered_set<std::size_t> action_hashes_seen;

  while (!frontier.empty()) {
    std::size_t sh = frontier.front();
    frontier.pop();

    const State &s = _index_to_state.at(sh);

    if (_domain.is_terminal(s))
      continue;

    auto applicable = _domain.get_applicable_actions(s).get_elements();
    for (auto a : applicable) {
      std::size_t ah = typename Action::Hash()(a);

      if (action_hashes_seen.insert(ah).second) {
        _actions.push_back(a);
        _action_hash_to_idx[ah] = _actions.size() - 1;
      }

      auto next_dist = _domain.get_next_state_distribution(s, a).get_values();
      for (auto ns : next_dist) {
        std::size_t nsh = get_state_index(ns.state());
        if (visited.insert(nsh).second) {
          frontier.push(nsh);
        }
      }
    }
  }

  _states.clear();
  _state_hash_to_idx.clear();
  for (const auto &p : _index_to_state) {
    _state_hash_to_idx[p.first] = _states.size();
    _states.push_back(p.second);
  }

  if (_verbose)
    Logger::debug("Witness: found " + std::to_string(_states.size()) +
                  " states and " + std::to_string(_actions.size()) +
                  " actions");
}

// --- Model pre-caching ---

SK_WITNESS_TEMPLATE_DECL
void SK_WITNESS_CLASS::pre_cache_model() {
  std::size_t ns = _states.size();
  std::size_t na = _actions.size();

  _transitions.resize(ns);
  _obs_prob.resize(ns);
  _rewards.resize(ns, std::vector<double>(na, 0.0));
  _action_obs_hashes.resize(na);

  std::vector<std::unordered_set<std::size_t>> action_obs_sets(na);

  for (std::size_t i = 0; i < ns; ++i) {
    _transitions[i].resize(na);
    _obs_prob[i].resize(na);
  }

  for (std::size_t si = 0; si < ns; ++si) {
    const State &s = _states[si];
    if (_domain.is_terminal(s)) {
      // Terminal state: set reward to terminal value for all actions
      for (std::size_t ai = 0; ai < na; ++ai) {
        _rewards[si][ai] = _terminal_value(s).reward();
      }
      continue;
    }

    for (std::size_t ai = 0; ai < na; ++ai) {
      auto next_dist =
          _domain.get_next_state_distribution(s, _actions[ai]).get_values();

      if (next_dist.empty()) {
        // Action has no transitions: use terminal value
        _rewards[si][ai] = _terminal_value(s).reward();
        continue;
      }

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
            _domain.get_observation_distribution(_states[ns_idx], _actions[ai])
                .get_values();

        for (auto od : obs_dist) {
          std::size_t oh = typename Observation::Hash()(od.observation());
          _obs_prob[ns_idx][ai][oh] = od.probability();
          if (_obs_objects.find(oh) == _obs_objects.end()) {
            _obs_objects[oh] = od.observation();
          }
          action_obs_sets[ai].insert(oh);
        }
      }

      _rewards[si][ai] = weighted_reward;
    }
  }

  for (std::size_t ai = 0; ai < na; ++ai) {
    _action_obs_hashes[ai].assign(action_obs_sets[ai].begin(),
                                  action_obs_sets[ai].end());
  }

  _obs_hash_to_pos.resize(na);
  for (std::size_t ai = 0; ai < na; ++ai) {
    for (std::size_t p = 0; p < _action_obs_hashes[ai].size(); ++p) {
      _obs_hash_to_pos[ai][_action_obs_hashes[ai][p]] = p;
    }
  }

  if (_verbose)
    Logger::debug("Witness: model pre-cached");
}

// --- Alpha-vector operations ---

SK_WITNESS_TEMPLATE_DECL
double SK_WITNESS_CLASS::dot_product(const AlphaVector &alpha,
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

SK_WITNESS_TEMPLATE_DECL
double SK_WITNESS_CLASS::dot_product_dense(const std::vector<double> &alpha,
                                           const std::vector<double> &b) const {
  double result = 0.0;
  for (std::size_t i = 0; i < alpha.size(); ++i) {
    result += alpha[i] * b[i];
  }
  return result;
}

SK_WITNESS_TEMPLATE_DECL
double SK_WITNESS_CLASS::evaluate(const Belief &b) const {
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

SK_WITNESS_TEMPLATE_DECL
std::size_t SK_WITNESS_CLASS::best_alpha_index(const Belief &b) const {
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

// --- Back projection ---

SK_WITNESS_TEMPLATE_DECL
std::vector<double>
SK_WITNESS_CLASS::compute_back(const std::vector<double> &alpha_values,
                               std::size_t action_idx,
                               std::size_t obs_hash) const {
  std::size_t ns = _states.size();
  std::vector<double> result(ns, 0.0);
  for (std::size_t si = 0; si < ns; ++si) {
    double val = 0.0;
    for (const auto &tr : _transitions[si][action_idx]) {
      auto obs_it = _obs_prob[tr.second][action_idx].find(obs_hash);
      double z = (obs_it != _obs_prob[tr.second][action_idx].end())
                     ? obs_it->second
                     : 0.0;
      val += alpha_values[tr.second] * tr.first * z;
    }
    result[si] = val;
  }
  return result;
}

SK_WITNESS_TEMPLATE_DECL
std::vector<std::vector<std::vector<double>>>
SK_WITNESS_CLASS::precompute_back_vectors(
    const std::vector<AlphaVector> &v_prev, std::size_t action_idx) const {
  std::size_t num_alphas = v_prev.size();
  std::size_t num_obs = _action_obs_hashes[action_idx].size();

  std::vector<std::vector<std::vector<double>>> back_vecs(
      num_alphas, std::vector<std::vector<double>>(num_obs));

  for (std::size_t ai = 0; ai < num_alphas; ++ai) {
    for (std::size_t op = 0; op < num_obs; ++op) {
      back_vecs[ai][op] = compute_back(v_prev[ai].values, action_idx,
                                       _action_obs_hashes[action_idx][op]);
    }
  }
  return back_vecs;
}

// --- besttree: construct optimal alpha-vector at belief b for action a ---

SK_WITNESS_TEMPLATE_DECL
typename SK_WITNESS_CLASS::AlphaVector SK_WITNESS_CLASS::besttree(
    const std::vector<double> &b_dense, std::size_t action_idx,
    const std::vector<AlphaVector> &v_prev,
    const std::vector<std::vector<std::vector<double>>> &back_vecs) const {

  std::size_t ns = _states.size();
  std::size_t num_obs = _action_obs_hashes[action_idx].size();
  std::size_t num_alphas = v_prev.size();

  AlphaVector result(ns, action_idx);
  result.obs_choices.resize(num_obs);

  // For each observation, find the V_prev alpha that maximizes b · back(α', a,
  // o)
  for (std::size_t op = 0; op < num_obs; ++op) {
    double best_val = -std::numeric_limits<double>::infinity();
    std::size_t best_idx = 0;

    for (std::size_t ai = 0; ai < num_alphas; ++ai) {
      double val = dot_product_dense(b_dense, back_vecs[ai][op]);
      if (val > best_val + _lp_tolerance * 1e-2) {
        best_val = val;
        best_idx = ai;
      } else if (std::abs(val - best_val) <= _lp_tolerance * 1e-2 &&
                 ai < best_idx) {
        best_idx = ai;
      }
    }
    result.obs_choices[op] = best_idx;
  }

  // Construct the new alpha-vector:
  // new[s] = R[s,a] + gamma * sum_o back(choice_o, a, o)[s]
  for (std::size_t si = 0; si < ns; ++si) {
    double val = _rewards[si][action_idx];
    for (std::size_t op = 0; op < num_obs; ++op) {
      val += _discount * back_vecs[result.obs_choices[op]][op][si];
    }
    result.values[si] = val;
  }

  return result;
}

// --- findb: find witness belief via LP ---

SK_WITNESS_TEMPLATE_DECL
std::vector<double> SK_WITNESS_CLASS::findb(
    std::size_t action_idx, const std::vector<AlphaVector> &v_prev,
    const std::vector<AlphaVector> &q_hat,
    const std::vector<std::vector<std::vector<double>>> &back_vecs,
    std::unordered_set<std::vector<std::size_t>, VectorHash<std::size_t>>
        &checked_candidates) const {

  std::size_t ns = _states.size();
  std::size_t num_obs = _action_obs_hashes[action_idx].size();
  std::size_t num_alphas = v_prev.size();
  std::size_t nq = q_hat.size();

  // Build a reusable LP skeleton: variables b[0..ns-1], simplex row,
  // and nq dominance rows whose coefficients we update per candidate.
  Highs highs;
  highs.setOptionValue("output_flag", false);

  for (std::size_t s = 0; s < ns; ++s) {
    highs.addVar(0.0, 1.0);
  }
  highs.changeObjectiveSense(ObjSense::kMaximize);

  // Row 0: simplex constraint
  {
    std::vector<HighsInt> cols(ns);
    std::vector<double> vals(ns, 1.0);
    std::iota(cols.begin(), cols.end(), 0);
    highs.addRow(1.0, 1.0, static_cast<HighsInt>(ns), cols.data(), vals.data());
  }

  // Rows 1..nq: dominance constraints (dense, all ns columns)
  std::vector<HighsInt> dom_rows(nq);
  {
    std::vector<HighsInt> cols(ns);
    std::vector<double> vals(ns, 0.0);
    std::iota(cols.begin(), cols.end(), 0);
    for (std::size_t qj = 0; qj < nq; ++qj) {
      highs.addRow(0.0, _lp_infinity, static_cast<HighsInt>(ns), cols.data(),
                   vals.data());
      dom_rows[qj] = static_cast<HighsInt>(qj + 1);
    }
  }

  // Search modifications of ALL alphas in Q_hat
  for (std::size_t qi = 0; qi < nq; ++qi) {
    const AlphaVector &region_q = q_hat[qi];

    for (std::size_t alpha_idx = 0; alpha_idx < num_alphas; ++alpha_idx) {
      for (std::size_t op = 0; op < num_obs; ++op) {
        std::size_t current_choice = region_q.obs_choices[op];

        if (alpha_idx == current_choice)
          continue;

        std::vector<std::size_t> new_choices = region_q.obs_choices;
        new_choices[op] = alpha_idx;
        if (!checked_candidates.insert(new_choices).second)
          continue;

        // beta[s] = back(alpha', a, o)[s] - back(current_choice, a, o)[s]
        // sigma[s] = region_q[s] + gamma * beta[s]
        std::vector<double> beta(ns);
        std::vector<double> sigma(ns);
        bool all_zero = true;
        for (std::size_t s = 0; s < ns; ++s) {
          beta[s] =
              back_vecs[alpha_idx][op][s] - back_vecs[current_choice][op][s];
          if (std::abs(beta[s]) > _lp_tolerance * 1e-2)
            all_zero = false;
          sigma[s] = region_q.values[s] + _discount * beta[s];
        }

        if (all_zero)
          continue;

        // Update objective: max beta · b
        for (std::size_t s = 0; s < ns; ++s) {
          highs.changeColCost(static_cast<HighsInt>(s), beta[s]);
        }

        // Update dominance rows: (sigma - q') · b >= 0
        for (std::size_t qj = 0; qj < nq; ++qj) {
          for (std::size_t s = 0; s < ns; ++s) {
            highs.changeCoeff(dom_rows[qj], static_cast<HighsInt>(s),
                              sigma[s] - q_hat[qj].values[s]);
          }
        }

        highs.run();

        if (highs.getModelStatus() == HighsModelStatus::kOptimal) {
          const auto &sol = highs.getSolution().col_value;
          double obj = 0.0;
          for (std::size_t s = 0; s < ns; ++s) {
            obj += beta[s] * sol[s];
          }
          if (obj > _lp_tolerance) {
            return std::vector<double>(sol.begin(), sol.begin() + ns);
          }
        }
      }
    }
  }

  return {};
}

// --- witness: find all non-dominated alpha-vectors for one action ---

SK_WITNESS_TEMPLATE_DECL
std::vector<typename SK_WITNESS_CLASS::AlphaVector>
SK_WITNESS_CLASS::witness_action(const std::vector<AlphaVector> &v_prev,
                                 std::size_t action_idx) const {
  std::size_t ns = _states.size();

  auto back_vecs = precompute_back_vectors(v_prev, action_idx);

  // Start with corner belief e_0
  std::vector<double> b_dense(ns, 0.0);
  b_dense[0] = 1.0;

  std::vector<AlphaVector> q_hat;
  q_hat.push_back(besttree(b_dense, action_idx, v_prev, back_vecs));

  // Track ALL candidate choice vectors checked by findb to avoid re-checking
  std::unordered_set<std::vector<std::size_t>, VectorHash<std::size_t>>
      checked_candidates;
  checked_candidates.insert(q_hat.back().obs_choices);

  auto b = findb(action_idx, v_prev, q_hat, back_vecs, checked_candidates);
  while (!b.empty()) {
    auto new_alpha = besttree(b, action_idx, v_prev, back_vecs);
    // besttree might return choices already in Q_hat (different from the
    // candidate that triggered findb). Always add unique trees.
    bool unique = true;
    for (const auto &q : q_hat) {
      if (q.obs_choices == new_alpha.obs_choices) {
        unique = false;
        break;
      }
    }
    if (unique) {
      checked_candidates.insert(new_alpha.obs_choices);
      q_hat.push_back(std::move(new_alpha));
    }
    b = findb(action_idx, v_prev, q_hat, back_vecs, checked_candidates);
  }

  if (_verbose)
    Logger::debug("Witness: action " + std::to_string(action_idx) +
                  " produced " + std::to_string(q_hat.size()) +
                  " alpha-vectors");

  return q_hat;
}

// --- purge: remove dominated alpha-vectors via Monahan LP ---

SK_WITNESS_TEMPLATE_DECL
std::vector<typename SK_WITNESS_CLASS::AlphaVector>
SK_WITNESS_CLASS::purge(const std::vector<AlphaVector> &v) const {
  if (v.size() <= 1)
    return v;

  std::size_t ns = _states.size();
  std::size_t nv = v.size();
  std::vector<AlphaVector> kept;

  Highs highs;
  highs.setOptionValue("output_flag", false);

  // Variables: b[0..ns-1], delta
  for (std::size_t s = 0; s < ns; ++s) {
    highs.addVar(0.0, 1.0);
  }
  highs.addVar(-_lp_infinity, _lp_infinity);

  // Objective: max delta
  highs.changeColCost(static_cast<HighsInt>(ns), 1.0);
  highs.changeObjectiveSense(ObjSense::kMaximize);

  // Row 0: simplex constraint sum b[s] = 1
  {
    std::vector<HighsInt> cols(ns);
    std::vector<double> vals(ns, 1.0);
    std::iota(cols.begin(), cols.end(), 0);
    highs.addRow(1.0, 1.0, static_cast<HighsInt>(ns), cols.data(), vals.data());
  }

  // Rows 1..nv-1: dominance constraints, initially for candidate i=0
  std::vector<HighsInt> dom_rows(nv - 1);
  for (std::size_t k = 0; k < nv - 1; ++k) {
    std::size_t j = k + 1;
    // Add a dense row with all belief vars + delta
    std::vector<HighsInt> cols(ns + 1);
    std::vector<double> vals(ns + 1);
    std::iota(cols.begin(), cols.end(), 0);
    for (std::size_t s = 0; s < ns; ++s) {
      vals[s] = v[0].values[s] - v[j].values[s];
    }
    vals[ns] = -1.0;
    highs.addRow(0.0, _lp_infinity, static_cast<HighsInt>(ns + 1), cols.data(),
                 vals.data());
    dom_rows[k] = static_cast<HighsInt>(k + 1);
  }

  for (std::size_t i = 0; i < nv; ++i) {
    // Update dominance row coefficients for candidate i
    std::size_t k = 0;
    for (std::size_t j = 0; j < nv; ++j) {
      if (j == i)
        continue;
      for (std::size_t s = 0; s < ns; ++s) {
        highs.changeCoeff(dom_rows[k], static_cast<HighsInt>(s),
                          v[i].values[s] - v[j].values[s]);
      }
      ++k;
    }

    highs.run();

    bool dominated = true;
    if (highs.getModelStatus() == HighsModelStatus::kOptimal) {
      double delta = highs.getSolution().col_value[ns];
      if (delta > -_lp_tolerance) {
        dominated = false;
      }
    } else if (highs.getModelStatus() == HighsModelStatus::kUnbounded) {
      dominated = false;
    }

    if (!dominated) {
      kept.push_back(v[i]);
    }
  }

  if (_verbose)
    Logger::debug("Witness: purge reduced from " + std::to_string(v.size()) +
                  " to " + std::to_string(kept.size()) + " alpha-vectors");

  return kept;
}

// --- Convergence check ---

SK_WITNESS_TEMPLATE_DECL
bool SK_WITNESS_CLASS::check_convergence(
    const std::vector<AlphaVector> &v_prev,
    const std::vector<AlphaVector> &v_next) const {
  std::size_t ns = _states.size();
  double max_diff = 0.0;

  for (std::size_t si = 0; si < ns; ++si) {
    double max_prev = -std::numeric_limits<double>::infinity();
    for (const auto &alpha : v_prev) {
      max_prev = std::max(max_prev, alpha.values[si]);
    }

    double max_next = -std::numeric_limits<double>::infinity();
    for (const auto &alpha : v_next) {
      max_next = std::max(max_next, alpha.values[si]);
    }

    max_diff = std::max(max_diff, std::abs(max_next - max_prev));
  }

  if (_verbose)
    Logger::debug("Witness: max corner-belief change = " +
                  std::to_string(max_diff));

  return max_diff < _epsilon;
}

// --- Belief operations ---

SK_WITNESS_TEMPLATE_DECL
typename SK_WITNESS_CLASS::Belief
SK_WITNESS_CLASS::compute_posterior(const Belief &b, std::size_t action_idx,
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

  if (normalizer > 0.0) {
    for (auto &p : posterior) {
      p.second /= normalizer;
    }
  }

  return posterior;
}

SK_WITNESS_TEMPLATE_DECL
void SK_WITNESS_CLASS::update_current_belief(const Observation &obs) {
  if (!_last_action)
    return;

  std::size_t ah = typename Action::Hash()(*_last_action);
  auto ait = _action_hash_to_idx.find(ah);
  if (ait == _action_hash_to_idx.end())
    return;

  std::size_t obs_hash = typename Observation::Hash()(obs);
  _current_belief = compute_posterior(_current_belief, ait->second, obs_hash);
}

// --- Main solve loop ---

SK_WITNESS_TEMPLATE_DECL
void SK_WITNESS_CLASS::solve(
    const std::vector<std::pair<State, double>> &initial_distribution) {
  _start_time = std::chrono::high_resolution_clock::now();
  clear();

  // Build initial belief
  Belief b0;
  for (const auto &p : initial_distribution) {
    std::size_t sh = get_state_index(p.first);
    b0[sh] = p.second;
  }

  enumerate_states(b0);
  pre_cache_model();

  std::size_t ns = _states.size();
  std::size_t na = _actions.size();

  if (ns == 0 || na == 0) {
    Logger::warn("Witness: no states or actions found");
    _solving_time = elapsed_ms();
    return;
  }

  // Initialize V_prev with one zero-vector per action
  std::vector<AlphaVector> v_prev;
  {
    AlphaVector zero_vec(ns, 0);
    v_prev.push_back(zero_vec);
  }

  for (std::size_t iter = 0; iter < _max_iterations; ++iter) {
    if (_verbose)
      Logger::debug("Witness: iteration " + std::to_string(iter) +
                    ", V_prev has " + std::to_string(v_prev.size()) +
                    " alpha-vectors");

    std::vector<AlphaVector> v_next;

    for (std::size_t ai = 0; ai < na; ++ai) {
      auto q_a = witness_action(v_prev, ai);
      v_next.insert(v_next.end(), std::make_move_iterator(q_a.begin()),
                    std::make_move_iterator(q_a.end()));
    }

    v_next = purge(v_next);

    _nb_iterations = iter + 1;

    if (check_convergence(v_prev, v_next)) {
      if (_verbose)
        Logger::debug("Witness: converged after " +
                      std::to_string(_nb_iterations) + " iterations");
      v_prev = std::move(v_next);
      break;
    }

    if (_callback(*this, _domain)) {
      if (_verbose)
        Logger::debug("Witness: stopped by callback");
      v_prev = std::move(v_next);
      break;
    }

    v_prev = std::move(v_next);
  }

  _alpha_vectors = std::move(v_prev);
  _has_solution = !_alpha_vectors.empty();
  _initial_belief = b0;
  _current_belief = b0;
  _last_action.reset();
  _solving_time = elapsed_ms();

  if (_verbose)
    Logger::debug("Witness: done. " + std::to_string(_alpha_vectors.size()) +
                  " alpha-vectors, " + std::to_string(_solving_time) + " ms");
}

// --- Observation-based policy interface ---

SK_WITNESS_TEMPLATE_DECL
const typename SK_WITNESS_CLASS::Action &
SK_WITNESS_CLASS::get_best_action(const Observation &obs) {
  update_current_belief(obs);
  const Action &a = get_best_action_from_belief(_current_belief);
  _last_action = std::make_unique<Action>(a);
  return *_last_action;
}

SK_WITNESS_TEMPLATE_DECL
typename SK_WITNESS_CLASS::Value
SK_WITNESS_CLASS::get_best_value(const Observation &obs) {
  update_current_belief(obs);
  return get_best_value_from_belief(_current_belief);
}

SK_WITNESS_TEMPLATE_DECL
bool SK_WITNESS_CLASS::is_solution_defined_for(const Observation &obs) {
  return _has_solution;
}

SK_WITNESS_TEMPLATE_DECL
void SK_WITNESS_CLASS::reset_belief() {
  _current_belief = _initial_belief;
  _last_action.reset();
}

// --- Belief-based policy interface ---

SK_WITNESS_TEMPLATE_DECL
const typename SK_WITNESS_CLASS::Action &
SK_WITNESS_CLASS::get_best_action_from_belief(const Belief &b) {
  if (_alpha_vectors.empty())
    throw std::runtime_error("Witness: no solution available");

  double best_val = -std::numeric_limits<double>::infinity();
  std::size_t best_idx = 0;
  for (std::size_t i = 0; i < _alpha_vectors.size(); ++i) {
    double v = dot_product(_alpha_vectors[i], b);
    if (v > best_val) {
      best_val = v;
      best_idx = i;
    }
  }
  return _actions[_alpha_vectors[best_idx].action_idx];
}

SK_WITNESS_TEMPLATE_DECL
typename SK_WITNESS_CLASS::Value
SK_WITNESS_CLASS::get_best_value_from_belief(const Belief &b) {
  if (_alpha_vectors.empty())
    throw std::runtime_error("Witness: no solution available");

  double best_val = -std::numeric_limits<double>::infinity();
  for (const auto &alpha : _alpha_vectors) {
    double v = dot_product(alpha, b);
    if (v > best_val)
      best_val = v;
  }
  Value val;
  val.reward(best_val);
  return val;
}

SK_WITNESS_TEMPLATE_DECL
bool SK_WITNESS_CLASS::is_solution_defined_for_from_belief(const Belief &b) {
  return _has_solution;
}

// --- Statistics ---

SK_WITNESS_TEMPLATE_DECL
std::size_t SK_WITNESS_CLASS::get_nb_alpha_vectors() const {
  return _alpha_vectors.size();
}

SK_WITNESS_TEMPLATE_DECL
std::size_t SK_WITNESS_CLASS::get_nb_iterations() const {
  return _nb_iterations;
}

SK_WITNESS_TEMPLATE_DECL
std::size_t SK_WITNESS_CLASS::get_solving_time() const { return _solving_time; }

SK_WITNESS_TEMPLATE_DECL
const std::vector<typename SK_WITNESS_CLASS::AlphaVector> &
SK_WITNESS_CLASS::get_alpha_vectors() const {
  return _alpha_vectors;
}

SK_WITNESS_TEMPLATE_DECL
const std::vector<typename SK_WITNESS_CLASS::Action> &
SK_WITNESS_CLASS::get_action_list() const {
  return _actions;
}

} // namespace skdecide

#endif // SKDECIDE_WITNESS_IMPL_HH
