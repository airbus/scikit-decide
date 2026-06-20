/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_POMCP_IMPL_HH
#define SKDECIDE_POMCP_IMPL_HH

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <boost/range/irange.hpp>

#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

#define SK_POMCP_TEMPLATE_DECL                                                 \
  template <typename Tdomain, typename Texecution_policy>

#define SK_POMCP_CLASS POMCPSolver<Tdomain, Texecution_policy>

SK_POMCP_TEMPLATE_DECL
SK_POMCP_CLASS::POMCPSolver(Domain &domain, double exploration_constant,
                            double discount, std::size_t num_simulations,
                            std::size_t max_depth, double epsilon,
                            std::size_t time_budget,
                            std::size_t num_particles_belief_update,
                            double ess_threshold_ratio,
                            const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _exploration_constant(exploration_constant),
      _discount(discount), _num_simulations(num_simulations),
      _max_depth(max_depth), _epsilon(epsilon), _time_budget(time_budget),
      _num_particles_belief(num_particles_belief_update),
      _ess_threshold_ratio(ess_threshold_ratio), _callback(callback),
      _verbose(verbose), _rng(std::random_device{}()) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm POMCP");
  }
}

SK_POMCP_TEMPLATE_DECL
void SK_POMCP_CLASS::clear() {
  _current_tree.reset();
  _last_action.reset();
  _belief_particles.clear();
  _index_to_state.clear();
  _has_solution = false;
  _nb_tree_nodes = 0;
  _best_value_cache = 0.0;
}

SK_POMCP_TEMPLATE_DECL
void SK_POMCP_CLASS::solve(
    const std::vector<std::pair<State, double>> &initial_distribution) {
  clear();

  for (const auto &p : initial_distribution) {
    get_state_index(p.first);
  }

  _belief_particles = initial_distribution;

  _current_tree = std::make_unique<HistoryNode>();
  _nb_tree_nodes = 1;

  std::vector<double> probs;
  probs.reserve(_belief_particles.size());
  for (const auto &p : _belief_particles) {
    probs.push_back(p.second);
  }
  std::discrete_distribution<std::size_t> belief_dist(probs.begin(),
                                                      probs.end());
  for (std::size_t i = 0; i < _num_particles_belief; ++i) {
    std::size_t idx = belief_dist(_rng);
    _current_tree->particles.push_back(_belief_particles[idx].first);
  }

  _has_solution = true;

  if (_verbose)
    Logger::debug("POMCP: initialized with " +
                  std::to_string(_belief_particles.size()) +
                  " belief particles");
}

// --- Search: Algorithm 2 from the paper (parallel rollouts) ---

SK_POMCP_TEMPLATE_DECL
void SK_POMCP_CLASS::search(HistoryNode *root) {
  Logger::info("Running POMCP solver");
  _start_time = std::chrono::high_resolution_clock::now();

  if (root->particles.empty()) {
    Logger::warn("POMCP: search called with empty particle set");
    return;
  }

  boost::integer_range<std::size_t> parallel_rollouts(
      0, _domain.get_parallel_capacity());

  std::for_each(
      ExecutionPolicy::policy, parallel_rollouts.begin(),
      parallel_rollouts.end(), [this, root](const std::size_t &thread_id) {
        std::mt19937 thread_rng;
        _execution_policy.protect(
            [this, &thread_rng, &thread_id]() {
              thread_rng.seed(_rng() + thread_id);
            },
            _gen_mutex);

        std::size_t sim_count = 0;
        std::vector<std::pair<Observation, Action>> current_trajectory;
        do {
          State s;
          _execution_policy.protect(
              [&root, &s, &thread_rng]() {
                std::uniform_int_distribution<std::size_t> particle_dist(
                    0, root->particles.size() - 1);
                s = root->particles[particle_dist(thread_rng)];
              },
              root->mutex);

          current_trajectory.clear();
          simulate(s, root, 0, &thread_id, current_trajectory);

          // Save the last trajectory (from the last simulation)
          _execution_policy.protect(
              [this, &current_trajectory]() {
                _last_trajectory = current_trajectory;
              },
              _trajectory_mutex);

          sim_count++;
        } while (!_callback(*this, _domain) &&
                 (_time_budget == 0 || elapsed_ms() < _time_budget) &&
                 sim_count < _num_simulations);
      });

  double best_val = -std::numeric_limits<double>::infinity();
  bool found = false;
  for (auto &[key, an] : root->action_children) {
    if (an->visits_count > 0 && an->value > best_val) {
      best_val = an->value;
      _best_action_cache = an->action;
      found = true;
    }
  }
  _best_value_cache = found ? best_val : 0.0;

  if (_verbose) {
    Logger::debug(
        "POMCP: search done in " + std::to_string(elapsed_ms()) +
        "ms, tree nodes: " + std::to_string((std::size_t)_nb_tree_nodes));
  }

  Logger::info(
      "POMCP finished in " + StringConverter::from((double)elapsed_ms() / 1e3) +
      " seconds with " + StringConverter::from((std::size_t)_nb_tree_nodes) +
      " tree nodes.");
}

// --- Simulate: Algorithm 1 from the paper ---

SK_POMCP_TEMPLATE_DECL
double SK_POMCP_CLASS::simulate(
    const State &s, HistoryNode *h, std::size_t depth,
    const std::size_t *thread_id,
    std::vector<std::pair<Observation, Action>> &trajectory) {
  if (std::pow(_discount, static_cast<double>(depth)) < _epsilon)
    return 0.0;
  if (depth >= _max_depth)
    return 0.0;
  if (_domain.is_terminal(s, thread_id))
    return 0.0;

  bool is_leaf = false;
  _execution_policy.protect(
      [&h, &is_leaf]() { is_leaf = h->action_children.empty(); }, h->mutex);

  if (is_leaf) {
    _execution_policy.protect(
        [this, &h, &s, &thread_id]() {
          if (h->action_children.empty()) {
            expand(h, s, thread_id);
          }
        },
        h->mutex);

    double v = rollout(s, depth, thread_id);

    _execution_policy.protect(
        [&h, &s]() {
          h->visits_count++;
          h->particles.push_back(s);
        },
        h->mutex);

    return v;
  }

  ActionNode *an = nullptr;
  _execution_policy.protect([this, &h, &an]() { an = select_action_ucb1(h); },
                            h->mutex);

  if (!an) {
    return 0.0;
  }

  SimulationResult result = simulate_transition(s, an->action, thread_id);

  // Record the observation-action pair in the trajectory
  trajectory.push_back(std::make_pair(result.observation, an->action));

  double R;
  if (result.terminal) {
    R = result.reward;
  } else {
    std::size_t obs_hash = typename Observation::Hash()(result.observation);

    HistoryNode *child = nullptr;
    _execution_policy.protect(
        [this, &an, &obs_hash, &child]() {
          auto it = an->observation_children.find(obs_hash);
          if (it == an->observation_children.end()) {
            auto new_child = std::make_unique<HistoryNode>();
            new_child->parent = an;
            it =
                an->observation_children.emplace(obs_hash, std::move(new_child))
                    .first;
            ++_nb_tree_nodes;
          }
          child = it->second.get();
        },
        an->mutex);

    R = result.reward + _discount * simulate(result.next_state, child,
                                             depth + 1, thread_id, trajectory);
  }

  _execution_policy.protect(
      [&h, &s]() {
        h->particles.push_back(s);
        h->visits_count++;
      },
      h->mutex);

  an->visits_count++;
  double n = static_cast<double>(an->visits_count);
  an->value = (double)an->value + (R - (double)an->value) / n;

  return R;
}

// --- Rollout: random policy ---

SK_POMCP_TEMPLATE_DECL
double SK_POMCP_CLASS::rollout(const State &s, std::size_t depth,
                               const std::size_t *thread_id) {
  if (std::pow(_discount, static_cast<double>(depth)) < _epsilon)
    return 0.0;
  if (depth >= _max_depth)
    return 0.0;
  if (_domain.is_terminal(s, thread_id))
    return 0.0;

  auto actions = _domain.get_applicable_actions(s, thread_id);
  std::vector<Action> action_vec;
  for (auto a : actions.get_elements()) {
    action_vec.push_back(a);
  }
  if (action_vec.empty())
    return 0.0;

  std::size_t action_idx;
  _execution_policy.protect(
      [this, &action_idx, &action_vec]() {
        std::uniform_int_distribution<std::size_t> action_dist(
            0, action_vec.size() - 1);
        action_idx = action_dist(_rng);
      },
      _gen_mutex);
  const Action &action = action_vec[action_idx];

  auto next_dist =
      _domain.get_next_state_distribution(s, action, thread_id).get_values();

  std::vector<State> t_states;
  std::vector<double> t_probs;
  for (auto ns_item : next_dist) {
    t_states.push_back(ns_item.state());
    t_probs.push_back(ns_item.probability());
  }
  if (t_states.empty())
    return 0.0;

  std::size_t ns_idx;
  _execution_policy.protect(
      [this, &ns_idx, &t_probs]() {
        std::discrete_distribution<std::size_t> t_dist(t_probs.begin(),
                                                       t_probs.end());
        ns_idx = t_dist(_rng);
      },
      _gen_mutex);
  const State &next_state = t_states[ns_idx];

  double reward =
      _domain.get_transition_value(s, action, next_state, thread_id).reward();
  bool terminal = _domain.is_terminal(next_state, thread_id);

  if (terminal)
    return reward;

  return reward + _discount * rollout(next_state, depth + 1, thread_id);
}

// --- UCB1 action selection ---

SK_POMCP_TEMPLATE_DECL
typename SK_POMCP_CLASS::ActionNode *
SK_POMCP_CLASS::select_action_ucb1(HistoryNode *h) {
  ActionNode *best = nullptr;
  double best_score = -std::numeric_limits<double>::infinity();
  double log_n = std::log(static_cast<double>(h->visits_count));

  for (auto &[key, an] : h->action_children) {
    if (an->visits_count == 0) {
      return an.get();
    }
    double score = an->value +
                   _exploration_constant *
                       std::sqrt(log_n / static_cast<double>(an->visits_count));
    if (score > best_score) {
      best_score = score;
      best = an.get();
    }
  }
  return best;
}

// --- Expand: create action children for a new leaf ---

SK_POMCP_TEMPLATE_DECL
void SK_POMCP_CLASS::expand(HistoryNode *h, const State &s,
                            const std::size_t *thread_id) {
  auto actions = _domain.get_applicable_actions(s, thread_id);
  for (auto a : actions.get_elements()) {
    auto an = std::make_unique<ActionNode>(a, h);
    h->action_children.emplace(a, std::move(an));
  }
}

// --- Simulate transition: call domain methods ---

SK_POMCP_TEMPLATE_DECL
typename SK_POMCP_CLASS::SimulationResult
SK_POMCP_CLASS::simulate_transition(const State &s, const Action &a,
                                    const std::size_t *thread_id) {
  auto next_dist =
      _domain.get_next_state_distribution(s, a, thread_id).get_values();

  std::vector<State> t_states;
  std::vector<double> t_probs;
  for (auto ns_item : next_dist) {
    t_states.push_back(ns_item.state());
    t_probs.push_back(ns_item.probability());
  }

  if (t_states.empty()) {
    return {s, Observation(), 0.0, true};
  }

  std::size_t ns_idx;
  _execution_policy.protect(
      [this, &ns_idx, &t_probs]() {
        std::discrete_distribution<std::size_t> t_dist(t_probs.begin(),
                                                       t_probs.end());
        ns_idx = t_dist(_rng);
      },
      _gen_mutex);
  State next_state = t_states[ns_idx];
  get_state_index(next_state);

  double reward =
      _domain.get_transition_value(s, a, next_state, thread_id).reward();
  bool terminal = _domain.is_terminal(next_state, thread_id);

  auto obs_dist = _domain.get_observation_distribution(next_state, a, thread_id)
                      .get_values();

  std::vector<Observation> o_obs;
  std::vector<double> o_probs;
  for (auto o_item : obs_dist) {
    o_obs.push_back(o_item.observation());
    o_probs.push_back(o_item.probability());
  }

  Observation obs;
  if (!o_obs.empty()) {
    std::size_t o_idx;
    _execution_policy.protect(
        [this, &o_idx, &o_probs]() {
          std::discrete_distribution<std::size_t> o_dist(o_probs.begin(),
                                                         o_probs.end());
          o_idx = o_dist(_rng);
        },
        _gen_mutex);
    obs = o_obs[o_idx];
  }

  return {next_state, obs, reward, terminal};
}

// --- Belief tracking for observation-based interface ---

SK_POMCP_TEMPLATE_DECL
void SK_POMCP_CLASS::update_belief_particles(const Observation &obs) {
  if (!_last_action || _belief_particles.empty())
    return;

  const Action &action = *_last_action;
  std::size_t obs_hash = typename Observation::Hash()(obs);

  std::vector<std::pair<State, double>> new_particles;
  double total_weight = 0.0;

  for (const auto &particle : _belief_particles) {
    if (_domain.is_terminal(particle.first, nullptr))
      continue;

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

  if (total_weight > 0.0) {
    for (auto &p : new_particles) {
      p.second /= total_weight;
    }
    _belief_particles = std::move(new_particles);
  }

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

SK_POMCP_TEMPLATE_DECL
typename SK_POMCP_CLASS::Belief SK_POMCP_CLASS::particles_to_belief() const {
  Belief b;
  for (const auto &p : _belief_particles) {
    std::size_t h = typename State::Hash()(p.first);
    b[h] += p.second;
  }
  return b;
}

// --- Online planning from belief ---

SK_POMCP_TEMPLATE_DECL
void SK_POMCP_CLASS::plan_from_belief(const Belief &b) {
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
    throw std::runtime_error("POMCP: empty belief, cannot plan");
  }

  auto root = std::make_unique<HistoryNode>();

  std::discrete_distribution<std::size_t> belief_dist(probs.begin(),
                                                      probs.end());
  for (std::size_t i = 0; i < _num_particles_belief; ++i) {
    std::size_t idx = belief_dist(_rng);
    root->particles.push_back(states[idx]);
  }

  _current_tree = std::move(root);
  _nb_tree_nodes = 1;

  search(_current_tree.get());
}

// --- Observation-based interface ---

SK_POMCP_TEMPLATE_DECL
const typename SK_POMCP_CLASS::Action &
SK_POMCP_CLASS::get_best_action(const Observation &obs) {
  if (_last_action && _current_tree) {
    auto action_it = _current_tree->action_children.find(*_last_action);
    if (action_it != _current_tree->action_children.end()) {
      std::size_t obs_hash = typename Observation::Hash()(obs);
      auto obs_it = action_it->second->observation_children.find(obs_hash);
      if (obs_it != action_it->second->observation_children.end()) {
        auto subtree = std::move(obs_it->second);
        subtree->parent = nullptr;
        _current_tree = std::move(subtree);
      } else {
        _current_tree = std::make_unique<HistoryNode>();
        _nb_tree_nodes = 1;
      }
    } else {
      _current_tree = std::make_unique<HistoryNode>();
      _nb_tree_nodes = 1;
    }
  }

  update_belief_particles(obs);

  if (_current_tree->particles.empty()) {
    std::vector<double> probs;
    for (const auto &p : _belief_particles)
      probs.push_back(p.second);
    if (!probs.empty()) {
      std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
      for (std::size_t i = 0; i < _num_particles_belief; ++i) {
        std::size_t idx = dist(_rng);
        _current_tree->particles.push_back(_belief_particles[idx].first);
      }
    }
  }

  search(_current_tree.get());

  _last_action = std::make_unique<Action>(_best_action_cache);
  return _best_action_cache;
}

SK_POMCP_TEMPLATE_DECL
typename SK_POMCP_CLASS::Value
SK_POMCP_CLASS::get_best_value(const Observation &obs) {
  update_belief_particles(obs);
  Belief b = particles_to_belief();
  plan_from_belief(b);
  return Value(_best_value_cache, true);
}

SK_POMCP_TEMPLATE_DECL
bool SK_POMCP_CLASS::is_solution_defined_for(const Observation &obs) {
  return _has_solution;
}

SK_POMCP_TEMPLATE_DECL
void SK_POMCP_CLASS::reset_belief() {
  _last_action.reset();
  _current_tree.reset();
}

// --- Belief-based interface ---

SK_POMCP_TEMPLATE_DECL
const typename SK_POMCP_CLASS::Action &
SK_POMCP_CLASS::get_best_action_from_belief(const Belief &b) {
  plan_from_belief(b);
  return _best_action_cache;
}

SK_POMCP_TEMPLATE_DECL
typename SK_POMCP_CLASS::Value
SK_POMCP_CLASS::get_best_value_from_belief(const Belief &b) {
  plan_from_belief(b);
  return Value(_best_value_cache, true);
}

SK_POMCP_TEMPLATE_DECL
bool SK_POMCP_CLASS::is_solution_defined_for_from_belief(const Belief &b) {
  return _has_solution;
}

// --- Statistics ---

SK_POMCP_TEMPLATE_DECL
std::size_t SK_POMCP_CLASS::get_nb_tree_nodes() const { return _nb_tree_nodes; }

SK_POMCP_TEMPLATE_DECL
std::size_t SK_POMCP_CLASS::get_solving_time() const {
  std::size_t ms;
  const_cast<POMCPSolver *>(this)->_execution_policy.protect(
      [this, &ms]() {
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::high_resolution_clock::now() - _start_time)
                 .count();
      },
      const_cast<typename ExecutionPolicy::Mutex &>(_time_mutex));
  return ms;
}

SK_POMCP_TEMPLATE_DECL
std::size_t SK_POMCP_CLASS::get_state_index(const State &s) {
  std::size_t h = typename State::Hash()(s);
  _execution_policy.protect(
      [&]() {
        if (_index_to_state.find(h) == _index_to_state.end()) {
          _index_to_state[h] = s;
        }
      },
      _state_index_mutex);
  return h;
}

SK_POMCP_TEMPLATE_DECL
const std::unordered_map<std::size_t, typename SK_POMCP_CLASS::State> &
SK_POMCP_CLASS::get_index_to_state() const {
  return _index_to_state;
}

SK_POMCP_TEMPLATE_DECL
std::size_t SK_POMCP_CLASS::elapsed_ms() const {
  return const_cast<POMCPSolver *>(this)->get_solving_time();
}

SK_POMCP_TEMPLATE_DECL
std::vector<std::pair<typename SK_POMCP_CLASS::Observation,
                      typename SK_POMCP_CLASS::Action>>
SK_POMCP_CLASS::get_last_trajectory() const {
  std::vector<std::pair<Observation, Action>> trajectory;
  const_cast<POMCPSolver *>(this)->_execution_policy.protect(
      [&]() { trajectory = _last_trajectory; },
      const_cast<typename ExecutionPolicy::Mutex &>(_trajectory_mutex));
  return trajectory;
}

} // namespace skdecide

#endif // SKDECIDE_POMCP_IMPL_HH
