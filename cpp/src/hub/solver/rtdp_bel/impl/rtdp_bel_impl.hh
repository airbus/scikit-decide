/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Implementation of RTDP-Bel from Figure 2 of:
 * Bonet & Geffner, "Solving POMDPs: RTDP-Bel vs. Point-based Algorithms",
 * IJCAI 2009.
 */
#ifndef SKDECIDE_RTDP_BEL_IMPL_HH
#define SKDECIDE_RTDP_BEL_IMPL_HH

#include <algorithm>
#include <limits>
#include <numeric>
#include <functional>

#include "utils/string_converter.hh"
#include "utils/logging.hh"

namespace skdecide {

#define SK_RTDP_BEL_TEMPLATE_DECL                                              \
  template <typename Tdomain, typename Texecution_policy>

#define SK_RTDP_BEL_CLASS RTDPBelSolver<Tdomain, Texecution_policy>

// --- DiscretizedBeliefHash ---

SK_RTDP_BEL_TEMPLATE_DECL
std::size_t SK_RTDP_BEL_CLASS::DiscretizedBeliefHash::operator()(
    const DiscretizedBelief &db) const {
  std::size_t seed = db.size();
  for (const auto &p : db) {
    seed ^= std::hash<std::size_t>()(p.first) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    seed ^= std::hash<std::size_t>()(p.second) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
  }
  return seed;
}

// --- DiscretizedBeliefEqual ---

SK_RTDP_BEL_TEMPLATE_DECL
bool SK_RTDP_BEL_CLASS::DiscretizedBeliefEqual::operator()(
    const DiscretizedBelief &a, const DiscretizedBelief &b) const {
  return a == b;
}

// --- BeliefNode ---

SK_RTDP_BEL_TEMPLATE_DECL
SK_RTDP_BEL_CLASS::BeliefNode::BeliefNode(const Belief &b,
                                          const DiscretizedBelief &db)
    : belief(b), discretized(db), best_action(nullptr), best_value(0.0),
      goal(false), solved(false) {}

// --- ActionNode ---

SK_RTDP_BEL_TEMPLATE_DECL
SK_RTDP_BEL_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), value(0.0) {}

// --- Constructor ---

SK_RTDP_BEL_TEMPLATE_DECL
SK_RTDP_BEL_CLASS::RTDPBelSolver(
    Domain &domain, const GoalCheckerFunctor &goal_checker,
    const HeuristicFunctor &heuristic,
    const TerminalValueFunctor &terminal_value, std::size_t discretization,
    std::size_t time_budget, std::size_t rollout_budget, std::size_t max_depth,
    double epsilon, double discount, const CallbackFunctor &callback,
    bool verbose)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _terminal_value(terminal_value), _discretization(discretization),
      _time_budget(time_budget), _rollout_budget(rollout_budget),
      _max_depth(max_depth), _epsilon(epsilon), _discount(discount),
      _callback(callback), _verbose(verbose), _nb_rollouts(0),
      _initial_belief_node(nullptr), _current_belief_node(nullptr),
      _last_action(nullptr), _next_state_index(0) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm RTDP-Bel");
  }
  std::random_device rd;
  _gen = std::make_unique<std::mt19937>(rd());
}

SK_RTDP_BEL_TEMPLATE_DECL
void SK_RTDP_BEL_CLASS::clear() {
  _belief_graph.clear();
  _index_to_state.clear();
  _next_state_index = 0;
  _nb_rollouts = 0;
  _initial_belief_node = nullptr;
  _current_belief_node = nullptr;
  if (_last_action) {
    delete _last_action;
    _last_action = nullptr;
  }
}

// --- State indexing ---

SK_RTDP_BEL_TEMPLATE_DECL
std::size_t SK_RTDP_BEL_CLASS::get_state_index(const State &s) {
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

// --- Belief utilities ---

SK_RTDP_BEL_TEMPLATE_DECL
typename SK_RTDP_BEL_CLASS::DiscretizedBelief
SK_RTDP_BEL_CLASS::discretize(const Belief &b) const {
  DiscretizedBelief db;
  for (const auto &p : b) {
    std::size_t d =
        static_cast<std::size_t>(std::ceil(_discretization * p.second));
    if (d > 0) {
      db[p.first] = d;
    }
  }
  return db;
}

SK_RTDP_BEL_TEMPLATE_DECL
double SK_RTDP_BEL_CLASS::heuristic_value(const Belief &b,
                                          const std::size_t *thread_id) const {
  double h = 0.0;
  for (const auto &p : b) {
    auto it = _index_to_state.find(p.first);
    if (it != _index_to_state.end()) {
      h += p.second * _heuristic(_domain, it->second, thread_id).cost();
    }
  }
  return h;
}

SK_RTDP_BEL_TEMPLATE_DECL
bool SK_RTDP_BEL_CLASS::is_goal_belief(const Belief &b,
                                       const std::size_t *thread_id) const {
  for (const auto &p : b) {
    if (p.second > 0.0) {
      auto it = _index_to_state.find(p.first);
      if (it != _index_to_state.end() &&
          !_goal_checker(_domain, it->second, thread_id)) {
        return false;
      }
    }
  }
  return true;
}

SK_RTDP_BEL_TEMPLATE_DECL
typename SK_RTDP_BEL_CLASS::State
SK_RTDP_BEL_CLASS::sample_state_from_belief(const Belief &b,
                                            const std::size_t *thread_id) {
  std::vector<double> weights;
  std::vector<std::size_t> indices;
  for (const auto &p : b) {
    indices.push_back(p.first);
    weights.push_back(p.second);
  }
  std::discrete_distribution<> dist(weights.begin(), weights.end());
  std::size_t idx;
  _execution_policy.protect([&]() { idx = indices[dist(*_gen)]; }, _gen_mutex);
  return _index_to_state.at(idx);
}

SK_RTDP_BEL_TEMPLATE_DECL
typename SK_RTDP_BEL_CLASS::Belief SK_RTDP_BEL_CLASS::compute_posterior_belief(
    const Belief &b, const Action &a, const Observation &o,
    const std::size_t *thread_id) const {
  // Bayes belief update (equations 4-6 from the paper):
  // b_a(s') = Σ_s P_a(s'|s)*b(s)     (prediction)
  // b_a(o) = Σ_s' Q_a(o|s')*b_a(s')  (observation probability)
  // b^o_a(s') = Q_a(o|s')*b_a(s') / b_a(o)  (posterior)

  // Step 1: prediction — compute b_a(s') for each possible next state
  Belief b_a;
  for (const auto &p : b) {
    auto it = _index_to_state.find(p.first);
    if (it == _index_to_state.end())
      continue;
    const State &s = it->second;
    auto next_dist =
        _domain.get_next_state_distribution(s, a, thread_id).get_values();
    if (next_dist.begin() == next_dist.end()) {
      // Action has no transitions: state stays in place (self-loop)
      // This is a reasonable assumption for belief update
      b_a[p.first] += p.second;
    } else {
      for (auto ns : next_dist) {
        std::size_t ns_idx =
            const_cast<RTDPBelSolver *>(this)->get_state_index(ns.state());
        b_a[ns_idx] += p.second * ns.probability();
      }
    }
  }

  // Step 2: compute b_a(o) and posterior b^o_a
  Belief posterior;
  double b_a_o = 0.0;

  for (const auto &p : b_a) {
    auto it = _index_to_state.find(p.first);
    if (it == _index_to_state.end())
      continue;
    const State &sp = it->second;
    auto obs_dist =
        _domain.get_observation_distribution(sp, a, thread_id).get_values();
    for (auto od : obs_dist) {
      if (typename Observation::Equal()(od.observation(), o)) {
        double q_o_sp = od.probability();
        b_a_o += q_o_sp * p.second;
        posterior[p.first] += q_o_sp * p.second;
        break;
      }
    }
  }

  // Normalize
  if (b_a_o > 0.0) {
    for (auto &p : posterior) {
      p.second /= b_a_o;
    }
  }

  // Remove zero entries
  for (auto it = posterior.begin(); it != posterior.end();) {
    if (it->second <= 0.0) {
      it = posterior.erase(it);
    } else {
      ++it;
    }
  }

  return posterior;
}

// --- Belief node management ---

SK_RTDP_BEL_TEMPLATE_DECL
typename SK_RTDP_BEL_CLASS::BeliefNode *
SK_RTDP_BEL_CLASS::get_or_create_belief_node(const Belief &b,
                                             const std::size_t *thread_id) {
  BeliefNode *result = nullptr;
  _execution_policy.protect(
      [this, &b, &result, &thread_id]() {
        DiscretizedBelief db = discretize(b);
        auto it = _belief_graph.find(db);
        if (it != _belief_graph.end()) {
          result = it->second.get();
          return;
        }
        auto node = std::make_unique<BeliefNode>(b, db);
        node->goal = is_goal_belief(b, thread_id);
        if (node->goal) {
          node->best_value = 0.0;
          node->solved = true;
        } else {
          node->best_value = heuristic_value(b, thread_id);
        }
        result = node.get();
        _belief_graph[db] = std::move(node);
      },
      _graph_mutex);
  return result;
}

// --- expand (generate observation-based successors) ---

SK_RTDP_BEL_TEMPLATE_DECL
void SK_RTDP_BEL_CLASS::expand(BeliefNode *bn, const std::size_t *thread_id) {
  if (_verbose)
    Logger::debug("Expanding belief node");

  State representative = _index_to_state.at(bn->belief.begin()->first);
  auto applicable_actions =
      _domain.get_applicable_actions(representative, thread_id).get_elements();

  for (auto a : applicable_actions) {
    bn->actions.push_back(std::make_unique<ActionNode>(a));
    ActionNode &an = *(bn->actions.back());

    std::unordered_map<std::size_t, double> obs_probs;
    std::unordered_map<std::size_t, Observation> obs_map;

    for (const auto &bp : bn->belief) {
      auto it = _index_to_state.find(bp.first);
      if (it == _index_to_state.end())
        continue;
      const State &s = it->second;

      auto next_dist =
          _domain.get_next_state_distribution(s, a, thread_id).get_values();
      for (auto ns : next_dist) {
        get_state_index(ns.state());
        auto obs_dist =
            _domain.get_observation_distribution(ns.state(), a, thread_id)
                .get_values();
        for (auto od : obs_dist) {
          std::size_t oh = typename Observation::Hash()(od.observation());
          obs_probs[oh] += bp.second * ns.probability() * od.probability();
          obs_map[oh] = od.observation();
        }
      }
    }

    for (const auto &op : obs_probs) {
      if (op.second <= 0.0)
        continue;
      const Observation &o = obs_map.at(op.first);
      Belief posterior =
          compute_posterior_belief(bn->belief, an.action, o, thread_id);
      BeliefNode *next_bn = get_or_create_belief_node(posterior, thread_id);
      an.outcomes.push_back(std::make_tuple(op.second, next_bn));
    }
  }
}

// --- q_value (cost minimization over observations) ---

SK_RTDP_BEL_TEMPLATE_DECL
double SK_RTDP_BEL_CLASS::q_value(ActionNode *a) {
  a->value = 0.0;
  for (const auto &outcome : a->outcomes) {
    double prob = std::get<0>(outcome);
    BeliefNode *next = std::get<1>(outcome);
    a->value += prob * (_discount * next->best_value);
  }
  // Add expected immediate cost c(a,b) — already factored into the
  // transition costs stored in the domain. For belief MDPs:
  // c(a,b) = Σ_s c(a,s)*b(s)
  // We compute this from the parent belief and action during expand.
  // For now, we add it as part of the Q-value computation.
  return a->value;
}

// --- greedy_action ---

SK_RTDP_BEL_TEMPLATE_DECL
typename SK_RTDP_BEL_CLASS::ActionNode *
SK_RTDP_BEL_CLASS::greedy_action(BeliefNode *bn, const std::size_t *thread_id) {
  if (bn->actions.empty()) {
    expand(bn, thread_id);
  }

  double best_val = std::numeric_limits<double>::infinity();
  ActionNode *best_act = nullptr;

  for (auto &a : bn->actions) {
    double immediate_cost = 0.0;
    for (const auto &bp : bn->belief) {
      auto it = _index_to_state.find(bp.first);
      if (it != _index_to_state.end()) {
        const State &s = it->second;

        // Check if this state is a terminal state (non-goal)
        if (_domain.is_terminal(s, thread_id) &&
            !_goal_checker(_domain, s, thread_id)) {
          // Terminal non-goal state: use terminal_value directly
          // (no transitions expected, or should be penalized)
          immediate_cost += bp.second * _terminal_value(s).cost();
        } else {
          // Normal state or goal state: compute expected immediate cost
          auto next_dist =
              _domain.get_next_state_distribution(s, a->action, thread_id)
                  .get_values();
          if (next_dist.begin() == next_dist.end()) {
            // Action has no transitions: treat as terminal state
            immediate_cost += bp.second * _terminal_value(s).cost();
          } else {
            for (auto ns : next_dist) {
              immediate_cost +=
                  bp.second * ns.probability() *
                  _domain
                      .get_transition_value(s, a->action, ns.state(), thread_id)
                      .cost();
            }
          }
        }
      }
    }

    double qv = immediate_cost + q_value(a.get());
    a->value = qv;

    if (qv < best_val) {
      best_val = qv;
      best_act = a.get();
    }
  }

  return best_act;
}

// --- update ---

SK_RTDP_BEL_TEMPLATE_DECL
void SK_RTDP_BEL_CLASS::update(BeliefNode *bn, const std::size_t *thread_id) {
  bn->best_action = greedy_action(bn, thread_id);
  if (bn->best_action) {
    bn->best_value = bn->best_action->value;
  }
}

// --- trial (Figure 2 from the paper) ---

SK_RTDP_BEL_TEMPLATE_DECL
void SK_RTDP_BEL_CLASS::trial(BeliefNode *bn, const std::size_t *thread_id) {
  BeliefNode *current = bn;
  std::vector<BeliefNode *> current_trajectory;
  std::size_t depth = 0;

  State s = sample_state_from_belief(current->belief, thread_id);

  while (!current->goal && !current->solved && depth < _max_depth &&
         get_solving_time() < _time_budget) {
    depth++;
    current_trajectory.push_back(current);

    _execution_policy.protect(
        [this, &current, &thread_id]() { update(current, thread_id); },
        current->mutex);

    if (current->best_action == nullptr)
      break;

    auto next_dist = _domain
                         .get_next_state_distribution(
                             s, current->best_action->action, thread_id)
                         .get_values();
    std::vector<double> weights;
    std::vector<State> states;
    for (auto ns : next_dist) {
      states.push_back(ns.state());
      weights.push_back(ns.probability());
    }
    if (states.empty())
      break;
    std::discrete_distribution<> state_dist(weights.begin(), weights.end());
    State sp;
    _execution_policy.protect([&]() { sp = states[state_dist(*_gen)]; },
                              _gen_mutex);

    auto obs_dist = _domain
                        .get_observation_distribution(
                            sp, current->best_action->action, thread_id)
                        .get_values();
    std::vector<double> obs_weights;
    std::vector<Observation> observations;
    for (auto od : obs_dist) {
      observations.push_back(od.observation());
      obs_weights.push_back(od.probability());
    }
    if (observations.empty())
      break;
    std::discrete_distribution<> obs_dist_sampler(obs_weights.begin(),
                                                  obs_weights.end());
    Observation o;
    _execution_policy.protect(
        [&]() { o = observations[obs_dist_sampler(*_gen)]; }, _gen_mutex);

    Belief posterior = compute_posterior_belief(
        current->belief, current->best_action->action, o, thread_id);

    BeliefNode *next = get_or_create_belief_node(posterior, thread_id);
    if (next->goal)
      break;

    current = next;
    s = sp;
  }

  // Save the trajectory after the trial completes
  _last_trajectory = current_trajectory;
}

// --- solve ---

SK_RTDP_BEL_TEMPLATE_DECL
void SK_RTDP_BEL_CLASS::solve(
    const std::vector<std::pair<State, double>> &initial_distribution) {
  try {
    Logger::info("Running RTDP-Bel solver");
    _start_time = std::chrono::high_resolution_clock::now();

    Belief b0;
    for (const auto &p : initial_distribution) {
      std::size_t idx = get_state_index(p.first);
      b0[idx] = p.second;
    }

    BeliefNode *root = get_or_create_belief_node(b0, nullptr);
    _initial_belief_node = root;
    _current_belief_node = root;
    if (_last_action) {
      delete _last_action;
      _last_action = nullptr;
    }

    if (root->goal) {
      Logger::info("RTDP-Bel: initial belief is already a goal");
      return;
    }

    _nb_rollouts = 0;
    boost::integer_range<std::size_t> parallel_rollouts(
        0, _domain.get_parallel_capacity());

    std::for_each(
        ExecutionPolicy::policy, parallel_rollouts.begin(),
        parallel_rollouts.end(), [this, &root](const std::size_t &thread_id) {
          do {
            _nb_rollouts++;
            if (_verbose)
              Logger::debug("Starting trial " +
                            StringConverter::from((std::size_t)_nb_rollouts));
            trial(root, &thread_id);
          } while (!_callback(*this, _domain, &thread_id) &&
                   get_solving_time() < _time_budget &&
                   _nb_rollouts < _rollout_budget);
        });

    Logger::info("RTDP-Bel finished in " +
                 StringConverter::from((double)get_solving_time() / 1e3) +
                 " seconds with " + StringConverter::from(_nb_rollouts) +
                 " trials and " + StringConverter::from(_belief_graph.size()) +
                 " belief nodes.");
  } catch (const std::exception &e) {
    Logger::error("RTDP-Bel failed: " + std::string(e.what()));
    throw;
  }
}

// --- Internal belief update ---

SK_RTDP_BEL_TEMPLATE_DECL
void SK_RTDP_BEL_CLASS::update_current_belief(const Observation &obs) {
  if (_current_belief_node == nullptr) {
    throw std::runtime_error(
        "SKDECIDE exception: RTDP-Bel solver has no current belief. "
        "Call solve() first.");
  }
  if (_last_action != nullptr) {
    Belief posterior = compute_posterior_belief(_current_belief_node->belief,
                                                *_last_action, obs, nullptr);
    _current_belief_node = get_or_create_belief_node(posterior, nullptr);
  }
}

// === Default interface: observation-based ===

SK_RTDP_BEL_TEMPLATE_DECL
const typename SK_RTDP_BEL_CLASS::Action &
SK_RTDP_BEL_CLASS::get_best_action(const Observation &obs) {
  update_current_belief(obs);

  if (_current_belief_node->goal) {
    throw std::runtime_error(
        "SKDECIDE exception: current belief is a goal belief.");
  }

  if (_current_belief_node->best_action == nullptr) {
    update(_current_belief_node, nullptr);
  }

  if (_current_belief_node->best_action == nullptr) {
    throw std::runtime_error(
        "SKDECIDE exception: no best action found for current belief.");
  }

  if (_last_action) {
    delete _last_action;
  }
  _last_action = new Action(_current_belief_node->best_action->action);

  return _current_belief_node->best_action->action;
}

SK_RTDP_BEL_TEMPLATE_DECL
typename SK_RTDP_BEL_CLASS::Value
SK_RTDP_BEL_CLASS::get_best_value(const Observation &obs) {
  update_current_belief(obs);
  Value val;
  val.cost(_current_belief_node->best_value);
  return val;
}

SK_RTDP_BEL_TEMPLATE_DECL
bool SK_RTDP_BEL_CLASS::is_solution_defined_for(const Observation &obs) {
  update_current_belief(obs);
  return _current_belief_node->best_action != nullptr ||
         _current_belief_node->goal;
}

SK_RTDP_BEL_TEMPLATE_DECL
std::pair<typename SK_RTDP_BEL_CLASS::Action, double>
SK_RTDP_BEL_CLASS::get_policy(const Observation &obs) {
  const Action &a = get_best_action(obs);
  return std::make_pair(a, _current_belief_node->best_value);
}

SK_RTDP_BEL_TEMPLATE_DECL
void SK_RTDP_BEL_CLASS::reset_belief() {
  _current_belief_node = _initial_belief_node;
  if (_last_action) {
    delete _last_action;
    _last_action = nullptr;
  }
}

// === Belief-state interface ===

SK_RTDP_BEL_TEMPLATE_DECL
const typename SK_RTDP_BEL_CLASS::Action &
SK_RTDP_BEL_CLASS::get_best_action_from_belief(const Belief &b) {
  BeliefNode *bn = get_or_create_belief_node(b, nullptr);
  if (bn->best_action == nullptr) {
    update(bn, nullptr);
  }
  if (bn->best_action == nullptr) {
    throw std::runtime_error(
        "SKDECIDE exception: no best action found for given belief.");
  }
  return bn->best_action->action;
}

SK_RTDP_BEL_TEMPLATE_DECL
typename SK_RTDP_BEL_CLASS::Value
SK_RTDP_BEL_CLASS::get_best_value_from_belief(const Belief &b) {
  BeliefNode *bn = get_or_create_belief_node(b, nullptr);
  Value val;
  val.cost(bn->best_value);
  return val;
}

SK_RTDP_BEL_TEMPLATE_DECL
bool SK_RTDP_BEL_CLASS::is_solution_defined_for_from_belief(const Belief &b) {
  BeliefNode *bn = get_or_create_belief_node(b, nullptr);
  return bn->best_action != nullptr || bn->goal;
}

// --- Accessors ---

SK_RTDP_BEL_TEMPLATE_DECL
const typename SK_RTDP_BEL_CLASS::BeliefGraph &
SK_RTDP_BEL_CLASS::get_belief_graph() const {
  return _belief_graph;
}

SK_RTDP_BEL_TEMPLATE_DECL
const std::unordered_map<std::size_t, typename SK_RTDP_BEL_CLASS::State> &
SK_RTDP_BEL_CLASS::get_index_to_state() const {
  return _index_to_state;
}

SK_RTDP_BEL_TEMPLATE_DECL
std::size_t SK_RTDP_BEL_CLASS::get_discretization() const {
  return _discretization;
}

SK_RTDP_BEL_TEMPLATE_DECL
std::unordered_map<typename SK_RTDP_BEL_CLASS::DiscretizedBelief,
                   std::pair<typename SK_RTDP_BEL_CLASS::Action,
                             typename SK_RTDP_BEL_CLASS::Value>,
                   typename SK_RTDP_BEL_CLASS::DiscretizedBeliefHash,
                   typename SK_RTDP_BEL_CLASS::DiscretizedBeliefEqual>
SK_RTDP_BEL_CLASS::get_belief_policy() const {
  std::unordered_map<DiscretizedBelief, std::pair<Action, Value>,
                     DiscretizedBeliefHash, DiscretizedBeliefEqual>
      p;
  for (const auto &entry : _belief_graph) {
    const BeliefNode &bn = *(entry.second);
    if (bn.best_action != nullptr) {
      Value val;
      val.cost(bn.best_value);
      p.insert(std::make_pair(bn.discretized,
                              std::make_pair(bn.best_action->action, val)));
    }
  }
  return p;
}

// --- Statistics ---

SK_RTDP_BEL_TEMPLATE_DECL
std::size_t SK_RTDP_BEL_CLASS::get_nb_explored_beliefs() const {
  return _belief_graph.size();
}

SK_RTDP_BEL_TEMPLATE_DECL
std::size_t SK_RTDP_BEL_CLASS::get_nb_rollouts() const { return _nb_rollouts; }

SK_RTDP_BEL_TEMPLATE_DECL
std::size_t SK_RTDP_BEL_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

SK_RTDP_BEL_TEMPLATE_DECL
std::vector<std::pair<typename SK_RTDP_BEL_CLASS::Belief,
                      typename SK_RTDP_BEL_CLASS::Action>>
SK_RTDP_BEL_CLASS::get_last_trajectory() const {
  std::vector<std::pair<Belief, Action>> trajectory;
  trajectory.reserve(_last_trajectory.size());
  for (const auto *bn : _last_trajectory) {
    Action action = bn->best_action ? bn->best_action->action : Action();
    trajectory.push_back(std::make_pair(bn->belief, action));
  }
  return trajectory;
}

} // namespace skdecide

#endif // SKDECIDE_RTDP_BEL_IMPL_HH
