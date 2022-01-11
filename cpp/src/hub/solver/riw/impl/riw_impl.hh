/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_RIW_IMPL_HH
#define SKDECIDE_RIW_IMPL_HH

#include <boost/range/irange.hpp>
#include <iostream>

#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === DomainStateHash implementation ===

#define SK_RIW_DOMAIN_STATE_HASH_TEMPLATE_DECL                                 \
  template <typename Tdomain, typename Tfeature_vector>

#define SK_RIW_DOMAIN_STATE_HASH_CLASS DomainStateHash<Tdomain, Tfeature_vector>

SK_RIW_DOMAIN_STATE_HASH_TEMPLATE_DECL
template <typename Tnode>
const typename SK_RIW_DOMAIN_STATE_HASH_CLASS::Key &
SK_RIW_DOMAIN_STATE_HASH_CLASS::get_key(const Tnode &n) {
  return n.state;
}

SK_RIW_DOMAIN_STATE_HASH_TEMPLATE_DECL
std::size_t
SK_RIW_DOMAIN_STATE_HASH_CLASS::Hash::operator()(const Key &k) const {
  return typename Tdomain::State::Hash()(k);
}

SK_RIW_DOMAIN_STATE_HASH_TEMPLATE_DECL
bool SK_RIW_DOMAIN_STATE_HASH_CLASS::Equal::operator()(const Key &k1,
                                                       const Key &k2) const {
  return typename Tdomain::State::Equal()(k1, k2);
}

// === StateFeatureHash implementation ===

#define SK_RIW_STATE_FEATURE_HASH_TEMPLATE_DECL                                \
  template <typename Tdomain, typename Tfeature_vector>

#define SK_RIW_STATE_FEATURE_HASH_CLASS                                        \
  StateFeatureHash<Tdomain, Tfeature_vector>

SK_RIW_STATE_FEATURE_HASH_TEMPLATE_DECL
template <typename Tnode>
const typename SK_RIW_STATE_FEATURE_HASH_CLASS::Key &
SK_RIW_STATE_FEATURE_HASH_CLASS::get_key(const Tnode &n) {
  return *n.features;
}

SK_RIW_STATE_FEATURE_HASH_TEMPLATE_DECL
std::size_t
SK_RIW_STATE_FEATURE_HASH_CLASS::Hash::operator()(const Key &k) const {
  std::size_t seed = 0;
  for (std::size_t i = 0; i < k.size(); i++) {
    boost::hash_combine(seed, k[i]);
  }
  return seed;
}

SK_RIW_STATE_FEATURE_HASH_TEMPLATE_DECL
bool SK_RIW_STATE_FEATURE_HASH_CLASS::Equal::operator()(const Key &k1,
                                                        const Key &k2) const {
  std::size_t size = k1.size();
  if (size != k2.size()) {
    return false;
  }
  for (std::size_t i = 0; i < size; i++) {
    if (!(k1[i] == k2[i])) {
      return false;
    }
  }
  return true;
}

// === EnvironmentRollout implementation ===

#define SK_RIW_ENVIRONMENT_ROLLOUT_TEMPLATE_DECL template <typename Tdomain>

#define SK_RIW_ENVIRONMENT_ROLLOUT_CLASS EnvironmentRollout<Tdomain>

SK_RIW_ENVIRONMENT_ROLLOUT_TEMPLATE_DECL
void SK_RIW_ENVIRONMENT_ROLLOUT_CLASS::init_rollout(
    Tdomain &domain, const std::size_t *thread_id) {
  domain.reset(thread_id);
  std::for_each(_action_prefix.begin(), _action_prefix.end(),
                [&domain, &thread_id](const typename Tdomain::Action &a) {
                  domain.step(a, thread_id);
                });
}

SK_RIW_ENVIRONMENT_ROLLOUT_TEMPLATE_DECL
typename Tdomain::EnvironmentOutcome SK_RIW_ENVIRONMENT_ROLLOUT_CLASS::progress(
    Tdomain &domain, const typename Tdomain::State &state,
    const typename Tdomain::Action &action, const std::size_t *thread_id) {
  return domain.step(action, thread_id);
}

SK_RIW_ENVIRONMENT_ROLLOUT_TEMPLATE_DECL
void SK_RIW_ENVIRONMENT_ROLLOUT_CLASS::advance(
    Tdomain &domain, const typename Tdomain::State &state,
    const typename Tdomain::Action &action, bool record_action,
    const std::size_t *thread_id) {
  if (record_action) {
    _action_prefix.push_back(action);
  } else {
    domain.step(action, thread_id);
  }
}

SK_RIW_ENVIRONMENT_ROLLOUT_TEMPLATE_DECL
std::list<typename Tdomain::Action>
SK_RIW_ENVIRONMENT_ROLLOUT_CLASS::action_prefix() const {
  return _action_prefix;
}

// === SimulationRollout implementation ===

#define SK_RIW_SIMULATION_ROLLOUT_TEMPLATE_DECL template <typename Tdomain>

#define SK_RIW_SIMULATION_ROLLOUT_CLASS SimulationRollout<Tdomain>

SK_RIW_SIMULATION_ROLLOUT_TEMPLATE_DECL
void SK_RIW_SIMULATION_ROLLOUT_CLASS::init_rollout(
    [[maybe_unused]] Tdomain &domain,
    [[maybe_unused]] const std::size_t *thread_id) {}

SK_RIW_SIMULATION_ROLLOUT_TEMPLATE_DECL
typename Tdomain::EnvironmentOutcome SK_RIW_SIMULATION_ROLLOUT_CLASS::progress(
    Tdomain &domain, const typename Tdomain::State &state,
    const typename Tdomain::Action &action, const std::size_t *thread_id) {
  return domain.sample(state, action, thread_id);
}

SK_RIW_SIMULATION_ROLLOUT_TEMPLATE_DECL
void SK_RIW_SIMULATION_ROLLOUT_CLASS::advance(
    [[maybe_unused]] Tdomain &domain,
    [[maybe_unused]] const typename Tdomain::State &state,
    [[maybe_unused]] const typename Tdomain::Action &action,
    [[maybe_unused]] bool record_action,
    [[maybe_unused]] const std::size_t *thread_id) {}

SK_RIW_SIMULATION_ROLLOUT_TEMPLATE_DECL
std::list<typename Tdomain::Action>
SK_RIW_SIMULATION_ROLLOUT_CLASS::action_prefix() const {
  return std::list<typename Tdomain::Action>();
}

// === RIWSolver implementation ===

#define SK_RIW_SOLVER_TEMPLATE_DECL                                            \
  template <typename Tdomain, typename Tfeature_vector,                        \
            template <typename...> class Thashing_policy,                      \
            template <typename...> class Trollout_policy,                      \
            typename Texecution_policy>

#define SK_RIW_SOLVER_CLASS                                                    \
  RIWSolver<Tdomain, Tfeature_vector, Thashing_policy, Trollout_policy,        \
            Texecution_policy>

SK_RIW_SOLVER_TEMPLATE_DECL
SK_RIW_SOLVER_CLASS::RIWSolver(
    Domain &domain,
    const std::function<std::unique_ptr<FeatureVector>(
        Domain &, const State &, const std::size_t *)> &state_features,
    std::size_t time_budget, std::size_t rollout_budget, std::size_t max_depth,
    double exploration, std::size_t epsilon_moving_average_window,
    double epsilon, double discount, bool online_node_garbage, bool debug_logs,
    const WatchdogFunctor &watchdog)
    : _domain(domain), _state_features(state_features),
      _time_budget(time_budget), _rollout_budget(rollout_budget),
      _max_depth(max_depth), _exploration(exploration),
      _epsilon_moving_average_window(epsilon_moving_average_window),
      _epsilon(epsilon), _discount(discount),
      _online_node_garbage(online_node_garbage),
      _min_reward(std::numeric_limits<double>::max()), _nb_rollouts(0),
      _debug_logs(debug_logs), _watchdog(watchdog) {

  if (debug_logs) {
    Logger::check_level(logging::debug, "algorithm RIW");
  }

  std::random_device rd;
  _gen = std::make_unique<std::mt19937>(rd());
}

SK_RIW_SOLVER_TEMPLATE_DECL
void SK_RIW_SOLVER_CLASS::clear() {
  _graph.clear();
  _min_reward = std::numeric_limits<double>::max();
}

SK_RIW_SOLVER_TEMPLATE_DECL
void SK_RIW_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " RIW solver from state " + s.print());
    auto start_time = std::chrono::high_resolution_clock::now();
    _nb_rollouts = 0;
    std::size_t nb_of_binary_features =
        _state_features(_domain, s, nullptr)->size();

    TupleVector feature_tuples;
    bool found_solution = false;

    for (atomic_size_t w = 1; w <= nb_of_binary_features; w++) {
      if (WidthSolver(*this, _domain, _state_features, _time_budget,
                      _rollout_budget, _max_depth, _exploration,
                      _epsilon_moving_average_window, _epsilon,
                      _epsilon_moving_average, _epsilons, _discount,
                      _min_reward, w, _graph, _rollout_policy,
                      _execution_policy, *_gen, _gen_mutex, _time_mutex,
                      _epsilons_protect, _debug_logs, _watchdog)
              .solve(s, start_time, _nb_rollouts, feature_tuples)) {
        found_solution = true;
        break;
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end_time - start_time)
                        .count();
    auto exploration_statistics = get_exploration_statistics();
    std::string solution_str(found_solution ? ("finished to solve")
                                            : ("could not find a solution"));
    Logger::info("RIW " + solution_str + " from state " + s.print() + " in " +
                 StringConverter::from((double)duration / (double)1e9) +
                 " seconds with " + StringConverter::from(_nb_rollouts) +
                 " rollouts and pruned " +
                 StringConverter::from(exploration_statistics.second) +
                 " states among " +
                 StringConverter::from(exploration_statistics.first) +
                 " visited states.");
  } catch (const std::exception &e) {
    Logger::error("RIW failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_RIW_SOLVER_TEMPLATE_DECL
bool SK_RIW_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(Node(s, _domain, _state_features, nullptr));
  if ((si == _graph.end()) ||
      (si->best_action == nullptr)) { // || (si->solved == false)) {
    return false;
  } else {
    return true;
  }
}

SK_RIW_SOLVER_TEMPLATE_DECL
typename SK_RIW_SOLVER_CLASS::Action
SK_RIW_SOLVER_CLASS::get_best_action(const State &s) {
  auto si = _graph.find(Node(s, _domain, _state_features, nullptr));
  if ((si == _graph.end()) || (si->best_action == nullptr)) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  _rollout_policy.advance(_domain, s, *(si->best_action), true, nullptr);
  Action best_action = *(si->best_action);
  std::unordered_set<Node *> root_subgraph;
  compute_reachable_subgraph(const_cast<Node &>(*si),
                             root_subgraph); // we won't change the real key
                                             // (Node::state) so we are safe
  Node *next_node = nullptr;
  for (auto &child : si->children) {
    if (&std::get<0>(child) == si->best_action) {
      next_node = std::get<2>(child);
      break;
    }
  }
  if (next_node == nullptr) {
    Logger::error("SKDECIDE exception: best action's next node from state " +
                  s.print() + " not found in the graph");
    throw std::runtime_error(
        "SKDECIDE exception: best action's next node from state " + s.print() +
        " not found in the graph");
  }
  if (_debug_logs) {
    Logger::debug("Expected outcome of best action " +
                  si->best_action->print() + ": " + next_node->state.print());
  }
  std::unordered_set<Node *> child_subgraph;
  if (_online_node_garbage) {
    compute_reachable_subgraph(*next_node, child_subgraph);
  }
  update_graph(root_subgraph, child_subgraph);
  return best_action;
}

SK_RIW_SOLVER_TEMPLATE_DECL
double SK_RIW_SOLVER_CLASS::get_best_value(const State &s) const {
  auto si = _graph.find(Node(s, _domain, _state_features, nullptr));
  if (si == _graph.end()) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return si->value;
}

SK_RIW_SOLVER_TEMPLATE_DECL
std::size_t SK_RIW_SOLVER_CLASS::get_nb_of_explored_states() const {
  return _graph.size();
}

SK_RIW_SOLVER_TEMPLATE_DECL
std::size_t SK_RIW_SOLVER_CLASS::get_nb_of_pruned_states() const {
  std::size_t cnt = 0;
  for (const auto &n : _graph) {
    if (n.pruned) {
      cnt++;
    }
  }
  return cnt;
}

SK_RIW_SOLVER_TEMPLATE_DECL
std::pair<std::size_t, std::size_t>
SK_RIW_SOLVER_CLASS::get_exploration_statistics() const {
  std::size_t pruned = 0;
  std::size_t explored = 0;
  for (const auto &n : _graph) {
    explored++;
    if (n.pruned) {
      pruned++;
    }
  }
  return std::make_pair(explored, pruned);
}

SK_RIW_SOLVER_TEMPLATE_DECL
std::size_t SK_RIW_SOLVER_CLASS::get_nb_rollouts() const {
  return _nb_rollouts;
}

SK_RIW_SOLVER_TEMPLATE_DECL
std::list<typename SK_RIW_SOLVER_CLASS::Action>
SK_RIW_SOLVER_CLASS::action_prefix() const {
  return _rollout_policy.action_prefix();
}

SK_RIW_SOLVER_TEMPLATE_DECL
typename MapTypeDeducer<
    typename SK_RIW_SOLVER_CLASS::State,
    std::pair<typename SK_RIW_SOLVER_CLASS::Action, double>>::Map
SK_RIW_SOLVER_CLASS::policy() {
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map p;
  for (auto &n : _graph) {
    if (n.best_action != nullptr) {
      p.insert(std::make_pair(n.state,
                              std::make_pair(*n.best_action, (double)n.value)));
    }
  }
  return p;
}

SK_RIW_SOLVER_TEMPLATE_DECL
void SK_RIW_SOLVER_CLASS::compute_reachable_subgraph(
    Node &node, std::unordered_set<Node *> &subgraph) {
  std::unordered_set<Node *> frontier;
  frontier.insert(&node);
  subgraph.insert(&node);
  while (!frontier.empty()) {
    std::unordered_set<Node *> new_frontier;
    for (auto &n : frontier) {
      if (n) {
        for (auto &child : n->children) {
          if (subgraph.find(std::get<2>(child)) == subgraph.end()) {
            new_frontier.insert(std::get<2>(child));
            subgraph.insert(std::get<2>(child));
          }
        }
      }
    }
    frontier = new_frontier;
  }
}

SK_RIW_SOLVER_TEMPLATE_DECL
void SK_RIW_SOLVER_CLASS::update_graph(
    std::unordered_set<Node *> &root_subgraph,
    std::unordered_set<Node *> &child_subgraph) {
  std::unordered_set<Node *> removed_subgraph;
  // First pass: look for nodes in root_subgraph but not child_subgraph and
  // remove those nodes from their children's parents Don't actually remove
  // those nodes in the first pass otherwise some children to remove won't exist
  // anymore when looking for their parents
  std::unordered_set<Node *> frontier;
  for (auto &n : root_subgraph) {
    if (n) {
      if (_online_node_garbage &&
          (child_subgraph.find(n) == child_subgraph.end())) {
        for (auto &child : n->children) {
          if (std::get<2>(child)) {
            std::get<2>(child)->parents.erase(n);
          }
        }
        removed_subgraph.insert(n);
      } else {
        n->depth -= 1;
        // if (n->solved) {
        if (n->children.empty()) {
          frontier.insert(n);
        }
      }
    }
  }
  // Second pass: actually remove nodes in root_subgraph but not in
  // child_subgraph
  for (auto &n : removed_subgraph) {
    _graph.erase(Node(n->state, _domain, _state_features, nullptr));
  }
  // Third pass: recompute fscores
  backup_values(frontier);
}

SK_RIW_SOLVER_TEMPLATE_DECL
void SK_RIW_SOLVER_CLASS::backup_values(std::unordered_set<Node *> &frontier) {
  std::size_t depth = 0; // used to prevent infinite loop in case of cycles
  for (auto &n : frontier) {
    _execution_policy.protect(
        [this, &n]() {
          if (n->pruned) {
            n->value = 0;
            for (std::size_t d = 0; d < (_max_depth - n->depth); d++) {
              n->value = _min_reward + (_discount * (n->value));
            }
          }
        },
        n->mutex);
  }

  while (!frontier.empty() && depth <= _max_depth) {
    depth += 1;
    std::unordered_set<Node *> new_frontier;
    for (auto &n : frontier) {
      update_frontier(new_frontier, n);
    }
    frontier = new_frontier;
  }
}

SK_RIW_SOLVER_TEMPLATE_DECL
template <typename TTexecution_policy>
struct SK_RIW_SOLVER_CLASS::UpdateFrontierImplementation<
    TTexecution_policy,
    typename std::enable_if<
        std::is_same<TTexecution_policy, SequentialExecution>::value>::type> {
  static void update_frontier(TTexecution_policy &execution_policy,
                              const double &discount,
                              std::unordered_set<Node *> &new_frontier,
                              Node *n) {
    for (auto &p : n->parents) {
      p->solved = true;
      p->value = -std::numeric_limits<double>::max();
      p->best_action = nullptr;
      for (auto &nn : p->children) {
        p->solved = p->solved && std::get<2>(nn) && std::get<2>(nn)->solved;
        if (std::get<2>(nn)) {
          double tentative_value =
              std::get<1>(nn) + (discount * std::get<2>(nn)->value);
          if (p->value < tentative_value) {
            p->value = tentative_value;
            p->best_action = &std::get<0>(nn);
          }
        }
      }
      new_frontier.insert(p);
    }
  }
};

SK_RIW_SOLVER_TEMPLATE_DECL
template <typename TTexecution_policy>
struct SK_RIW_SOLVER_CLASS::UpdateFrontierImplementation<
    TTexecution_policy,
    typename std::enable_if<
        std::is_same<TTexecution_policy, ParallelExecution>::value>::type> {
  static void update_frontier(TTexecution_policy &execution_policy,
                              const double &discount,
                              std::unordered_set<Node *> &new_frontier,
                              Node *n) {
    std::list<Node *> parents;
    execution_policy.protect(
        [&n, &parents]() {
          std::copy(n->parents.begin(), n->parents.end(),
                    std::back_inserter(parents));
        },
        n->mutex);
    for (auto &p : parents) {
      p->solved = true;
      p->value = -std::numeric_limits<double>::max();
      p->best_action = nullptr;
      execution_policy.protect(
          [&discount, &p]() {
            for (auto &nn : p->children) {
              p->solved =
                  p->solved && std::get<2>(nn) && std::get<2>(nn)->solved;
              if (std::get<2>(nn)) {
                double tentative_value =
                    std::get<1>(nn) + (discount * std::get<2>(nn)->value);
                if (p->value < tentative_value) {
                  p->value = tentative_value;
                  p->best_action = &std::get<0>(nn);
                }
              }
            }
          },
          p->mutex);
      new_frontier.insert(p);
    }
  }
};

SK_RIW_SOLVER_TEMPLATE_DECL
void SK_RIW_SOLVER_CLASS::update_frontier(
    std::unordered_set<Node *> &new_frontier, Node *n) {
  UpdateFrontierImplementation<ExecutionPolicy>::update_frontier(
      _execution_policy, _discount, new_frontier, n);
}

// === RIWSolver::Node implementation ===

SK_RIW_SOLVER_TEMPLATE_DECL
SK_RIW_SOLVER_CLASS::Node::Node(
    const State &s, Domain &d,
    const std::function<std::unique_ptr<FeatureVector>(
        Domain &, const State &, const std::size_t *)> &state_features,
    const std::size_t *thread_id)
    : state(s), value(-std::numeric_limits<double>::max()),
      depth(std::numeric_limits<std::size_t>::max()),
      novelty(std::numeric_limits<std::size_t>::max()), best_action(nullptr),
      terminal(false), pruned(false), solved(false) {
  features = state_features(d, s, thread_id);
}

SK_RIW_SOLVER_TEMPLATE_DECL
SK_RIW_SOLVER_CLASS::Node::Node(const Node &n)
    : state(n.state),
      features(
          std::move(const_cast<std::unique_ptr<FeatureVector> &>(n.features))),
      children(n.children), parents(n.parents), value((double)n.value),
      depth((std::size_t)n.depth), novelty((std::size_t)n.novelty),
      best_action(n.best_action), terminal((bool)n.terminal),
      pruned((bool)n.pruned), solved((bool)n.solved) {}

SK_RIW_SOLVER_TEMPLATE_DECL
const typename SK_RIW_SOLVER_CLASS::HashingPolicy::Key &
SK_RIW_SOLVER_CLASS::Node::Key::operator()(const Node &n) const {
  return HashingPolicy::get_key(n);
}

// === RIWSolver::WidthSolver implementation ===

SK_RIW_SOLVER_TEMPLATE_DECL
SK_RIW_SOLVER_CLASS::WidthSolver::WidthSolver(
    RIWSolver &parent_solver, Domain &domain,
    const StateFeatureFunctor &state_features, const atomic_size_t &time_budget,
    const atomic_size_t &rollout_budget, const atomic_size_t &max_depth,
    const atomic_double &exploration,
    const atomic_size_t &epsilon_moving_average_window,
    const atomic_double &epsilon, atomic_double &epsilon_moving_average,
    std::list<double> &epsilons, const atomic_double &discount,
    atomic_double &min_reward, const atomic_size_t &width, Graph &graph,
    RolloutPolicy &rollout_policy, ExecutionPolicy &execution_policy,
    std::mt19937 &gen, typename ExecutionPolicy::Mutex &gen_mutex,
    typename ExecutionPolicy::Mutex &time_mutex,
    typename ExecutionPolicy::Mutex &epsilons_protect,
    const atomic_bool &debug_logs, const WatchdogFunctor &watchdog)
    : _parent_solver(parent_solver), _domain(domain),
      _state_features(state_features), _time_budget(time_budget),
      _rollout_budget(rollout_budget), _max_depth(max_depth),
      _exploration(exploration),
      _epsilon_moving_average_window(epsilon_moving_average_window),
      _epsilon(epsilon), _epsilon_moving_average(epsilon_moving_average),
      _epsilons(epsilons), _discount(discount), _min_reward(min_reward),
      _min_reward_changed(false), _width(width), _graph(graph),
      _rollout_policy(rollout_policy), _execution_policy(execution_policy),
      _gen(gen), _gen_mutex(gen_mutex), _time_mutex(time_mutex),
      _epsilons_protect(epsilons_protect), _debug_logs(debug_logs),
      _watchdog(watchdog) {}

SK_RIW_SOLVER_TEMPLATE_DECL
bool SK_RIW_SOLVER_CLASS::WidthSolver::solve(
    const State &s,
    const std::chrono::time_point<std::chrono::high_resolution_clock>
        &start_time,
    atomic_size_t &nb_rollouts, TupleVector &feature_tuples) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() + " RIW(" +
                 StringConverter::from(_width) + ") solver from state " +
                 s.print());
    auto local_start_time = std::chrono::high_resolution_clock::now();

    // Clear the solved bits
    // /!\ 'solved' bit set to 1 in RIW even if no solution found with previous
    // width so we need to clear all the bits
    std::for_each(_graph.begin(), _graph.end(), [](const Node &n) {
      // we don't change the real key (Node::state) so we are safe
      const_cast<Node &>(n).solved = false;
      const_cast<Node &>(n).pruned = false;
    });

    // Create the root node containing the given state s
    auto si = _graph.emplace(Node(s, _domain, _state_features, nullptr));
    Node &root_node = const_cast<Node &>(*(
        si.first)); // we won't change the real key (Node::state) so we are safe
    root_node.depth = 0;
    atomic_bool states_pruned(false);
    atomic_bool reached_end_of_trajectory_once(false);

    // Vector of sets of state feature tuples generated so far, for each w <=
    // _width
    if (feature_tuples.size() < _width) {
      feature_tuples.push_back(typename TupleVector::value_type());
    }
    novelty(feature_tuples, root_node,
            true); // initialize feature_tuples with the root node's bits

    boost::integer_range<std::size_t> parallel_rollouts(
        0, _domain.get_parallel_capacity());
    atomic_size_t etime = 0;
    atomic_size_t epsilons_size = 0;

    std::for_each(
        ExecutionPolicy::policy, parallel_rollouts.begin(),
        parallel_rollouts.end(),
        [this, &start_time, &root_node, &feature_tuples, &nb_rollouts,
         &states_pruned, &reached_end_of_trajectory_once, &etime,
         &epsilons_size](const std::size_t &thread_id) {
          // Start rollouts
          do {
            double root_node_record_value = root_node.value;
            rollout(root_node, feature_tuples, nb_rollouts, states_pruned,
                    reached_end_of_trajectory_once, start_time, &thread_id);
            epsilons_size = update_epsilon_moving_average(
                root_node, root_node_record_value);
          } while (_watchdog(etime = elapsed_time(start_time), nb_rollouts,
                             root_node.value,
                             (epsilons_size >= _epsilon_moving_average_window)
                                 ? (double)_epsilon_moving_average
                                 : std::numeric_limits<double>::infinity()) &&
                   !root_node.solved && (etime < _time_budget) &&
                   (nb_rollouts < _rollout_budget) &&
                   (_epsilon_moving_average > _epsilon));
        });

    if (_debug_logs)
      Logger::debug(
          "time budget: " + StringConverter::from(elapsed_time(start_time)) +
          " ms, rollout budget: " + StringConverter::from(nb_rollouts) +
          ", states pruned: " + StringConverter::from(states_pruned));

    if (!_watchdog(etime = elapsed_time(start_time), nb_rollouts,
                   root_node.value,
                   (epsilons_size >= _epsilon_moving_average_window)
                       ? (double)_epsilon_moving_average
                       : std::numeric_limits<double>::infinity()) &&
        etime < _time_budget && nb_rollouts < _rollout_budget &&
        !reached_end_of_trajectory_once && states_pruned) {
      Logger::info("RIW(" + StringConverter::from(_width) +
                   ") could not find a solution from state " + s.print());
      return false;
    } else {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          end_time - local_start_time)
                          .count();
      Logger::info("RIW(" + StringConverter::from(_width) +
                   ") finished to solve from state " + s.print() + " in " +
                   StringConverter::from((double)duration / (double)1e9) +
                   " seconds.");
      return true;
    }
  } catch (const std::exception &e) {
    Logger::error("RIW(" + StringConverter::from(_width) +
                  ") failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_RIW_SOLVER_TEMPLATE_DECL
void SK_RIW_SOLVER_CLASS::WidthSolver::rollout(
    Node &root_node, TupleVector &feature_tuples, atomic_size_t &nb_rollouts,
    atomic_bool &states_pruned, atomic_bool &reached_end_of_trajectory_once,
    const std::chrono::time_point<std::chrono::high_resolution_clock>
        &start_time,
    const std::size_t *thread_id) {
  // Start new rollout
  nb_rollouts += 1;
  Node *current_node = &root_node;

  if (_debug_logs)
    Logger::debug("New rollout" + ExecutionPolicy::print_thread() +
                  " from state: " + current_node->state.print() +
                  ", depth=" + StringConverter::from(current_node->depth) +
                  ", value=" + StringConverter::from(current_node->value) +
                  ExecutionPolicy::print_thread());
  _rollout_policy.init_rollout(_domain, thread_id);
  bool break_loop = false;

  while (!(current_node->solved) && !break_loop) {

    std::vector<std::size_t> unsolved_children;
    std::vector<double> probabilities;

    _execution_policy.protect(
        [this, &current_node, &unsolved_children, &probabilities,
         &thread_id]() {
          if (_debug_logs)
            Logger::debug(
                "Current state" + ExecutionPolicy::print_thread() + ": " +
                current_node->state.print() +
                ", depth=" + StringConverter::from(current_node->depth) +
                ", value=" + StringConverter::from(current_node->value) +
                ExecutionPolicy::print_thread());

          if (current_node->children.empty()) {
            // Generate applicable actions
            auto applicable_actions =
                _domain.get_applicable_actions(current_node->state, thread_id)
                    .get_elements();
            std::for_each(applicable_actions.begin(), applicable_actions.end(),
                          [&current_node](auto a) {
                            current_node->children.push_back(
                                std::make_tuple(a, 0, nullptr));
                          });
          }

          // Sample unsolved child
          for (std::size_t i = 0; i < current_node->children.size(); i++) {
            Node *n = std::get<2>(current_node->children[i]);
            if (!n) {
              unsolved_children.push_back(i);
              probabilities.push_back(_exploration);
            } else if (!(n->solved)) {
              unsolved_children.push_back(i);
              probabilities.push_back((1.0 - _exploration) /
                                      ((double)n->novelty));
            }
          }
        },
        current_node->mutex);

    // In parallel execution mode, child nodes can have been solved since we
    // have checked for this current node's solve bit
    if (unsolved_children.empty()) {
      if (std::is_same<ExecutionPolicy, SequentialExecution>::value) {
        throw std::runtime_error("In sequential mode, nodes labelled as "
                                 "unsolved must have unsolved children.");
      }
      current_node->solved = true;
      break;
    }

    std::size_t pick = 0;
    _execution_policy.protect(
        [this, &pick, &unsolved_children, &probabilities]() {
          pick = unsolved_children[std::discrete_distribution<>(
              probabilities.begin(), probabilities.end())(_gen)];
        },
        _gen_mutex);
    bool new_node = false;

    if (fill_child_node(current_node, pick, new_node,
                        thread_id)) { // terminal state
      if (_debug_logs) {
        _execution_policy.protect(
            [&current_node]() {
              Logger::debug(
                  "Found" + ExecutionPolicy::print_thread() +
                  " a terminal state: " + current_node->state.print() +
                  ", depth=" + StringConverter::from(current_node->depth) +
                  ", value=" + StringConverter::from(current_node->value) +
                  ExecutionPolicy::print_thread());
            },
            current_node->mutex);
      }
      update_node(*current_node, true);
      reached_end_of_trajectory_once = true;
      break_loop = true;
    } else if (!novelty(feature_tuples, *current_node,
                        new_node)) { // no new tuple or not reached with lower
                                     // depth => terminal node
      if (_debug_logs) {
        _execution_policy.protect(
            [&current_node]() {
              Logger::debug(
                  "Pruning" + ExecutionPolicy::print_thread() +
                  " state: " + current_node->state.print() +
                  ", depth=" + StringConverter::from(current_node->depth) +
                  ", value=" + StringConverter::from(current_node->value) +
                  ExecutionPolicy::print_thread());
            },
            current_node->mutex);
      }
      states_pruned = true;
      current_node->pruned = true;
      // /!\ current_node can become solved with some unsolved children in case
      // it was already visited and novel but now some of its features are
      // reached with lower depth
      update_node(*current_node, true);
      break_loop = true;
    } else if (current_node->depth >= _max_depth) {
      if (_debug_logs) {
        _execution_policy.protect(
            [&current_node]() {
              Logger::debug(
                  "Max depth reached" + ExecutionPolicy::print_thread() +
                  "in state: " + current_node->state.print() +
                  ", depth=" + StringConverter::from(current_node->depth) +
                  ", value=" + StringConverter::from(current_node->value) +
                  ExecutionPolicy::print_thread());
            },
            current_node->mutex);
      }
      update_node(*current_node, true);
      reached_end_of_trajectory_once = true;
      break_loop = true;
    } else if (elapsed_time(start_time) >= _time_budget) {
      if (_debug_logs) {
        _execution_policy.protect(
            [&current_node]() {
              Logger::debug(
                  "Time budget consumed" + ExecutionPolicy::print_thread() +
                  " in state: " + current_node->state.print() +
                  ", depth=" + StringConverter::from(current_node->depth) +
                  ", value=" + StringConverter::from(current_node->value) +
                  ExecutionPolicy::print_thread());
            },
            current_node->mutex);
      }
      // next test: unexpanded node considered as a temporary (i.e. not solved)
      // terminal node don't backup expanded node at this point otherwise the
      // fscore initialization in update_node is wrong!
      bool current_node_no_children = false;
      _execution_policy.protect(
          [&current_node_no_children, &current_node]() {
            current_node_no_children = current_node->children.empty();
          },
          current_node->mutex);
      if (current_node_no_children) {
        update_node(*current_node, false);
      }
      break_loop = true;
    }
  }
}

SK_RIW_SOLVER_TEMPLATE_DECL
bool SK_RIW_SOLVER_CLASS::WidthSolver::novelty(TupleVector &feature_tuples,
                                               Node &n, bool nn) const {
  // feature_tuples is a set of state variable combinations of size _width
  std::size_t nov = n.features->size() + 1;
  const FeatureVector &state_features = *n.features;
  bool novel_depth = false;

  for (std::size_t k = 1;
       k <= std::min((std::size_t)_width, (std::size_t)state_features.size());
       k++) {
    // we must recompute combinations from previous width values just in case
    // this state would be visited for the first time across width iterations
    generate_tuples(
        k, state_features.size(),
        [this, &state_features, &feature_tuples, &k, &novel_depth, &n, &nn,
         &nov](TupleType &cv) {
          for (auto &e : cv) {
            e.second = state_features[e.first];
          }
          _execution_policy.protect([&feature_tuples, &cv, &k, &novel_depth, &n,
                                     &nn, &nov]() -> void {
            auto it = feature_tuples[k - 1].insert(
                std::make_pair(cv, (std::size_t)n.depth));
            novel_depth = novel_depth || it.second ||
                          (nn && (it.first->second > n.depth)) ||
                          (!nn && (it.first->second == n.depth));
            it.first->second = std::min(it.first->second, (std::size_t)n.depth);
            if (it.second) {
              nov = std::min(nov, k);
            }
          });
        });
  }
  n.novelty = nov;
  if (_debug_logs)
    Logger::debug("Novelty: " + StringConverter::from(nov));
  if (_debug_logs)
    Logger::debug("Novelty depth check: " + StringConverter::from(novel_depth));
  return novel_depth;
}

SK_RIW_SOLVER_TEMPLATE_DECL
void SK_RIW_SOLVER_CLASS::WidthSolver::generate_tuples(
    const std::size_t &k, const std::size_t &n,
    const std::function<void(TupleType &)> &f) const {
  TupleType cv(k); // one combination (the first one)
  for (std::size_t i = 0; i < k; i++) {
    cv[i].first = i;
  }
  f(cv);
  bool more_combinations = true;
  while (more_combinations) {
    more_combinations = false;
    // find the rightmost element that has not yet reached its highest possible
    // value
    for (std::size_t i = k; i > 0; i--) {
      if (cv[i - 1].first < n - k + i - 1) {
        // once finding this element, we increment it by 1,
        // and assign the lowest valid value to all subsequent elements
        cv[i - 1].first++;
        for (std::size_t j = i; j < k; j++) {
          cv[j].first = cv[j - 1].first + 1;
        }
        f(cv);
        more_combinations = true;
        break;
      }
    }
  }
}

SK_RIW_SOLVER_TEMPLATE_DECL
bool SK_RIW_SOLVER_CLASS::WidthSolver::fill_child_node(
    Node *&node, std::size_t action_number, bool &new_node,
    const std::size_t *thread_id) {
  Node *node_child = nullptr;

  _execution_policy.protect(
      [this, &node, &node_child, &action_number]() {
        if (_debug_logs)
          Logger::debug(
              "Applying " + ExecutionPolicy::print_thread() +
              " action: " + std::get<0>(node->children[action_number]).print() +
              ExecutionPolicy::print_thread());
        node_child = std::get<2>(node->children[action_number]);
      },
      node->mutex);

  if (!node_child) { // first visit
    // Sampled child has not been visited so far, so generate it
    std::unique_ptr<typename Domain::EnvironmentOutcome> outcome;
    _execution_policy.protect(
        [this, &node, &outcome, &action_number, &thread_id]() {
          outcome = std::make_unique<typename Domain::EnvironmentOutcome>(
              _rollout_policy.progress(
                  _domain, node->state,
                  std::get<0>(node->children[action_number]), thread_id));
        },
        node->mutex);

    _execution_policy.protect(
        [this, &node_child, &thread_id, &new_node, &outcome]() {
          auto i = _graph.emplace(Node(outcome->observation(), _domain,
                                       _state_features, thread_id));
          new_node = i.second;
          node_child =
              &const_cast<Node &>(*(i.first)); // we won't change the real key
                                               // (Node::state) so we are safe
        });
    Node &next_node = *node_child;
    double reward = outcome->transition_value().reward();
    _execution_policy.protect(
        [&node, &action_number, &reward, &node_child]() {
          std::get<1>(node->children[action_number]) = reward;
          std::get<2>(node->children[action_number]) = node_child;
        },
        node->mutex);
    if (reward < _min_reward) {
      _min_reward = reward;
      _min_reward_changed = true;
    }

    _execution_policy.protect(
        [this, &node, &next_node, &new_node, &outcome]() {
          next_node.parents.insert(node);
          if (new_node) {
            if (_debug_logs)
              Logger::debug(
                  "Exploring" + ExecutionPolicy::print_thread() +
                  " new outcome: " + next_node.state.print() +
                  ", depth=" + StringConverter::from(next_node.depth) +
                  ", value=" + StringConverter::from(next_node.value) +
                  ExecutionPolicy::print_thread());
            next_node.depth = node->depth + 1;
            node = &next_node;
            node->terminal = outcome->termination();
            if (node->terminal && _min_reward > 0.0) {
              _min_reward = 0.0;
              _min_reward_changed = true;
            }
          } else { // outcome already explored
            if (_debug_logs)
              Logger::debug(
                  "Exploring" + ExecutionPolicy::print_thread() +
                  " known outcome: " + next_node.state.print() +
                  ", depth=" + StringConverter::from(next_node.depth) +
                  ", value=" + StringConverter::from(next_node.value) +
                  ExecutionPolicy::print_thread());
            next_node.depth = std::min((std::size_t)next_node.depth,
                                       (std::size_t)node->depth + 1);
            node = &next_node;
          }
        },
        next_node.mutex);
  } else { // second visit, unsolved child
    new_node = false;
    // call the simulator to be coherent with the new current node /!\ Assumes
    // deterministic environment!
    _execution_policy.protect(
        [this, &node, &action_number, &thread_id]() {
          _rollout_policy.advance(_domain, node->state,
                                  std::get<0>(node->children[action_number]),
                                  false, thread_id);
        },
        node->mutex);
    Node &next_node = *node_child;
    next_node.depth =
        std::min((std::size_t)next_node.depth, (std::size_t)node->depth + 1);
    node = node_child;
    if (_debug_logs) {
      _execution_policy.protect(
          [&node]() {
            Logger::debug("Exploring" + ExecutionPolicy::print_thread() +
                          " known outcome: " + node->state.print() +
                          ", depth=" + StringConverter::from(node->depth) +
                          ", value=" + StringConverter::from(node->value) +
                          ExecutionPolicy::print_thread());
          },
          next_node.mutex);
    }
  }
  return (node->terminal) || (node->solved); // consider solved node as terminal
                                             // to stop current rollout
}

SK_RIW_SOLVER_TEMPLATE_DECL
void SK_RIW_SOLVER_CLASS::WidthSolver::update_node(Node &node, bool solved) {
  node.solved = solved;
  node.value = 0;
  std::unordered_set<Node *> frontier;
  if (_min_reward_changed) {
    // need for backtracking all leaf nodes in the graph
    _execution_policy.protect([this, &frontier]() {
      for (auto &n : _graph) {
        _execution_policy.protect(
            [&n, &frontier]() {
              if (n.children.empty()) {
                frontier.insert(
                    &const_cast<Node &>(n)); // we won't change the real key
                                             // (Node::state) so we are safe
              }
            },
            n.mutex);
      }
    });
    _min_reward_changed = false;
  } else {
    frontier.insert(&node);
  }
  _parent_solver.backup_values(frontier);
}

SK_RIW_SOLVER_TEMPLATE_DECL
std::size_t SK_RIW_SOLVER_CLASS::WidthSolver::update_epsilon_moving_average(
    const Node &node, const double &node_record_value) {
  std::size_t epsilons_size = 0;
  if (_epsilon_moving_average_window > 0) {
    double current_epsilon = std::fabs(node_record_value - node.value);
    _execution_policy.protect(
        [this, &epsilons_size, &current_epsilon]() {
          if (_epsilons.size() < _epsilon_moving_average_window) {
            _epsilon_moving_average =
                ((double)_epsilon_moving_average) +
                (current_epsilon / ((double)_epsilon_moving_average_window));
          } else {
            _epsilon_moving_average =
                ((double)_epsilon_moving_average) +
                ((current_epsilon - _epsilons.front()) /
                 ((double)_epsilon_moving_average_window));
            _epsilons.pop_front();
          }
          _epsilons.push_back(current_epsilon);
          epsilons_size = _epsilons.size();
        },
        _epsilons_protect);
  }
  return epsilons_size;
}

SK_RIW_SOLVER_TEMPLATE_DECL
std::size_t SK_RIW_SOLVER_CLASS::WidthSolver::elapsed_time(
    const std::chrono::time_point<std::chrono::high_resolution_clock>
        &start_time) {
  std::size_t milliseconds_duration;
  _execution_policy.protect(
      [&milliseconds_duration, &start_time]() {
        milliseconds_duration = static_cast<std::size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time)
                .count());
      },
      _time_mutex);
  return milliseconds_duration;
}

} // namespace skdecide

#endif // SKDECIDE_RIW_IMPL_HH
