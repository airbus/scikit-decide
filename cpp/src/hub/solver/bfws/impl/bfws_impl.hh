/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_BFWS_IMPL_HH
#define SKDECIDE_BFWS_IMPL_HH

#include <queue>

#include "utils/string_converter.hh"
#include "utils/logging.hh"

namespace skdecide {

// === DomainStateHash implementation ===

#define SK_BFWS_DOMAIN_STATE_HASH_TEMPLATE_DECL                                \
  template <typename Tdomain, typename Tfeature_vector>

#define SK_BFWS_DOMAIN_STATE_HASH_CLASS                                        \
  DomainStateHash<Tdomain, Tfeature_vector>

SK_BFWS_DOMAIN_STATE_HASH_TEMPLATE_DECL
template <typename Tnode>
const typename SK_BFWS_DOMAIN_STATE_HASH_CLASS::Key &
SK_BFWS_DOMAIN_STATE_HASH_CLASS::get_key(const Tnode &n) {
  return n.state;
}

SK_BFWS_DOMAIN_STATE_HASH_TEMPLATE_DECL
std::size_t
SK_BFWS_DOMAIN_STATE_HASH_CLASS::Hash::operator()(const Key &k) const {
  return typename Tdomain::State::Hash()(k);
}

SK_BFWS_DOMAIN_STATE_HASH_TEMPLATE_DECL
bool SK_BFWS_DOMAIN_STATE_HASH_CLASS::Equal::operator()(const Key &k1,
                                                        const Key &k2) const {
  return typename Tdomain::State::Equal()(k1, k2);
}

// === StateFeatureHash implementation ===

#define SK_BFWS_STATE_FEATURE_HASH_TEMPLATE_DECL                               \
  template <typename Tdomain, typename Tfeature_vector>

#define SK_BFWS_STATE_FEATURE_HASH_CLASS                                       \
  StateFeatureHash<Tdomain, Tfeature_vector>

SK_BFWS_STATE_FEATURE_HASH_TEMPLATE_DECL
template <typename Tnode>
const typename SK_BFWS_STATE_FEATURE_HASH_CLASS::Key &
SK_BFWS_STATE_FEATURE_HASH_CLASS::get_key(const Tnode &n) {
  return *n.features;
}

SK_BFWS_STATE_FEATURE_HASH_TEMPLATE_DECL
std::size_t
SK_BFWS_STATE_FEATURE_HASH_CLASS::Hash::operator()(const Key &k) const {
  std::size_t seed = 0;
  for (std::size_t i = 0; i < k.size(); i++) {
    boost::hash_combine(seed, k[i]);
  }
  return seed;
}

SK_BFWS_STATE_FEATURE_HASH_TEMPLATE_DECL
bool SK_BFWS_STATE_FEATURE_HASH_CLASS::Equal::operator()(const Key &k1,
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

// === BFWSSolver implementation ===

#define SK_BFWS_SOLVER_TEMPLATE_DECL                                           \
  template <typename Tdomain, typename Tfeature_vector,                        \
            template <typename...> class Thashing_policy,                      \
            typename Texecution_policy>

#define SK_BFWS_SOLVER_CLASS                                                   \
  BFWSSolver<Tdomain, Tfeature_vector, Thashing_policy, Texecution_policy>

SK_BFWS_SOLVER_TEMPLATE_DECL
SK_BFWS_SOLVER_CLASS::BFWSSolver(Domain &domain,
                                 const GoalCheckerFunctor &goal_checker,
                                 const StateFeatureFunctor &state_features,
                                 const HeuristicFunctor &heuristic,
                                 const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _goal_checker(goal_checker),
      _state_features(state_features), _heuristic(heuristic),
      _callback(callback), _verbose(verbose) {

  if (verbose) {
    Logger::check_level(logging::debug, "algorithm BFWS");
  }
}

SK_BFWS_SOLVER_TEMPLATE_DECL
void SK_BFWS_SOLVER_CLASS::clear() {
  _open_queue = PriorityQueue();
  _graph.clear();
}

SK_BFWS_SOLVER_TEMPLATE_DECL
void SK_BFWS_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " BFWS solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();

    // Map from heuristic values to set of state features with that given
    // heuristic value whose value has changed at least once since the beginning
    // of the search (stored by their index and value)
    PairMap heuristic_features_map;

    // Create the root node containing the given state s
    auto si = _graph.emplace(Node(s, _domain, _state_features));
    if (si.first->solved ||
        _goal_checker(_domain,
                      s)) { // problem already solved from this state
                            // (was present in _graph and already solved)
      return;
    }
    Node &root_node = const_cast<Node &>(*(
        si.first)); // we won't change the real key (Node::state) so we are safe
    root_node.gscore = 0;
    root_node.heuristic = _heuristic(_domain, root_node.state).cost();
    root_node.novelty =
        novelty(heuristic_features_map, root_node.heuristic, root_node);

    // Priority queue used to sort non-goal unsolved tip nodes by increasing
    // cost-to-go values (so-called OPEN container)
    _open_queue = PriorityQueue();
    _open_queue.push(&root_node);

    // Set of states that have already been explored
    std::unordered_set<Node *> closed_set;

    while (!_open_queue.empty() && !_callback(*this, _domain)) {
      auto best_tip_node = _open_queue.top();
      _open_queue.pop();

      // Check that the best tip node has not already been closed before
      // (since this implementation's open_queue does not check for element
      // uniqueness, it can contain many copies of the same node pointer that
      // could have been closed earlier)
      if (closed_set.find(best_tip_node) !=
          closed_set
              .end()) { // this implementation's open_queue can contain several
        continue;
      }

      if (_verbose)
        Logger::debug("Current best tip node (h=" +
                      StringConverter::from(best_tip_node->heuristic) +
                      ", n=" + StringConverter::from(best_tip_node->novelty) +
                      "): " + best_tip_node->state.print());

      if (_goal_checker(_domain, best_tip_node->state) ||
          best_tip_node->solved) {
        if (_verbose)
          Logger::debug("Found a goal or previously solved state: " +
                        best_tip_node->state.print());
        auto current_node = best_tip_node;
        if (!(best_tip_node->solved)) {
          current_node->fscore = current_node->gscore;
        } // goal state

        while (current_node != &root_node) {
          Node *parent_node = std::get<0>(current_node->best_parent);
          parent_node->best_action = std::make_pair(
              &std::get<1>(current_node->best_parent), current_node);
          // if everything went fine we should have parent_node->fscore ==
          // current_node->fscore but let's be conservative here just in case
          parent_node->fscore = parent_node->gscore +
                                std::get<2>(current_node->best_parent) +
                                current_node->fscore - current_node->gscore;
          parent_node->solved = true;
          current_node = parent_node;
        }

        Logger::info(
            "BFWS finished to solve from state " + s.print() + " in " +
            StringConverter::from((double)get_solving_time() / (double)1e6) +
            " seconds.");
        return;
      }

      closed_set.insert(best_tip_node);

      // Expand best tip node
      auto applicable_actions =
          _domain.get_applicable_actions(best_tip_node->state).get_elements();
      std::for_each(
          ExecutionPolicy::policy, applicable_actions.begin(),
          applicable_actions.end(),
          [this, &best_tip_node, &closed_set, &heuristic_features_map](auto a) {
            if (_verbose)
              Logger::debug("Current expanded action: " + a.print() +
                            ExecutionPolicy::print_thread());
            auto next_state = _domain.get_next_state(best_tip_node->state, a);
            if (_verbose)
              Logger::debug("Exploring next state " + next_state.print() +
                            ExecutionPolicy::print_thread());
            std::pair<typename Graph::iterator, bool> i;
            _execution_policy.protect([this, &i, &next_state] {
              i = _graph.emplace(Node(next_state, _domain, _state_features));
            });
            Node &neighbor = const_cast<Node &>(
                *(i.first)); // we won't change the real key (StateNode::state)
                             // so we are safe

            if (closed_set.find(&neighbor) != closed_set.end()) {
              // Ignore the neighbor which is already evaluated
              return;
            }

            double transition_cost =
                _domain
                    .get_transition_value(best_tip_node->state, a,
                                          neighbor.state)
                    .cost();
            double tentative_gscore = best_tip_node->gscore + transition_cost;

            if ((i.second) || (tentative_gscore < neighbor.gscore)) {
              if (_verbose)
                Logger::debug("New gscore: " +
                              StringConverter::from(best_tip_node->gscore) +
                              "+" + StringConverter::from(transition_cost) +
                              "=" + StringConverter::from(tentative_gscore) +
                              ExecutionPolicy::print_thread());
              neighbor.gscore = tentative_gscore;
              neighbor.best_parent =
                  std::make_tuple(best_tip_node, a, transition_cost);
            }

            neighbor.heuristic = _heuristic(_domain, neighbor.state).cost();
            if (_verbose)
              Logger::debug(
                  "Heuristic: " + StringConverter::from(neighbor.heuristic) +
                  ExecutionPolicy::print_thread());
            _execution_policy.protect([this, &heuristic_features_map,
                                       &neighbor] {
              neighbor.novelty = this->novelty(heuristic_features_map,
                                               neighbor.heuristic, neighbor);
              _open_queue.push(&neighbor);
              if (_verbose)
                Logger::debug(
                    "Novelty: " + StringConverter::from(neighbor.novelty) +
                    ExecutionPolicy::print_thread());
            });
          });
    }

    Logger::info("BFWS could not find a solution from state " + s.print());
  } catch (const std::exception &e) {
    Logger::error("BFWS failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_BFWS_SOLVER_TEMPLATE_DECL
bool SK_BFWS_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(Node(s, _domain, _state_features));
  if ((si == _graph.end()) || (si->best_action.first == nullptr) ||
      (si->solved == false)) {
    return false;
  } else {
    return true;
  }
}

SK_BFWS_SOLVER_TEMPLATE_DECL
const typename SK_BFWS_SOLVER_CLASS::Action &
SK_BFWS_SOLVER_CLASS::get_best_action(const State &s) const {
  auto si = _graph.find(Node(s, _domain, _state_features));
  if ((si == _graph.end()) || (si->best_action.first == nullptr)) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return *(si->best_action.first);
}

SK_BFWS_SOLVER_TEMPLATE_DECL
typename SK_BFWS_SOLVER_CLASS::Value
SK_BFWS_SOLVER_CLASS::get_best_value(const State &s) const {
  auto si = _graph.find(Node(s, _domain, _state_features));
  if (si == _graph.end()) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  Value val;
  val.cost(si->fscore - si->gscore);
  return val;
}

SK_BFWS_SOLVER_TEMPLATE_DECL
std::size_t SK_BFWS_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_BFWS_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_BFWS_SOLVER_CLASS::State>::Set
SK_BFWS_SOLVER_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored_states;
  for (const auto &s : _graph) {
    explored_states.insert(s.state);
  }
  return explored_states;
}

SK_BFWS_SOLVER_TEMPLATE_DECL std::size_t
SK_BFWS_SOLVER_CLASS::get_nb_tip_states() const {
  return _open_queue.size();
}

SK_BFWS_SOLVER_TEMPLATE_DECL
const typename SK_BFWS_SOLVER_CLASS::State &
SK_BFWS_SOLVER_CLASS::get_top_tip_state() const {
  if (_open_queue.empty()) {
    Logger::error(
        "SKDECIDE exception: no top tip state (empty priority queue)");
    throw std::runtime_error(
        "SKDECIDE exception: no top tip state (empty priority queue)");
  }
  return _open_queue.top()->state;
}

SK_BFWS_SOLVER_TEMPLATE_DECL
std::size_t SK_BFWS_SOLVER_CLASS::get_solving_time() const {
  std::size_t milliseconds_duration;
  milliseconds_duration = static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
  return milliseconds_duration;
}

SK_BFWS_SOLVER_TEMPLATE_DECL
std::vector<std::tuple<typename SK_BFWS_SOLVER_CLASS::State,
                       typename SK_BFWS_SOLVER_CLASS::Action,
                       typename SK_BFWS_SOLVER_CLASS::Value>>
SK_BFWS_SOLVER_CLASS::get_plan(
    const typename SK_BFWS_SOLVER_CLASS::State &from_state) const {
  std::vector<std::tuple<State, Action, Value>> p;
  auto si = _graph.find(Node(from_state, _domain, _state_features));
  if (si == _graph.end()) {
    Logger::warn("SKDECIDE warning: no plan found starting in state " +
                 from_state.print());
    return p;
  }
  const Node *cur_node = &(*si);
  std::unordered_set<const Node *> plan_nodes;
  plan_nodes.insert(cur_node);
  while (!_goal_checker(_domain, cur_node->state) &&
         cur_node->best_action.first != nullptr) {
    Value val;
    val.cost(cur_node->best_action.second->gscore - cur_node->gscore);
    p.push_back(
        std::make_tuple(cur_node->state, *(cur_node->best_action.first), val));
    cur_node = cur_node->best_action.second;
    if (!plan_nodes.insert(cur_node).second) {
      Logger::error("SKDECIDE exception: cycle detected in the solution plan "
                    "starting in state " +
                    from_state.print());
      throw std::runtime_error("SKDECIDE exception: cycle detected in the "
                               "solution plan starting in state " +
                               from_state.print());
    }
  }
  return p;
}

SK_BFWS_SOLVER_TEMPLATE_DECL typename MapTypeDeducer<
    typename SK_BFWS_SOLVER_CLASS::State,
    std::pair<typename SK_BFWS_SOLVER_CLASS::Action,
              typename SK_BFWS_SOLVER_CLASS::Value>>::Map
SK_BFWS_SOLVER_CLASS::get_policy() const {
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map p;
  for (auto &n : _graph) {
    if (n.best_action.first != nullptr) {
      Value val;
      val.cost(n.fscore - n.gscore);
      p.insert(
          std::make_pair(n.state, std::make_pair(*(n.best_action.first), val)));
    }
  }
  return p;
}

SK_BFWS_SOLVER_TEMPLATE_DECL
std::size_t SK_BFWS_SOLVER_CLASS::novelty(PairMap &heuristic_features_map,
                                          const double &heuristic_value,
                                          Node &n) const {
  auto r = heuristic_features_map.emplace(
      heuristic_value, std::unordered_set<PairType, boost::hash<PairType>>());
  std::unordered_set<PairType, boost::hash<PairType>> &features =
      r.first->second;
  std::size_t nov = 0;
  const FeatureVector &state_features = *n.features;
  for (std::size_t i = 0; i < state_features.size(); i++) {
    nov += (std::size_t)features.insert(std::make_pair(i, state_features[i]))
               .second;
  }
  if (r.second) {
    nov = 0;
  } else if (nov == 0) {
    nov = n.features->size() + 1;
  }
  return nov;
}

// === BFWSSolver::Node implementation ===

SK_BFWS_SOLVER_TEMPLATE_DECL
SK_BFWS_SOLVER_CLASS::Node::Node(
    const State &s, Domain &d,
    const std::function<std::unique_ptr<FeatureVector>(
        Domain &d, const State &s)> &state_features)
    : state(s), novelty(std::numeric_limits<std::size_t>::max()),
      gscore(std::numeric_limits<double>::infinity()),
      fscore(std::numeric_limits<double>::infinity()),
      best_action({nullptr, nullptr}), solved(false) {
  features = state_features(d, s);
}

SK_BFWS_SOLVER_TEMPLATE_DECL
const typename SK_BFWS_SOLVER_CLASS::HashingPolicy::Key &
SK_BFWS_SOLVER_CLASS::Node::Key::operator()(const Node &n) const {
  return HashingPolicy::get_key(n);
}

// === AStarSolver::NodeCompare implementation ===

SK_BFWS_SOLVER_TEMPLATE_DECL
bool SK_BFWS_SOLVER_CLASS::NodeCompare::operator()(Node *&a, Node *&b) const {
  // smallest element appears at the top of the priority_queue => cost
  // optimization rank first by heuristic values then by novelty measures
  return ((a->heuristic) > (b->heuristic)) ||
         (((a->heuristic) == (b->heuristic)) && ((a->novelty) > (b->novelty)));
}

} // namespace skdecide

#endif // SKDECIDE_BFWS_IMPL_HH
