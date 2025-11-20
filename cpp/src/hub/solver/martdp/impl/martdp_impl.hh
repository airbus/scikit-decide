/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MARTDP_IMPL_HH
#define SKDECIDE_MARTDP_IMPL_HH

#include "utils/string_converter.hh"
#include "utils/logging.hh"

namespace skdecide {

// === MARTDPSolver implementation ===

#define SK_MARTDP_SOLVER_TEMPLATE_DECL template <typename Tdomain>

#define SK_MARTDP_SOLVER_CLASS MARTDPSolver<Tdomain>

SK_MARTDP_SOLVER_TEMPLATE_DECL
SK_MARTDP_SOLVER_CLASS::MARTDPSolver(
    Domain &domain, const GoalCheckerFunctor &goal_checker,
    const HeuristicFunctor &heuristic, std::size_t time_budget,
    std::size_t rollout_budget, std::size_t max_depth,
    std::size_t max_feasibility_trials, double graph_expansion_rate,
    std::size_t residual_moving_average_window, double epsilon, double discount,
    double action_choice_noise, const double &dead_end_cost,
    bool online_node_garbage, const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _time_budget(time_budget), _rollout_budget(rollout_budget),
      _max_depth(max_depth), _max_feasibility_trials(max_feasibility_trials),
      _graph_expansion_rate(graph_expansion_rate),
      _residual_moving_average_window(residual_moving_average_window),
      _epsilon(epsilon), _discount(discount), _dead_end_cost(dead_end_cost),
      _online_node_garbage(online_node_garbage), _callback(callback),
      _verbose(verbose), _residual_moving_average(0), _current_state(nullptr),
      _nb_rollouts(0), _nb_agents(0) {

  if (verbose) {
    Logger::check_level(logging::debug, "algorithm MA-RTDP");
  }

  std::random_device rd;
  _gen = std::make_unique<std::mt19937>(rd());
  _action_choice_noise_dist = std::bernoulli_distribution(action_choice_noise);
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
void SK_MARTDP_SOLVER_CLASS::clear() { _graph.clear(); }

SK_MARTDP_SOLVER_TEMPLATE_DECL
void SK_MARTDP_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running MARTDP solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();

    if (_nb_agents != s.size()) {
      // We are solving a new problem.
      _graph.clear();
      _nb_agents = s.size();
      _agents_orders.resize(_nb_agents);

      for (std::size_t a = 0; a < _nb_agents; a++) {
        _agents_orders[a].resize(_nb_agents - 1);
        for (std::size_t aa = 0; aa < _nb_agents - 1; aa++) {
          if (aa < a) {
            _agents_orders[a][aa] = aa;
          } else {
            _agents_orders[a][aa] = aa + 1;
          }
        }
      }

      _agents.clear();
      _agents.reserve(_nb_agents);

      for (auto a : s) {
        _agents.push_back(a.agent());
      }

      if (_max_feasibility_trials == 0) {
        _max_feasibility_trials = _nb_agents;
      }
    }

    auto si = _graph.emplace(s);
    StateNode &root_node = const_cast<StateNode &>(
        *(si.first)); // we won't change the real key (StateNode::state) so we
                      // are safe

    if (si.second) {
      initialize_node(root_node, _goal_checker(_domain, root_node.state));
    }

    if (root_node.all_goal) { // problem already solved from this state (was
                              // present in _graph and already solved)
      Logger::info("MARTDP finished to solve from state " + s.print() +
                   " [goal state]");
      return;
    }

    _nb_rollouts = 0;
    _residual_moving_average = 0.0;
    _residuals.clear();

    do {
      if (_verbose)
        Logger::debug("Starting rollout " +
                      StringConverter::from(_nb_rollouts));

      _nb_rollouts++;
      double root_node_record_value = root_node.all_value;
      trial(&root_node);
      update_residual_moving_average(root_node, root_node_record_value);
    } while (!_callback(*this, _domain) &&
             (get_solving_time() < _time_budget) &&
             (_nb_rollouts < _rollout_budget) &&
             (get_residual_moving_average() > _epsilon));

    Logger::info(
        "MARTDP finished to solve from state " + s.print() + " in " +
        StringConverter::from((double)get_solving_time() / (double)1e3) +
        " seconds with " + StringConverter::from(_nb_rollouts) +
        " rollouts and visited " + StringConverter::from(_graph.size()) +
        " states. ");
  } catch (const std::exception &e) {
    Logger::error("MARTDP failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
bool SK_MARTDP_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || !(si->action)) {
    return false;
  } else {
    return true;
  }
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
const typename SK_MARTDP_SOLVER_CLASS::Action &
SK_MARTDP_SOLVER_CLASS::get_best_action(const State &s) {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || !(si->action)) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  } else {
    if (_verbose) {
      std::string str = "(";
      for (const auto &o : si->action->outcomes) {
        str += "\n    " + o.first->state.print();
      }
      str += "\n)";
      Logger::debug("Best action's outcomes:\n" + str);
    }
    if (_online_node_garbage && _current_state) {
      std::unordered_set<StateNode *> root_subgraph, child_subgraph;
      compute_reachable_subgraph(_current_state, root_subgraph);
      compute_reachable_subgraph(
          const_cast<StateNode *>(&(*si)),
          child_subgraph); // we won't change the real key (StateNode::state) so
                           // we are safe
      remove_subgraph(root_subgraph, child_subgraph);
    }
    _current_state = const_cast<StateNode *>(&(
        *si)); // we won't change the real key (StateNode::state) so we are safe
    return si->action->action;
  }
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
typename SK_MARTDP_SOLVER_CLASS::Value
SK_MARTDP_SOLVER_CLASS::get_best_value(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  Value val;
  for (std::size_t a = 0; a < _nb_agents; a++) {
    val[_agents[a]].cost(si->value[a]);
  }
  return val;
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
std::size_t SK_MARTDP_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
const std::size_t &
SK_MARTDP_SOLVER_CLASS::get_state_nb_actions(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    throw std::runtime_error("SKDECIDE exception: state " + s.print() +
                             " not in the search graph");
  }
  return si->expansions_count;
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
std::size_t SK_MARTDP_SOLVER_CLASS::get_nb_rollouts() const {
  return _nb_rollouts;
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
double SK_MARTDP_SOLVER_CLASS::get_residual_moving_average() const {
  if (_residuals.size() >= _residual_moving_average_window) {
    return (double)_residual_moving_average;
  } else {
    return std::numeric_limits<double>::infinity();
  }
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
std::size_t SK_MARTDP_SOLVER_CLASS::get_solving_time() const {
  std::size_t milliseconds_duration;
  milliseconds_duration = static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
  return milliseconds_duration;
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
typename MapTypeDeducer<typename SK_MARTDP_SOLVER_CLASS::State,
                        std::pair<typename SK_MARTDP_SOLVER_CLASS::Action,
                                  typename SK_MARTDP_SOLVER_CLASS::Value>>::Map
SK_MARTDP_SOLVER_CLASS::policy() const {
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map p;
  for (auto &n : _graph) {
    if (n.action) {
      Value val;
      for (std::size_t a = 0; a < _nb_agents; a++) {
        val[_agents[a]].cost(n.value[a]);
      }
      p.insert(std::make_pair(n.state, std::make_pair(n.action->action, val)));
    }
  }
  return p;
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
void SK_MARTDP_SOLVER_CLASS::expand_state(StateNode *s) {
  if (_verbose)
    Logger::debug("Trying to expand state " + s->state.print());

  if (s->actions.empty()) {
    Predicate termination;
    for (auto a : _agents) {
      termination[a] = false;
    }
    initialize_node(*s, termination);
    s->expansions_count += generate_more_actions(s);
  } else {
    std::bernoulli_distribution dist_state_expansion(
        std::exp(-_graph_expansion_rate * (s->expansions_count)));
    if (dist_state_expansion(*_gen)) {
      s->expansions_count += generate_more_actions(s);
    }
  }
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
typename SK_MARTDP_SOLVER_CLASS::StateNode *
SK_MARTDP_SOLVER_CLASS::expand_action(ActionNode *a) {
  if (_verbose)
    Logger::debug("Trying to expand action " + a->action.print());
  EnvironmentOutcome outcome = _domain.sample(a->parent->state, a->action);
  auto i = _graph.emplace(outcome.observation());
  StateNode &next_node =
      const_cast<StateNode &>(*(i.first)); // we won't change the real key
                                           // (StateNode::state) so we are safe

  auto tv = outcome.transition_value();
  double transition_cost = 0.0;
  for (const auto &agent : _agents) {
    transition_cost += tv[agent].cost();
  }

  if (i.second) { // new node
    initialize_node(next_node, outcome.termination());
  }
  auto ins = a->outcomes.emplace(
      std::make_pair(&next_node, std::make_pair(transition_cost, 1)));

  // Update the outcome's reward and visits count
  if (ins.second) { // new outcome
    if (_verbose)
      Logger::debug("Discovered new outcome " + next_node.state.print());
    a->dist_to_outcome.push_back(ins.first);
    a->expansions_count += 1;
    next_node.parents.insert(a);
  } else { // known outcome
    if (_verbose)
      Logger::debug("Discovered known outcome " + next_node.state.print());
    std::pair<double, std::size_t> &mp = ins.first->second;
    mp.first = ((double)(mp.second * mp.first) + transition_cost) /
               ((double)(mp.second + 1));
    mp.second += 1;
  }

  // Reconstruct the probability distribution
  std::vector<double> weights(a->dist_to_outcome.size());
  for (unsigned int oid = 0; oid < weights.size(); oid++) {
    weights[oid] = (double)a->dist_to_outcome[oid]->second.second;
  }
  a->dist = std::discrete_distribution<>(weights.begin(), weights.end());

  return &next_node;
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
bool SK_MARTDP_SOLVER_CLASS::generate_more_actions(StateNode *s) {
  if (_verbose)
    Logger::debug("Generating (more) actions for state " + s->state.print());
  bool new_actions = false;

  for (std::size_t agent = 0; agent < _nb_agents; agent++) {
    if (_verbose)
      Logger::debug("Trying agent " + _agents[agent].print() + " actions");
    auto agent_applicable_actions =
        _domain.get_agent_applicable_actions(s->state, Action(), _agents[agent])
            .get_elements();

    if (agent_applicable_actions.empty()) {
      if (_verbose)
        Logger::debug("No agent applicable actions");
      continue;
    } else {
      for (auto action : agent_applicable_actions) {
        if (_verbose)
          Logger::debug("Trying agent action " + action.print());
        Action agent_actions;
        agent_actions[_agents[agent]] = action;

        // try to find a feasible assignment
        bool feasible = false;
        std::size_t feasibility_trial = 0;

        while (!feasible && feasibility_trial < _max_feasibility_trials) {
          std::shuffle(_agents_orders[agent].begin(),
                       _agents_orders[agent].end(), *_gen);
          feasibility_trial++;
          feasible = true;

          // construct the joint action
          for (auto other_agent : _agents_orders[agent]) {
            auto other_agent_aa = _domain.get_agent_applicable_actions(
                s->state, agent_actions, _agents[other_agent]);

            if (other_agent_aa.empty()) {
              feasible = false;
              break;
            }

            // Is the agent's optimal action applicable ?
            if ((s->best_action) &&
                other_agent_aa.contains(
                    (*(s->best_action))[_agents[other_agent]]) &&
                !(_action_choice_noise_dist(
                    *_gen))) { // choose it with high probability
              agent_actions[_agents[other_agent]] =
                  (*(s->best_action))[_agents[other_agent]];
            } else {
              agent_actions[_agents[other_agent]] = other_agent_aa.sample();
            }
          }
        }

        if (feasible) {
          // Add the action
          auto a = s->actions.emplace(agent_actions);
          ActionNode *action_node = &const_cast<ActionNode &>(
              *(a.first)); // we won't change the real key (ActionNode::action)
                           // so we are safe
          if (a.second) {  // new action
            new_actions = true;
            action_node->parent = s;
            action_node->value.resize(_nb_agents,
                                      std::numeric_limits<double>::infinity());
            action_node->all_value = std::numeric_limits<double>::infinity();

            // add one sampled outcome
            expand_action(action_node);
          }
        } else if (_verbose)
          Logger::debug("Failed finding a joint applicable action");
      }
    }
  }

  return new_actions;
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
typename SK_MARTDP_SOLVER_CLASS::ActionNode *
SK_MARTDP_SOLVER_CLASS::greedy_action(StateNode *s) {
  if (_verbose)
    Logger::debug("Updating state " + s->state.print());

  double best_value = std::numeric_limits<double>::infinity();
  ActionNode *best_action = nullptr;

  for (const ActionNode &act : s->actions) {
    if (_verbose)
      Logger::debug("Computing Q-value of (" + s->state.print() + ", " +
                    act.action.print() + ")");

    ActionNode &action =
        const_cast<ActionNode &>(act); // we won't change the real key
                                       // (ActionNode::action) so we are safe
    action.all_value = 0.0;
    std::for_each(action.value.begin(), action.value.end(),
                  [](double &v) { v = 0.0; });

    for (const auto &outcome : action.outcomes) {
      double outcome_cost = outcome.second.first;
      double outcome_probability =
          ((double)outcome.second.second) / ((double)action.expansions_count);
      for (std::size_t a = 0; a < _nb_agents; a++) {
        action.value[a] =
            outcome_probability *
            (outcome_cost + (_discount * outcome.first->value[a]));
        action.all_value += action.value[a];
      }
    }

    if (action.all_value < best_value) {
      best_value = action.all_value;
      best_action = &action;
    }

    if (_verbose)
      Logger::debug("Updated Q-value of action " + action.action.print() +
                    " with value " + StringConverter::from(action.all_value));
  }

  if (_verbose) {
    if (best_action) {
      Logger::debug("Greedy action of state " + s->state.print() + ": " +
                    best_action->action.print() + " with value " +
                    StringConverter::from(best_value));
    } else {
      Logger::debug("State " + s->state.print() +
                    " is a dead-end or a goal (no feasible actions found)");
    }
  }

  if (best_action) {
    s->best_action = std::make_unique<Action>(best_action->action);
  }
  s->action = best_action;
  s->all_value = best_value;
  return s->action; // action is nullptr if state is a dead-end or a goal
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
typename SK_MARTDP_SOLVER_CLASS::StateNode *
SK_MARTDP_SOLVER_CLASS::pick_next_state(ActionNode *a) {
  if (_verbose)
    Logger::debug("Picking next state from State " + a->parent->state.print() +
                  " with action " + a->action.print());

  StateNode *next_node = expand_action(a);

  if (_verbose)
    Logger::debug("Picked next state " + next_node->state.print() +
                  " from state " + a->parent->state.print() + " and action " +
                  a->action.print());
  return next_node;
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
void SK_MARTDP_SOLVER_CLASS::backtrack_values(StateNode *s) {
  if (_verbose)
    Logger::debug("Backtracking values from state " + s->state.print());
  std::unordered_set<StateNode *> frontier;
  std::unordered_set<StateNode *> visited;
  frontier.insert(s);
  visited.insert(s);

  while (!frontier.empty()) {
    std::unordered_set<StateNode *> new_frontier;
    for (StateNode *f : frontier) {
      for (ActionNode *a : f->parents) {
        if (visited.find(a->parent) == visited.end()) {
          greedy_action(a->parent);
          visited.insert(a->parent);
          new_frontier.insert(a->parent);
        }
      }
    }
    frontier = new_frontier;
  }
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
void SK_MARTDP_SOLVER_CLASS::initialize_node(StateNode &n,
                                             const Predicate &termination) {
  if (_verbose) {
    Logger::debug("Initializing new state node " + n.state.print());
  }

  n.value.resize(_nb_agents, 0.0);
  n.all_value = 0.0;
  n.goal.resize(_nb_agents, false);
  n.all_goal = true;
  n.termination.resize(_nb_agents, false);
  n.all_termination = true;

  Predicate g = _goal_checker(_domain, n.state);
  auto h = _heuristic(_domain, n.state);

  n.action = nullptr;
  n.best_action = std::make_unique<Action>();

  for (std::size_t a = 0; a < _nb_agents; a++) {
    n.goal[a] = g[_agents[a]];
    n.all_goal = n.all_goal && n.goal[a];
    n.termination[a] = termination[_agents[a]];
    n.all_termination = n.all_termination && n.termination[a];

    if (n.goal[a]) {
      n.value[a] = 0.0;
    } else if (n.termination[a]) { // dead-end state
      n.value[a] = _dead_end_cost / ((double)_nb_agents);
    } else {
      n.value[a] = h.first[_agents[a]].cost();
      (*n.best_action)[_agents[a]] = h.second[_agents[a]];
    }

    n.all_value += n.value[a];
  }
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
void SK_MARTDP_SOLVER_CLASS::trial(StateNode *s) {
  StateNode *cs = s;
  std::size_t depth = 0;

  while ((get_solving_time() < _time_budget) && (depth < _max_depth)) {
    depth++;

    if (cs->all_goal) {
      if (_verbose)
        Logger::debug("Found goal state " + cs->state.print());
      break;
    } else if (cs->all_termination) {
      if (_verbose)
        Logger::debug("Found dead-end state " + cs->state.print());
      break;
    }

    expand_state(cs);
    ActionNode *action = greedy_action(cs);

    if (action) {
      cs = pick_next_state(action);
    } else { // current state is a dead-end
      cs->all_value = _dead_end_cost;
      std::for_each(cs->value.begin(), cs->value.end(), [this](auto &v) {
        v = _dead_end_cost / ((double)_nb_agents);
      });
      break;
    }
  }

  backtrack_values(cs);
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
void SK_MARTDP_SOLVER_CLASS::compute_reachable_subgraph(
    StateNode *node, std::unordered_set<StateNode *> &subgraph) {
  std::unordered_set<StateNode *> frontier;
  frontier.insert(node);
  subgraph.insert(node);
  while (!frontier.empty()) {
    std::unordered_set<StateNode *> new_frontier;
    for (auto &n : frontier) {
      for (auto &action : n->actions) {
        for (auto &outcome : action.outcomes) {
          if (subgraph.find(outcome.first) == subgraph.end()) {
            new_frontier.insert(outcome.first);
            subgraph.insert(outcome.first);
          }
        }
      }
    }
    frontier = new_frontier;
  }
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
void SK_MARTDP_SOLVER_CLASS::remove_subgraph(
    std::unordered_set<StateNode *> &root_subgraph,
    std::unordered_set<StateNode *> &child_subgraph) {
  std::unordered_set<StateNode *> removed_subgraph;
  // First pass: look for nodes in root_subgraph but not child_subgraph and
  // remove those nodes from their children's parents Don't actually remove
  // those nodes in the first pass otherwise some children to remove won't exist
  // anymore when looking for their parents
  for (auto &n : root_subgraph) {
    if (child_subgraph.find(n) == child_subgraph.end()) {
      for (auto &action : n->actions) {
        for (auto &outcome : action.outcomes) {
          // we won't change the real key (ActionNode::action) so we are safe
          outcome.first->parents.erase(&const_cast<ActionNode &>(action));
        }
      }
      removed_subgraph.insert(n);
    }
  }
  // Second pass: actually remove nodes in root_subgraph but not in
  // child_subgraph
  for (auto &n : removed_subgraph) {
    _graph.erase(StateNode(n->state));
  }
}

SK_MARTDP_SOLVER_TEMPLATE_DECL
void SK_MARTDP_SOLVER_CLASS::update_residual_moving_average(
    const StateNode &node, const double &node_record_value) {
  if (_residual_moving_average_window > 0) {
    double current_residual = std::fabs(node_record_value - node.all_value);
    if (_residuals.size() < _residual_moving_average_window) {
      _residual_moving_average =
          ((double)((_residual_moving_average * _residuals.size()) +
                    current_residual)) /
          ((double)(_residuals.size() + 1));
    } else {
      _residual_moving_average += (current_residual - _residuals.front()) /
                                  ((double)_residual_moving_average_window);
      _residuals.pop_front();
    }
    _residuals.push_back(current_residual);
  }
}

// === MARTDPSolver::ActionNode implementation ===

SK_MARTDP_SOLVER_TEMPLATE_DECL
SK_MARTDP_SOLVER_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), expansions_count(0), all_value(0.0), parent(nullptr) {}

SK_MARTDP_SOLVER_TEMPLATE_DECL
SK_MARTDP_SOLVER_CLASS::ActionNode::ActionNode(const ActionNode &a)
    : action(a.action), outcomes(a.outcomes),
      dist_to_outcome(a.dist_to_outcome), dist(a.dist),
      expansions_count(a.expansions_count), value(a.value),
      all_value(a.all_value), parent(a.parent) {}

SK_MARTDP_SOLVER_TEMPLATE_DECL
const typename SK_MARTDP_SOLVER_CLASS::Action &
SK_MARTDP_SOLVER_CLASS::ActionNode::Key::operator()(
    const ActionNode &an) const {
  return an.action;
}

// === MARTDPSolver::StateNode implementation ===

SK_MARTDP_SOLVER_TEMPLATE_DECL
SK_MARTDP_SOLVER_CLASS::StateNode::StateNode(const State &s)
    : state(s), action(nullptr), best_action(nullptr), expansions_count(0),
      all_value(std::numeric_limits<double>::infinity()), all_goal(false),
      all_termination(false) {}

SK_MARTDP_SOLVER_TEMPLATE_DECL
SK_MARTDP_SOLVER_CLASS::StateNode::StateNode(const StateNode &s)
    : state(s.state), action(s.action),
      best_action(s.best_action ? std::make_unique<Action>(*(s.best_action))
                                : nullptr),
      expansions_count(s.expansions_count), actions(s.actions), value(s.value),
      all_value(s.all_value), goal(s.goal), all_goal(s.all_goal),
      termination(s.termination), all_termination(s.all_termination),
      parents(s.parents) {}

SK_MARTDP_SOLVER_TEMPLATE_DECL
const typename SK_MARTDP_SOLVER_CLASS::State &
SK_MARTDP_SOLVER_CLASS::StateNode::Key::operator()(const StateNode &sn) const {
  return sn.state;
}

} // namespace skdecide

#endif // SKDECIDE_MARTDP_IMPL_HH
