/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_ILAOSTAR_IMPL_HH
#define SKDECIDE_ILAOSTAR_IMPL_HH

#include <queue>
#include <cmath>
#include <chrono>

#include "utils/string_converter.hh"
#include "utils/logging.hh"

namespace skdecide {

// === ILAOStarSolver implementation ===

#define SK_ILAOSTAR_SOLVER_TEMPLATE_DECL                                       \
  template <typename Tdomain, typename Texecution_policy>

#define SK_ILAOSTAR_SOLVER_CLASS ILAOStarSolver<Tdomain, Texecution_policy>

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
SK_ILAOSTAR_SOLVER_CLASS::ILAOStarSolver(Domain &domain,
                                         const GoalCheckerFunctor &goal_checker,
                                         const HeuristicFunctor &heuristic,
                                         double discount, double epsilon,
                                         const CallbackFunctor &callback,
                                         bool verbose)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _discount(discount), _epsilon(epsilon), _callback(callback),
      _verbose(verbose) {

  if (verbose) {
    Logger::check_level(logging::debug, "algorithm ILAO*");
  }
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
void SK_ILAOSTAR_SOLVER_CLASS::clear() { _graph.clear(); }

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
void SK_ILAOSTAR_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " ILAO* solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();

    auto si = _graph.emplace(s);
    StateNode &root_node = const_cast<StateNode &>(
        *(si.first)); // we won't change the real key (StateNode::state) so we
                      // are safe

    if (si.second) {
      root_node.best_value = _heuristic(_domain, s).cost();
    }

    if (root_node.solved ||
        _goal_checker(_domain,
                      s)) { // problem already solved from this state (was
                            // present in _graph and already solved)
      if (_verbose)
        Logger::debug("Found goal state " + s.print());
      return;
    }

    while (!root_node.solved && !_callback(*this, _domain)) {
      // perform postorder depth first search until not reaching unexpanded tip
      // nodes
      while (root_node.reach_tip_node) {
        depth_first_search(root_node);
      }

      // compute best solution graph for value iteration
      compute_best_solution_graph(root_node);
      // perform value iteration on the best solution graph
      value_iteration();
      // recompute best solution graph after value iteration updates
      compute_best_solution_graph(root_node);
      // compute unexpanded tip node reachability from every state in best
      // policy
      compute_reachability();
      // compute mean first passage time to goal states from states in best
      // policy
      compute_mfpt();
      // update solved bits; note that Hansen and Zilberstein compute only the
      // initial state solve status but we do it for every state of the best
      // policy (thus the computation above of tip node reachability for every
      // state of the best policy) in order to prematurely identify solved
      // states and speed-up the search
      update_solved_bits();

      if (_verbose) {
        std::string graph_str = "{";
        for (const auto &bs : _best_solution_graph) {
          graph_str += " " + bs->state.print() + " ;";
        }
        graph_str.back() = '}';
        Logger::debug("Current best solution graph is: " + graph_str);
      }
    }

    Logger::info(
        "ILAO* finished to solve from state " + s.print() + " in " +
        StringConverter::from((double)get_solving_time() / (double)1e6) +
        " seconds.");
  } catch (const std::exception &e) {
    Logger::error("ILAO* failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
bool SK_ILAOSTAR_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr)) {
    return false;
  } else {
    return true;
  }
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
const typename SK_ILAOSTAR_SOLVER_CLASS::Action &
SK_ILAOSTAR_SOLVER_CLASS::get_best_action(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr)) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return si->best_action->action;
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
typename SK_ILAOSTAR_SOLVER_CLASS::Value
SK_ILAOSTAR_SOLVER_CLASS::get_best_value(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  Value val;
  val.cost(si->best_value);
  return val;
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
std::size_t SK_ILAOSTAR_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_ILAOSTAR_SOLVER_CLASS::State>::Set
SK_ILAOSTAR_SOLVER_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored_states;
  for (const auto &s : _graph) {
    explored_states.insert(s.state);
  }
  return explored_states;
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
std::size_t SK_ILAOSTAR_SOLVER_CLASS::best_solution_graph_size() const {
  return _best_solution_graph.size();
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
std::size_t SK_ILAOSTAR_SOLVER_CLASS::get_solving_time() const {
  std::size_t milliseconds_duration;
  milliseconds_duration = static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
  return milliseconds_duration;
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
typename ::skdecide::MapTypeDeducer<
    typename SK_ILAOSTAR_SOLVER_CLASS::State,
    std::pair<typename SK_ILAOSTAR_SOLVER_CLASS::Action, double>>::Map
SK_ILAOSTAR_SOLVER_CLASS::policy() const {
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map p;
  for (auto &n : _graph) {
    if (n.best_action != nullptr) {
      p.insert(std::make_pair(n.state, std::make_pair(n.best_action->action,
                                                      (double)n.best_value)));
    }
  }
  return p;
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
void SK_ILAOSTAR_SOLVER_CLASS::expand(StateNode &s) {
  if (_verbose)
    Logger::debug("Expanding state " + s.state.print());
  auto applicable_actions =
      _domain.get_applicable_actions(s.state).get_elements();

  std::for_each(
      ExecutionPolicy::policy, applicable_actions.begin(),
      applicable_actions.end(), [this, &s](auto a) {
        if (_verbose)
          Logger::debug("Current expanded action: " + a.print() +
                        ExecutionPolicy::print_thread());
        _execution_policy.protect(
            [&s, &a] { s.actions.push_back(std::make_unique<ActionNode>(a)); });
        ActionNode &an = *(s.actions.back());
        auto next_states =
            _domain.get_next_state_distribution(s.state, a).get_values();

        for (auto ns : next_states) {
          if (_verbose)
            Logger::debug("Current next state expansion: " +
                          ns.state().print() + ExecutionPolicy::print_thread());
          std::pair<typename Graph::iterator, bool> i;
          _execution_policy.protect(
              [this, &i, &ns] { i = _graph.emplace(ns.state()); });
          StateNode &next_node = const_cast<StateNode &>(
              *(i.first)); // we won't change the real key (StateNode::state) so
                           // we are safe
          an.outcomes.push_back(std::make_tuple(
              ns.probability(),
              _domain.get_transition_value(s.state, a, next_node.state).cost(),
              &next_node));

          if (i.second) { // new node
            if (_goal_checker(_domain, next_node.state)) {
              if (_verbose)
                Logger::debug("Found goal state " + next_node.state.print() +
                              ExecutionPolicy::print_thread());
              next_node.goal = true;
              next_node.solved = true;
              next_node.best_value = 0.0;
            } else {
              next_node.best_value =
                  _heuristic(_domain, next_node.state).cost();
              if (_verbose)
                Logger::debug("New state " + next_node.state.print() +
                              " with heuristic value " +
                              StringConverter::from(next_node.best_value) +
                              ExecutionPolicy::print_thread());
            }
          }
        }
      });
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
void SK_ILAOSTAR_SOLVER_CLASS::depth_first_search(StateNode &s) {
  if (_verbose)
    Logger::debug("Running post-order depth-first search from state " +
                  s.state.print());
  std::unordered_set<StateNode *> visited;
  std::stack<StateNode *> open;
  open.push(&s);
  s.reach_tip_node = false;

  while (!open.empty()) {
    StateNode *cs = open.top();

    if (cs->solved) {
      if (_verbose)
        Logger::debug("Found solved state " + cs->state.print());
      open.pop();
    } else if (cs->goal) {
      if (_verbose)
        Logger::debug("Found goal state " + cs->state.print());
      cs->best_value = 0;
      open.pop();
    } else if (cs->actions.empty()) {
      if (_verbose)
        Logger::debug("Found unexpanded tip node " + cs->state.print());
      s.reach_tip_node = true;
      expand(*cs);
      update(*cs);
      open.pop();
    } else if (visited.find(cs) ==
               visited.end()) { // first visit, we push successor nodes
      if (_verbose)
        Logger::debug("Visiting successors of state " + cs->state.print());
      visited.insert(cs);
      for (const auto &o : cs->best_action->outcomes) {
        StateNode *ns = std::get<2>(o);
        if (visited.find(ns) == visited.end()) {
          open.push(ns);
        }
      }
    } else { // second visit, we update and pop the node
      if (_verbose)
        Logger::debug("Closing state " + cs->state.print());
      update(*cs);
      open.pop();
    }
  }
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
void SK_ILAOSTAR_SOLVER_CLASS::compute_best_solution_graph(StateNode &s) {
  if (_verbose)
    Logger::debug("Computing best solution graph from state " +
                  s.state.print());
  _best_solution_graph.clear();
  _best_solution_graph.insert(&s);
  std::unordered_set<StateNode *> frontier;
  frontier.insert(&s);

  while (!frontier.empty()) {
    std::unordered_set<StateNode *> new_frontier;
    for (const auto &fs : frontier) {
      if (fs->best_action != nullptr) {
        for (const auto &ns : fs->best_action->outcomes) {
          StateNode *nst = std::get<2>(ns);
          if ((nst->goal) || (nst->solved)) {
            if (_verbose)
              Logger::debug("Found terminal (either goal or solved) node " +
                            nst->state.print());
            nst->reach_tip_node = false;
          } else if (nst->actions.empty()) {
            if (_verbose)
              Logger::debug("Found unexpanded tip node " + nst->state.print());
            nst->reach_tip_node = true;
          } else if (_best_solution_graph.find(nst) ==
                     _best_solution_graph.end()) {
            if (_verbose)
              Logger::debug("Inserting node " + nst->state.print());
            nst->reach_tip_node = false;
            _best_solution_graph.insert(nst);
            new_frontier.insert(nst);
          }
        }
      }
    }
    frontier = new_frontier;
  }
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
double SK_ILAOSTAR_SOLVER_CLASS::update(StateNode &s) {
  if (_verbose)
    Logger::debug("Updating state " + s.state.print());
  double record_value = s.best_value;
  double best_value = std::numeric_limits<double>::infinity();
  s.best_action = nullptr;

  for (const auto &a : s.actions) {
    a->value = 0;
    for (const auto &o : a->outcomes) {
      a->value += std::get<0>(o) *
                  (std::get<1>(o) + (_discount * std::get<2>(o)->best_value));
    }
    if (_verbose)
      Logger::debug("Computed Q-value of action " + a->action.print() + " : " +
                    StringConverter::from(a->value));
    if ((a->value) < best_value) {
      best_value = a->value;
      s.best_action = a.get();
    }
  }

  s.best_value = best_value;
  s.residual = std::fabs(best_value - record_value);
  return s.residual;
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
void SK_ILAOSTAR_SOLVER_CLASS::value_iteration() {
  if (_verbose)
    Logger::debug("Running value iteration");
  atomic_double residual = std::numeric_limits<double>::infinity();

  while (residual > _epsilon) {
    residual = 0;
    std::for_each(ExecutionPolicy::policy, _best_solution_graph.begin(),
                  _best_solution_graph.end(), [this, &residual](auto &s) {
                    residual = std::max((double)residual, update(*s));
                  });
  }

  if (_verbose)
    Logger::debug("Value iteration converged with residual " +
                  StringConverter::from(residual));
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
bool SK_ILAOSTAR_SOLVER_CLASS::update_reachability(StateNode &s) {
  if (_verbose)
    Logger::debug("Updating unexpanded tip node reachability of state " +
                  s.state.print());
  bool record = s.reach_tip_node;
  bool reach_tip_node = false;

  for (const auto &o : s.best_action->outcomes) {
    reach_tip_node = reach_tip_node || (std::get<2>(o)->reach_tip_node);
  }

  if (_verbose)
    Logger::debug("Unexpanded tip node reachability : " +
                  StringConverter::from(reach_tip_node));
  s.reach_tip_node = reach_tip_node;
  return record != reach_tip_node;
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
void SK_ILAOSTAR_SOLVER_CLASS::compute_reachability() {
  if (_verbose)
    Logger::debug("Computing reachability of unexpanded tip nodes");
  atomic_bool changes = true;

  while (changes) {
    changes = false;
    std::for_each(ExecutionPolicy::policy, _best_solution_graph.begin(),
                  _best_solution_graph.end(), [this, &changes](auto &s) {
                    changes = update_reachability(*s) || changes;
                  });
  }

  if (_verbose)
    Logger::debug("Unexpanded tip node reachability converged");
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
double SK_ILAOSTAR_SOLVER_CLASS::update_mfpt(StateNode &s) {
  if (_verbose)
    Logger::debug("Updating mean first passage time of state " +
                  s.state.print());
  double record_value = s.first_passage_time;
  double first_passage_time = 0;

  for (const auto &o : s.best_action->outcomes) {
    first_passage_time +=
        std::get<0>(o) * (1.0 + (std::get<2>(o)->first_passage_time));
  }

  if (_verbose)
    Logger::debug("Mean first passage time : " +
                  StringConverter::from(first_passage_time));
  s.first_passage_time = first_passage_time;
  return std::fabs(first_passage_time - record_value);
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
void SK_ILAOSTAR_SOLVER_CLASS::compute_mfpt() {
  // Compute mean first passage time from s to any goal state.
  // We don't use exact computation with matrix inversion because
  // we are going to multiply the resuling value (which will be >> 1)
  // with the residual (which is << 1) thus approximating the mean
  // first passage time by a dynamic programming scheme at _epsilon
  // precision should be acceptable.

  if (_verbose)
    Logger::debug("Computing mean first passage times");
  atomic_double residual = std::numeric_limits<double>::infinity();
  while (residual > _epsilon) {
    residual = 0;
    std::for_each(ExecutionPolicy::policy, _best_solution_graph.begin(),
                  _best_solution_graph.end(), [this, &residual](auto &s) {
                    residual = std::max((double)residual, update_mfpt(*s));
                  });
  }
  if (_verbose)
    Logger::debug(
        "Mean first passage time computation converged with residual " +
        StringConverter::from(residual));
}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
void SK_ILAOSTAR_SOLVER_CLASS::update_solved_bits() {
  // check if unexpanded tip nodes are reached by the best policy
  // and the error bound falls below epsilon
  // (note: Hansen and Zilberstein seem to suggest that the convergence
  //  test should be a disjunction but in this case the mean first
  //  passage time computed during the evaluation of the error
  //  bound would be erroneous if the best solution graph can
  //  can reach an unexpanded tip node)
  if (_verbose)
    Logger::debug("Updating solved bits");
  std::for_each(
      ExecutionPolicy::policy, _best_solution_graph.begin(),
      _best_solution_graph.end(), [this](auto &s) {
        s->solved = !(s->reach_tip_node) &&
                    (((s->first_passage_time) * (s->residual)) < _epsilon);
        if (_verbose)
          Logger::debug(
              "Unexpanded tip node reachability and error bound of state " +
              s->state.print() + " : " +
              StringConverter::from(s->reach_tip_node) + " ; " +
              StringConverter::from(s->first_passage_time) + " * " +
              StringConverter::from(s->residual) + " = " +
              StringConverter::from((s->first_passage_time) * (s->residual)));
      });
}

// === ILAOStarSolver::StateNode implementation ===

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
SK_ILAOSTAR_SOLVER_CLASS::StateNode::StateNode(const State &s)
    : state(s), best_action(nullptr),
      best_value(std::numeric_limits<double>::infinity()),
      first_passage_time(0), residual(std::numeric_limits<double>::infinity()),
      goal(false), reach_tip_node(true), solved(false) {}

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
const typename SK_ILAOSTAR_SOLVER_CLASS::State &
SK_ILAOSTAR_SOLVER_CLASS::StateNode::Key::operator()(
    const StateNode &sn) const {
  return sn.state;
}

// === ILAOStarSolver::ActionNode implementation ===

SK_ILAOSTAR_SOLVER_TEMPLATE_DECL
SK_ILAOSTAR_SOLVER_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), value(std::numeric_limits<double>::infinity()) {}

} // namespace skdecide

#endif // SKDECIDE_ILAOSTAR_IMPL_HH
