/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Implementation of LDFS(MDP) from Figure 4 of:
 * Bonet & Geffner, "Learning Depth-First Search", ICAPS 2008.
 */
#ifndef SKDECIDE_LDFS_IMPL_HH
#define SKDECIDE_LDFS_IMPL_HH

#include <stack>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>

#include "utils/string_converter.hh"
#include "utils/logging.hh"

namespace skdecide {

#define SK_LDFS_SOLVER_TEMPLATE_DECL                                           \
  template <typename Tdomain, typename Texecution_policy>

#define SK_LDFS_SOLVER_CLASS LDFSSolver<Tdomain, Texecution_policy>

// --- StateNode ---

SK_LDFS_SOLVER_TEMPLATE_DECL
SK_LDFS_SOLVER_CLASS::StateNode::StateNode(const State &s)
    : state(s), best_action(nullptr), best_value(0.0), goal(false),
      terminal(false), solved(false), active(false), idx(IDX_UNDEF),
      low(IDX_UNDEF) {}

SK_LDFS_SOLVER_TEMPLATE_DECL
const typename SK_LDFS_SOLVER_CLASS::State &
SK_LDFS_SOLVER_CLASS::StateNode::Key::operator()(const StateNode &sn) const {
  return sn.state;
}

// --- ActionNode ---

SK_LDFS_SOLVER_TEMPLATE_DECL
SK_LDFS_SOLVER_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), value(0.0) {}

// --- Constructor ---

SK_LDFS_SOLVER_TEMPLATE_DECL
SK_LDFS_SOLVER_CLASS::LDFSSolver(Domain &domain,
                                 const GoalCheckerFunctor &goal_checker,
                                 const HeuristicFunctor &heuristic,
                                 const TerminalValueFunctor &terminal_value,
                                 double discount, double epsilon,
                                 std::size_t max_depth,
                                 const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _terminal_value(terminal_value), _discount(discount), _epsilon(epsilon),
      _max_depth(max_depth), _callback(callback), _verbose(verbose),
      _nb_tip_states(0), _tarjan_index(0) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm LDFS");
  }
}

SK_LDFS_SOLVER_TEMPLATE_DECL
void SK_LDFS_SOLVER_CLASS::clear() {
  _graph.clear();
  _sccs.clear();
  _nb_tip_states = 0;
  _tarjan_index = 0;
  while (!_tarjan_stack.empty())
    _tarjan_stack.pop();
}

// --- LDFS(MDP)-DRIVER (Figure 4, top) ---

SK_LDFS_SOLVER_TEMPLATE_DECL
void SK_LDFS_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " LDFS solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();

    auto si = _graph.emplace(s);
    StateNode &root = const_cast<StateNode &>(*(si.first));
    if (si.second) {
      root.best_value = _heuristic(_domain, s).cost();
    }

    while (!root.solved && !_callback(*this, _domain)) {
      _tarjan_index = 0;
      ldfs_mdp(root);
      clear_active_flags();
    }

    Logger::info(
        "LDFS finished from state " + s.print() + " in " +
        StringConverter::from((double)get_solving_time() / (double)1e3) +
        " seconds with " + StringConverter::from(_graph.size()) +
        " explored states.");
  } catch (const std::exception &e) {
    Logger::error("LDFS failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

// --- clear_active_flags (called by driver after each LDFS(MDP) call) ---

SK_LDFS_SOLVER_TEMPLATE_DECL
void SK_LDFS_SOLVER_CLASS::clear_active_flags() {
  for (auto &sn : _graph) {
    StateNode &node = const_cast<StateNode &>(sn);
    node.active = false;
  }
}

// --- expand ---

SK_LDFS_SOLVER_TEMPLATE_DECL
void SK_LDFS_SOLVER_CLASS::expand(StateNode &s) {
  if (_verbose)
    Logger::debug("Expanding state " + s.state.print());

  _nb_tip_states++;
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
          StateNode &next_node = const_cast<StateNode &>(*(i.first));
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
              next_node.best_value = 0.0;
            } else if (_domain.is_terminal(next_node.state)) {
              if (_verbose)
                Logger::debug("Found terminal state " +
                              next_node.state.print() +
                              ExecutionPolicy::print_thread());
              next_node.terminal = true;
              next_node.best_value = _terminal_value(next_node.state).cost();
            } else {
              next_node.best_value =
                  _heuristic(_domain, next_node.state).cost();
              if (_verbose)
                Logger::debug("New state " + next_node.state.print() +
                              " with initial value " +
                              StringConverter::from(next_node.best_value) +
                              ExecutionPolicy::print_thread());
            }
          }
        }
      });
}

// --- q_value (cost minimization) ---

SK_LDFS_SOLVER_TEMPLATE_DECL
double SK_LDFS_SOLVER_CLASS::q_value(ActionNode &a) {
  a.value = 0.0;
  for (const auto &o : a.outcomes) {
    a.value += std::get<0>(o) *
               (std::get<1>(o) + _discount * std::get<2>(o)->best_value);
  }
  return a.value;
}

// --- LDFS(MDP) (Figure 4, bottom) ---
//
// Explicit-stack implementation matching the paper's pseudo-code exactly.
// Each DFSFrame simulates one recursive call of LDFS(MDP)(s, ε, index, stack).
// The nested foreach-action / foreach-successor loops are tracked via
// iterators stored in the frame. When a child "call" is needed, we push
// a new frame and continue; when it "returns", we resume the parent frame
// using _last_rv to pass the return value.

SK_LDFS_SOLVER_TEMPLATE_DECL
void SK_LDFS_SOLVER_CLASS::ldfs_mdp(StateNode &root) {
  typedef typename std::list<std::unique_ptr<ActionNode>>::iterator ActionIter;
  typedef typename std::list<std::tuple<double, double, StateNode *>>::iterator
      OutcomeIter;

  struct DFSFrame {
    StateNode *node;
    ActionIter action_it;
    OutcomeIter outcome_it;
    ActionNode *current_action;
    StateNode *child_returned; // non-null when resuming after child call
    bool flag;
    bool initialized;

    DFSFrame(StateNode *n)
        : node(n), current_action(nullptr), child_returned(nullptr),
          flag(false), initialized(false) {}
  };

  std::stack<DFSFrame> dfs_stack;
  dfs_stack.push(DFSFrame(&root));

  while (!dfs_stack.empty()) {
    DFSFrame &f = dfs_stack.top();
    StateNode *s = f.node;

    // ===== INITIALIZATION (first entry into this frame) =====
    if (!f.initialized) {
      // "if s is SOLVED or terminal then"
      if (s->solved || s->goal || s->terminal) {
        if (s->goal) {
          s->best_value = 0.0;
        } else if (s->terminal) {
          s->best_value = _terminal_value(s->state).cost();
        }
        s->solved = true;
        _last_rv = true;
        dfs_stack.pop();
        continue;
      }

      // "if s is ACTIVE then return false"
      if (s->active) {
        _last_rv = false;
        dfs_stack.pop();
        continue;
      }

      // Expand on first visit
      if (s->actions.empty()) {
        expand(*s);
      }

      // "Push s into stack; s.idx := s.low := index; index := index + 1"
      _tarjan_stack.push(s);
      s->idx = _tarjan_index;
      s->low = _tarjan_index;
      _tarjan_index++;

      // "flag := false"
      f.flag = false;
      f.action_it = s->actions.begin();
      f.current_action = nullptr;
      f.child_returned = nullptr;
      f.initialized = true;
    }

    // ===== HANDLE CHILD RETURN =====
    if (f.child_returned != nullptr) {
      // "flag := LDFS(s', ...) & flag" — apply child's return value
      f.flag = _last_rv && f.flag;
      // "s.low := min{s.low, s'.low}"
      s->low = std::min(s->low, f.child_returned->low);
      f.child_returned = nullptr;
      // Fall through to continue outcome iteration
    }

    // ===== OUTCOME LOOP (continue iterating successors of current action)
    // =====
    bool pushed_child = false;

    if (f.current_action != nullptr) {
      // We're inside an action's outcome loop — continue from f.outcome_it
      while (f.outcome_it != f.current_action->outcomes.end()) {
        StateNode *sp = std::get<2>(*f.outcome_it);
        ++f.outcome_it;

        if (sp->idx == IDX_UNDEF) {
          // Depth limit check: if reached, treat as unsolved
          if (_max_depth > 0 && dfs_stack.size() >= _max_depth) {
            f.flag = false;
            continue;
          }
          // "flag := LDFS(s', ...) & flag" — push child frame
          f.child_returned = sp;
          dfs_stack.push(DFSFrame(sp));
          pushed_child = true;
          break;
        } else if (sp->active) {
          // "s.low := min{s.low, s'.idx}"
          s->low = std::min(s->low, sp->idx);
        }
      }

      if (pushed_child) {
        continue; // process child first
      }

      // Outcome loop finished for current action
      // "if flag then break"
      if (f.flag) {
        s->best_action = f.current_action;
        // Skip remaining actions — go to finalization
        f.action_it = s->actions.end();
      } else {
        // This action didn't work, try next
        ++f.action_it;
        f.current_action = nullptr;
      }
    }

    // ===== ACTION LOOP (find next consistent action) =====
    if (!pushed_child && f.current_action == nullptr) {
      bool found_action = false;

      while (f.action_it != s->actions.end()) {
        ActionNode &a = **f.action_it;

        // "if Q_V(a,s) - V(s) > ε then continue"
        if (q_value(a) - s->best_value > _epsilon) {
          ++f.action_it;
          continue;
        }

        // "Mark s as ACTIVE; flag := true"
        s->active = true;
        f.flag = true;
        f.current_action = &a;
        f.outcome_it = a.outcomes.begin();
        found_action = true;
        break;
      }

      if (found_action) {
        continue; // enter outcome loop for this action
      }
    }

    if (pushed_child) {
      continue;
    }

    // ===== FINALIZATION (after action loop exhausted) =====

    // "while stack.top.idx > s.idx do
    //    stack.top.idx := stack.top.low := ∞; Pop stack"
    while (!_tarjan_stack.empty() && _tarjan_stack.top()->idx > s->idx) {
      _tarjan_stack.top()->idx = IDX_UNDEF;
      _tarjan_stack.top()->low = IDX_UNDEF;
      _tarjan_stack.pop();
    }

    if (!f.flag) {
      // "V(s) := min_{a∈A(s)} Q_V(a,s)" — Bellman update
      double best_val = std::numeric_limits<double>::infinity();
      ActionNode *best_act = nullptr;
      for (auto &a_ptr : s->actions) {
        double qv = q_value(*a_ptr);
        if (qv < best_val) {
          best_val = qv;
          best_act = a_ptr.get();
        }
      }
      s->best_value = best_val;
      s->best_action = best_act;

      if (_verbose)
        Logger::debug("Updated state " + s->state.print() + " to value " +
                      StringConverter::from(s->best_value));

      // "s.idx := s.low := ∞; Pop stack"
      s->idx = IDX_UNDEF;
      s->low = IDX_UNDEF;
      if (!_tarjan_stack.empty() && _tarjan_stack.top() == s) {
        _tarjan_stack.pop();
      }

      _last_rv = false;
    } else if (s->low == s->idx) {
      // SCC root: "while stack.top.idx ≥ s.idx do
      //   Mark s as SOLVED; stack.top.idx := stack.top.low := ∞; Pop stack"
      std::vector<StateNode *> scc;
      while (!_tarjan_stack.empty() && _tarjan_stack.top()->idx >= s->idx) {
        StateNode *w = _tarjan_stack.top();
        w->solved = true;
        w->idx = IDX_UNDEF;
        w->low = IDX_UNDEF;
        scc.push_back(w);
        _tarjan_stack.pop();

        if (_verbose)
          Logger::debug("Labeling state " + w->state.print() + " as solved");
      }
      _sccs.push_back(std::move(scc));

      _last_rv = true;
    } else {
      // Part of larger SCC, not root — just return flag
      _last_rv = f.flag;
    }

    // "return flag"
    dfs_stack.pop();
  }
}

// --- Accessors ---

SK_LDFS_SOLVER_TEMPLATE_DECL
bool SK_LDFS_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr && !si->terminal)) {
    return false;
  }
  return true;
}

SK_LDFS_SOLVER_TEMPLATE_DECL
const typename SK_LDFS_SOLVER_CLASS::Action &
SK_LDFS_SOLVER_CLASS::get_best_action(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr)) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return si->best_action->action;
}

SK_LDFS_SOLVER_TEMPLATE_DECL
typename SK_LDFS_SOLVER_CLASS::Value
SK_LDFS_SOLVER_CLASS::get_best_value(const State &s) const {
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

SK_LDFS_SOLVER_TEMPLATE_DECL
std::size_t SK_LDFS_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_LDFS_SOLVER_TEMPLATE_DECL
std::size_t SK_LDFS_SOLVER_CLASS::get_nb_tip_states() const {
  return _nb_tip_states;
}

SK_LDFS_SOLVER_TEMPLATE_DECL
std::size_t SK_LDFS_SOLVER_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

SK_LDFS_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_LDFS_SOLVER_CLASS::State>::Set
SK_LDFS_SOLVER_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored;
  for (const auto &sn : _graph) {
    explored.insert(sn.state);
  }
  return explored;
}

SK_LDFS_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_LDFS_SOLVER_CLASS::State>::Set
SK_LDFS_SOLVER_CLASS::get_solved_states() const {
  typename SetTypeDeducer<State>::Set solved;
  for (const auto &sn : _graph) {
    if (sn.solved) {
      solved.insert(sn.state);
    }
  }
  return solved;
}

SK_LDFS_SOLVER_TEMPLATE_DECL
std::vector<typename SetTypeDeducer<typename SK_LDFS_SOLVER_CLASS::State>::Set>
SK_LDFS_SOLVER_CLASS::get_strongly_connected_components() const {
  std::vector<typename SetTypeDeducer<State>::Set> result;
  for (const auto &scc : _sccs) {
    typename SetTypeDeducer<State>::Set component;
    for (const auto *sn : scc) {
      component.insert(sn->state);
    }
    result.push_back(std::move(component));
  }
  return result;
}

SK_LDFS_SOLVER_TEMPLATE_DECL
typename MapTypeDeducer<
    typename SK_LDFS_SOLVER_CLASS::State,
    std::pair<typename SK_LDFS_SOLVER_CLASS::Action, double>>::Map
SK_LDFS_SOLVER_CLASS::policy() const {
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map p;
  for (const auto &sn : _graph) {
    if (sn.best_action != nullptr) {
      p.insert(std::make_pair(sn.state, std::make_pair(sn.best_action->action,
                                                       (double)sn.best_value)));
    }
  }
  return p;
}

// --- IDAstarSolver::get_plan ---

#define SK_IDASTAR_SOLVER_TEMPLATE_DECL                                        \
  template <typename Tdomain, typename Texecution_policy>

#define SK_IDASTAR_SOLVER_CLASS IDAstarSolver<Tdomain, Texecution_policy>

SK_IDASTAR_SOLVER_TEMPLATE_DECL
SK_IDASTAR_SOLVER_CLASS::IDAstarSolver(
    Tdomain &domain,
    const typename SK_IDASTAR_SOLVER_CLASS::GoalCheckerFunctor &goal_checker,
    const typename SK_IDASTAR_SOLVER_CLASS::HeuristicFunctor &heuristic,
    std::size_t max_depth,
    const typename SK_IDASTAR_SOLVER_CLASS::CallbackFunctor &callback,
    bool verbose)
    : Base(
          domain, goal_checker, heuristic,
          [](const typename Tdomain::State &) {
            return typename Tdomain::Value(0.0, false);
          },
          1.0, 0.0, max_depth, callback, verbose) {}

SK_IDASTAR_SOLVER_TEMPLATE_DECL
SK_IDASTAR_SOLVER_CLASS::IDAstarSolver(
    Tdomain &domain,
    const typename SK_IDASTAR_SOLVER_CLASS::GoalCheckerFunctor &goal_checker,
    const typename SK_IDASTAR_SOLVER_CLASS::HeuristicFunctor &heuristic,
    const typename SK_IDASTAR_SOLVER_CLASS::Base::TerminalValueFunctor &,
    double, double, std::size_t max_depth,
    const typename SK_IDASTAR_SOLVER_CLASS::CallbackFunctor &callback,
    bool verbose)
    : Base(
          domain, goal_checker, heuristic,
          [](const typename Tdomain::State &) {
            return typename Tdomain::Value(0.0, false);
          },
          1.0, 0.0, max_depth, callback, verbose) {}

SK_IDASTAR_SOLVER_TEMPLATE_DECL
std::vector<typename Tdomain::Action>
SK_IDASTAR_SOLVER_CLASS::get_plan(const typename Tdomain::State &s) const {
  typedef typename Tdomain::Action Action;
  typedef typename LDFSSolver<Tdomain, Texecution_policy>::StateNode StateNode;
  typedef typename LDFSSolver<Tdomain, Texecution_policy>::Graph Graph;

  std::vector<Action> plan;
  auto si = this->_graph.find(s);
  while (si != this->_graph.end() && si->best_action != nullptr) {
    plan.push_back(si->best_action->action);
    if (si->best_action->outcomes.empty()) {
      break;
    }
    auto *next = std::get<2>(si->best_action->outcomes.front());
    if (next->goal || next->terminal) {
      break;
    }
    si = this->_graph.find(next->state);
  }
  return plan;
}

SK_LDFS_SOLVER_TEMPLATE_DECL
template <typename Params>
std::unique_ptr<SK_LDFS_SOLVER_CLASS> SK_LDFS_SOLVER_CLASS::create_from_params(
    Domain &domain,
    std::function<Predicate(Domain &, const State &)> goal_checker,
    std::function<Value(Domain &, const State &)> heuristic,
    std::function<Value(const State &)> terminal_value, const Params &params,
    bool verbose) {
  return std::make_unique<LDFSSolver>(
      domain, goal_checker, heuristic, terminal_value,
      params.template get<double>("discount", 1.0),
      params.template get<double>("epsilon", 0.001),
      params.template get<std::size_t>("max_depth", 0),
      CallbackFunctor([](const LDFSSolver &, Domain &) { return false; }),
      params.template get<bool>("verbose", verbose));
}

} // namespace skdecide

#endif // SKDECIDE_LDFS_IMPL_HH
