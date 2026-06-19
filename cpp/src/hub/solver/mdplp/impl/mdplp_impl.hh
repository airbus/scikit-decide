/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MDPLP_IMPL_HH
#define SKDECIDE_MDPLP_IMPL_HH

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <vector>

#include "Highs.h"
#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

#define SK_MDPLP_TEMPLATE_DECL                                                 \
  template <typename Tdomain, typename Texecution_policy>

#define SK_MDPLP_CLASS MDPLPSolver<Tdomain, Texecution_policy>

// --- StateNode / ActionNode ---

SK_MDPLP_TEMPLATE_DECL
SK_MDPLP_CLASS::StateNode::StateNode(const State &s)
    : state(s), best_action(nullptr), best_value(0.0), terminal(false),
      index(0) {}

SK_MDPLP_TEMPLATE_DECL
SK_MDPLP_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), value(0.0) {}

// --- Constructor ---

SK_MDPLP_TEMPLATE_DECL
SK_MDPLP_CLASS::MDPLPSolver(Domain &domain, const HeuristicFunctor &heuristic,
                            const TerminalValueFunctor &terminal_value,
                            LPVariant variant, double discount, double epsilon,
                            double lp_infinity, const CallbackFunctor &callback,
                            bool verbose)
    : _domain(domain), _heuristic(heuristic), _terminal_value(terminal_value),
      _variant(variant), _discount(discount), _epsilon(epsilon),
      _lp_infinity(lp_infinity), _callback(callback), _verbose(verbose),
      _nb_lp_variables(0), _nb_lp_constraints(0) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm MDPLP");
  }
}

SK_MDPLP_TEMPLATE_DECL
void SK_MDPLP_CLASS::clear() {
  _graph.clear();
  _non_terminal_states.clear();
  _nb_lp_variables = 0;
  _nb_lp_constraints = 0;
}

// --- Expand ---

SK_MDPLP_TEMPLATE_DECL
void SK_MDPLP_CLASS::expand(StateNode &s) {
  auto applicable_actions =
      _domain.get_applicable_actions(s.state).get_elements();

  std::for_each(
      ExecutionPolicy::policy, applicable_actions.begin(),
      applicable_actions.end(), [this, &s](auto a) {
        if (_verbose)
          Logger::debug("MDPLP expanding action: " + a.print() +
                        ExecutionPolicy::print_thread());
        _execution_policy.protect(
            [&s, &a] { s.actions.push_back(std::make_unique<ActionNode>(a)); });
        ActionNode &an = *(s.actions.back());

        auto next_states =
            _domain.get_next_state_distribution(s.state, a).get_values();
        for (auto ns : next_states) {
          std::pair<typename Graph::iterator, bool> si;
          _execution_policy.protect(
              [this, &si, &ns] { si = _graph.emplace(ns.state()); });
          StateNode &next_node = const_cast<StateNode &>(*(si.first));

          if (si.second) {
            if (_domain.is_terminal(ns.state())) {
              next_node.terminal = true;
              next_node.best_value = _terminal_value(ns.state()).cost();
            }
          }

          double cost =
              _domain.get_transition_value(s.state, a, next_node.state).cost();
          _execution_policy.protect([&an, &ns, cost, &next_node] {
            an.outcomes.push_back(
                std::make_tuple(ns.probability(), cost, &next_node));
          });
        }
      });
}

// --- BFS state enumeration ---

SK_MDPLP_TEMPLATE_DECL
void SK_MDPLP_CLASS::enumerate_reachable_states(const State &s) {
  std::queue<StateNode *> frontier;

  auto si = _graph.emplace(s);
  StateNode &root = const_cast<StateNode &>(*(si.first));
  if (si.second) {
    if (_domain.is_terminal(s)) {
      root.terminal = true;
      root.best_value = _terminal_value(s).cost();
    }
  }

  if (!root.terminal) {
    frontier.push(&root);
  }

  while (!frontier.empty()) {
    StateNode *current = frontier.front();
    frontier.pop();

    if (current->terminal || !current->actions.empty()) {
      continue;
    }

    expand(*current);

    for (const auto &an : current->actions) {
      for (const auto &outcome : an->outcomes) {
        StateNode *ns = std::get<2>(outcome);
        if (ns->actions.empty() && !ns->terminal) {
          frontier.push(ns);
        }
      }
    }
  }

  _non_terminal_states.clear();
  std::size_t idx = 0;
  for (auto &sn : _graph) {
    StateNode &node = const_cast<StateNode &>(sn);
    node.index = idx++;
    if (!node.terminal) {
      _non_terminal_states.push_back(&node);
    }
  }

  if (_verbose) {
    Logger::debug("MDPLP: enumerated " + StringConverter::from(_graph.size()) +
                  " states (" +
                  StringConverter::from(_non_terminal_states.size()) +
                  " non-terminal)");
  }
}

// --- Primal LP using HiGHS ---

SK_MDPLP_TEMPLATE_DECL
void SK_MDPLP_CLASS::solve_primal_lp() {
  Highs highs;
  highs.setOptionValue("output_flag", _verbose);

  std::size_t n_vars = _non_terminal_states.size();
  std::map<std::size_t, std::size_t> index_to_col;

  // Add variables: V(s) ≥ 0 for each non-terminal state
  for (std::size_t i = 0; i < _non_terminal_states.size(); ++i) {
    highs.addVar(0.0, _lp_infinity);
    index_to_col[_non_terminal_states[i]->index] = i;
  }

  // Objective: max Σ V(s) (primal for cost minimization)
  for (std::size_t i = 0; i < n_vars; ++i) {
    highs.changeColCost(static_cast<HighsInt>(i), 1.0);
  }
  highs.changeObjectiveSense(ObjSense::kMaximize);

  // Constraints: V(s) ≤ C(s,a) + γ Σ P(s'|s,a) V(s') for all s, a
  std::size_t n_constraints = 0;
  for (auto *sn : _non_terminal_states) {
    std::size_t col_s = index_to_col.at(sn->index);

    for (const auto &an : sn->actions) {
      // Aggregate coefficients by column to handle self-loops
      std::map<HighsInt, double> coeff_map;
      coeff_map[static_cast<HighsInt>(col_s)] = 1.0;

      double rhs_const = 0.0;
      bool cost_set = false;

      for (const auto &outcome : an->outcomes) {
        double prob = std::get<0>(outcome);
        double cost = std::get<1>(outcome);
        StateNode *ns = std::get<2>(outcome);

        if (!cost_set) {
          rhs_const = cost;
          cost_set = true;
        }

        if (ns->terminal) {
          rhs_const += prob * _discount * ns->best_value;
        } else {
          auto it = index_to_col.find(ns->index);
          if (it != index_to_col.end()) {
            coeff_map[static_cast<HighsInt>(it->second)] -= prob * _discount;
          }
        }
      }

      std::vector<HighsInt> cols;
      std::vector<double> vals;
      for (const auto &[c, v] : coeff_map) {
        cols.push_back(c);
        vals.push_back(v);
      }

      highs.addRow(-_lp_infinity, rhs_const, static_cast<HighsInt>(cols.size()),
                   cols.data(), vals.data());
      n_constraints++;
    }
  }

  _nb_lp_variables = n_vars;
  _nb_lp_constraints = n_constraints;

  if (_verbose) {
    Logger::debug("MDPLP primal: " + StringConverter::from(_nb_lp_variables) +
                  " variables, " + StringConverter::from(_nb_lp_constraints) +
                  " constraints");
  }

  highs.run();
  HighsModelStatus model_status = highs.getModelStatus();
  if (model_status != HighsModelStatus::kOptimal) {
    throw std::runtime_error("MDPLP primal LP not optimal: " +
                             highs.modelStatusToString(model_status));
  }

  const std::vector<double> &solution = highs.getSolution().col_value;
  for (auto *sn : _non_terminal_states) {
    sn->best_value = solution[index_to_col.at(sn->index)];
  }
}

// --- Dual LP using HiGHS ---

SK_MDPLP_TEMPLATE_DECL
void SK_MDPLP_CLASS::solve_dual_lp(const State &s0) {
  Highs highs;
  highs.setOptionValue("output_flag", _verbose);

  // Variables: x(s,a) ≥ 0 for each (state, action) pair
  struct SAVar {
    std::size_t col;
    StateNode *sn;
    ActionNode *an;
  };
  std::vector<SAVar> sa_vars;
  std::map<std::size_t, std::vector<std::size_t>> state_to_vars;

  std::size_t col = 0;
  for (auto *sn : _non_terminal_states) {
    for (auto &an : sn->actions) {
      highs.addVar(0.0, _lp_infinity);
      sa_vars.push_back({col, sn, an.get()});
      state_to_vars[sn->index].push_back(col);
      col++;
    }
  }

  std::size_t n_vars = col;

  // Objective: min Σ C(s,a) x(s,a)
  for (const auto &sa : sa_vars) {
    double cost = 0.0;
    if (!sa.an->outcomes.empty()) {
      cost = std::get<1>(sa.an->outcomes.front());
    }
    highs.changeColCost(static_cast<HighsInt>(sa.col), cost);
  }
  highs.changeObjectiveSense(ObjSense::kMinimize);

  // Flow conservation constraints:
  // Σ_a x(s,a) - γ Σ_{s',a'} P(s|s',a') x(s',a') = α(s)
  // Aggregate coefficients per column to handle self-loops (same variable
  // appears in both outflow and inflow terms).
  std::size_t n_constraints = 0;
  for (auto *sn : _non_terminal_states) {
    std::map<HighsInt, double> coeff_map;

    // Outflow: +1 for each x(s,a)
    for (std::size_t var_col : state_to_vars.at(sn->index)) {
      coeff_map[static_cast<HighsInt>(var_col)] += 1.0;
    }

    // Inflow: -γ P(s|s',a') for each x(s',a') that transitions to s
    for (const auto &sa : sa_vars) {
      for (const auto &outcome : sa.an->outcomes) {
        StateNode *ns = std::get<2>(outcome);
        if (ns->index == sn->index) {
          double prob = std::get<0>(outcome);
          coeff_map[static_cast<HighsInt>(sa.col)] -= _discount * prob;
        }
      }
    }

    std::vector<HighsInt> cols;
    std::vector<double> vals;
    for (const auto &[c, v] : coeff_map) {
      cols.push_back(c);
      vals.push_back(v);
    }

    double alpha = (typename State::Equal()(sn->state, s0)) ? 1.0 : 0.0;
    highs.addRow(alpha, alpha, static_cast<HighsInt>(cols.size()), cols.data(),
                 vals.data());
    n_constraints++;
  }

  _nb_lp_variables = n_vars;
  _nb_lp_constraints = n_constraints;

  if (_verbose) {
    Logger::debug("MDPLP dual: " + StringConverter::from(_nb_lp_variables) +
                  " variables, " + StringConverter::from(_nb_lp_constraints) +
                  " constraints");
  }

  highs.run();
  HighsModelStatus dual_model_status = highs.getModelStatus();
  if (dual_model_status != HighsModelStatus::kOptimal) {
    throw std::runtime_error("MDPLP dual LP not optimal: " +
                             highs.modelStatusToString(dual_model_status));
  }

  // V*(s) = row dual of flow conservation constraint (LP strong duality)
  const std::vector<double> &row_duals = highs.getSolution().row_dual;
  for (std::size_t i = 0; i < _non_terminal_states.size(); ++i) {
    _non_terminal_states[i]->best_value = row_duals[i];
  }

  extract_policy_from_values();
}

// --- Extract policy from values ---

SK_MDPLP_TEMPLATE_DECL
void SK_MDPLP_CLASS::extract_policy_from_values() {
  for (auto *sn : _non_terminal_states) {
    double best_q = std::numeric_limits<double>::infinity();
    for (auto &an : sn->actions) {
      double qval = 0.0;
      bool cost_set = false;
      for (const auto &outcome : an->outcomes) {
        if (!cost_set) {
          qval += std::get<1>(outcome);
          cost_set = true;
        }
        qval +=
            std::get<0>(outcome) * _discount * std::get<2>(outcome)->best_value;
      }
      an->value = qval;
      if (qval < best_q) {
        best_q = qval;
        sn->best_action = an.get();
        sn->best_value = qval;
      }
    }
  }
}

// --- Solve ---

SK_MDPLP_TEMPLATE_DECL
void SK_MDPLP_CLASS::solve(const State &s) {
  try {
    Logger::info(std::string("Running MDPLP solver (variant=") +
                 (_variant == LPVariant::Primal ? "primal" : "dual") +
                 ", LP backend=HiGHS)");
    _start_time = std::chrono::high_resolution_clock::now();

    enumerate_reachable_states(s);

    switch (_variant) {
    case LPVariant::Primal:
      solve_primal_lp();
      extract_policy_from_values();
      break;
    case LPVariant::Dual:
      solve_dual_lp(s);
      break;
    }

    Logger::info("MDPLP finished in " +
                 StringConverter::from((double)get_solving_time() / 1e3) +
                 " seconds with " + StringConverter::from(_graph.size()) +
                 " states, " + StringConverter::from(_nb_lp_variables) +
                 " LP variables, " + StringConverter::from(_nb_lp_constraints) +
                 " LP constraints.");
  } catch (const std::exception &e) {
    Logger::error("MDPLP failed: " + std::string(e.what()));
    throw;
  }
}

// --- Policy query ---

SK_MDPLP_TEMPLATE_DECL
bool SK_MDPLP_CLASS::is_solution_defined_for(const State &s) const {
  auto it = _graph.find(s);
  if (it == _graph.end())
    return false;
  return it->best_action != nullptr || it->terminal;
}

SK_MDPLP_TEMPLATE_DECL
const typename SK_MDPLP_CLASS::Action &
SK_MDPLP_CLASS::get_best_action(const State &s) const {
  auto it = _graph.find(s);
  if (it != _graph.end() && it->best_action != nullptr) {
    return it->best_action->action;
  }
  throw std::runtime_error(
      "SKDECIDE exception: no best action found in MDPLP policy.");
}

SK_MDPLP_TEMPLATE_DECL
typename SK_MDPLP_CLASS::Value
SK_MDPLP_CLASS::get_best_value(const State &s) const {
  auto it = _graph.find(s);
  if (it != _graph.end()) {
    Value val;
    val.cost(it->best_value);
    return val;
  }
  throw std::runtime_error("SKDECIDE exception: state not found in MDPLP.");
}

// --- Statistics ---

SK_MDPLP_TEMPLATE_DECL
std::size_t SK_MDPLP_CLASS::get_nb_states() const { return _graph.size(); }

SK_MDPLP_TEMPLATE_DECL
std::size_t SK_MDPLP_CLASS::get_nb_lp_variables() const {
  return _nb_lp_variables;
}

SK_MDPLP_TEMPLATE_DECL
std::size_t SK_MDPLP_CLASS::get_nb_lp_constraints() const {
  return _nb_lp_constraints;
}

SK_MDPLP_TEMPLATE_DECL
std::size_t SK_MDPLP_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

SK_MDPLP_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_MDPLP_CLASS::State>::Set
SK_MDPLP_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored;
  for (const auto &sn : _graph) {
    explored.insert(sn.state);
  }
  return explored;
}

// ===========================================================================
// SSPLPSolver implementation (undiscounted SSP with goals)
// ===========================================================================

#define SK_SSPLP_TEMPLATE_DECL                                                 \
  template <typename Tdomain, typename Texecution_policy>

#define SK_SSPLP_CLASS SSPLPSolver<Tdomain, Texecution_policy>

SK_SSPLP_TEMPLATE_DECL
SK_SSPLP_CLASS::StateNode::StateNode(const State &s)
    : state(s), best_action(nullptr), best_value(0.0), terminal(false),
      goal(false), index(0) {}

SK_SSPLP_TEMPLATE_DECL
SK_SSPLP_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), value(0.0) {}

SK_SSPLP_TEMPLATE_DECL
SK_SSPLP_CLASS::SSPLPSolver(Domain &domain,
                            const GoalCheckerFunctor &goal_checker,
                            const HeuristicFunctor &heuristic,
                            LPVariant variant, double epsilon,
                            double lp_infinity, const CallbackFunctor &callback,
                            bool verbose)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _variant(variant), _epsilon(epsilon), _lp_infinity(lp_infinity),
      _callback(callback), _verbose(verbose), _nb_lp_variables(0),
      _nb_lp_constraints(0) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm SSPLP");
  }
}

SK_SSPLP_TEMPLATE_DECL
void SK_SSPLP_CLASS::clear() {
  _graph.clear();
  _non_goal_states.clear();
  _nb_lp_variables = 0;
  _nb_lp_constraints = 0;
}

SK_SSPLP_TEMPLATE_DECL
void SK_SSPLP_CLASS::expand(StateNode &s) {
  auto applicable_actions =
      _domain.get_applicable_actions(s.state).get_elements();

  std::for_each(
      ExecutionPolicy::policy, applicable_actions.begin(),
      applicable_actions.end(), [this, &s](auto a) {
        if (_verbose)
          Logger::debug("SSPLP expanding action: " + a.print() +
                        ExecutionPolicy::print_thread());
        _execution_policy.protect(
            [&s, &a] { s.actions.push_back(std::make_unique<ActionNode>(a)); });
        ActionNode &an = *(s.actions.back());

        auto next_states =
            _domain.get_next_state_distribution(s.state, a).get_values();
        for (auto ns : next_states) {
          std::pair<typename Graph::iterator, bool> si;
          _execution_policy.protect(
              [this, &si, &ns] { si = _graph.emplace(ns.state()); });
          StateNode &next_node = const_cast<StateNode &>(*(si.first));

          if (si.second) {
            bool is_goal = _goal_checker(_domain, ns.state());
            bool is_terminal = _domain.is_terminal(ns.state());
            next_node.goal = is_goal;
            next_node.terminal = is_terminal || is_goal;
            next_node.best_value = 0.0;
          }

          double cost =
              _domain.get_transition_value(s.state, a, next_node.state).cost();
          _execution_policy.protect([&an, &ns, cost, &next_node] {
            an.outcomes.push_back(
                std::make_tuple(ns.probability(), cost, &next_node));
          });
        }
      });
}

SK_SSPLP_TEMPLATE_DECL
void SK_SSPLP_CLASS::enumerate_reachable_states(const State &s) {
  std::queue<StateNode *> frontier;

  auto si = _graph.emplace(s);
  StateNode &root = const_cast<StateNode &>(*(si.first));
  if (si.second) {
    bool is_goal = _goal_checker(_domain, s);
    bool is_terminal = _domain.is_terminal(s);
    root.goal = is_goal;
    root.terminal = is_terminal || is_goal;
    root.best_value = 0.0;
  }

  if (!root.terminal) {
    frontier.push(&root);
  }

  while (!frontier.empty()) {
    StateNode *current = frontier.front();
    frontier.pop();

    if (current->terminal || !current->actions.empty()) {
      continue;
    }

    expand(*current);

    for (const auto &an : current->actions) {
      for (const auto &outcome : an->outcomes) {
        StateNode *ns = std::get<2>(outcome);
        if (ns->actions.empty() && !ns->terminal) {
          frontier.push(ns);
        }
      }
    }
  }

  _non_goal_states.clear();
  std::size_t idx = 0;
  for (auto &sn : _graph) {
    StateNode &node = const_cast<StateNode &>(sn);
    node.index = idx++;
    if (!node.terminal) {
      _non_goal_states.push_back(&node);
    }
  }

  if (_verbose) {
    Logger::debug("SSPLP: enumerated " + StringConverter::from(_graph.size()) +
                  " states (" + StringConverter::from(_non_goal_states.size()) +
                  " non-goal)");
  }
}

// Primal LP for SSP (undiscounted, γ=1):
// max  Σ V(s)
// s.t. V(s) ≤ C(s,a) + Σ P(s'|s,a) V(s')   ∀ non-goal s, a
//      V(g) = 0                                ∀ goal g
SK_SSPLP_TEMPLATE_DECL
void SK_SSPLP_CLASS::solve_primal_lp() {
  Highs highs;
  highs.setOptionValue("output_flag", _verbose);

  std::size_t n_vars = _non_goal_states.size();
  std::map<std::size_t, std::size_t> index_to_col;

  for (std::size_t i = 0; i < _non_goal_states.size(); ++i) {
    highs.addVar(0.0, _lp_infinity);
    index_to_col[_non_goal_states[i]->index] = i;
  }

  for (std::size_t i = 0; i < n_vars; ++i) {
    highs.changeColCost(static_cast<HighsInt>(i), 1.0);
  }
  highs.changeObjectiveSense(ObjSense::kMaximize);

  std::size_t n_constraints = 0;
  for (auto *sn : _non_goal_states) {
    std::size_t col_s = index_to_col.at(sn->index);

    for (const auto &an : sn->actions) {
      std::map<HighsInt, double> coeff_map;
      coeff_map[static_cast<HighsInt>(col_s)] = 1.0;

      double rhs_const = 0.0;
      bool cost_set = false;

      for (const auto &outcome : an->outcomes) {
        double prob = std::get<0>(outcome);
        double cost = std::get<1>(outcome);
        StateNode *ns = std::get<2>(outcome);

        if (!cost_set) {
          rhs_const = cost;
          cost_set = true;
        }

        // Goal/terminal states have V=0, so no contribution
        if (!ns->terminal) {
          auto it = index_to_col.find(ns->index);
          if (it != index_to_col.end()) {
            coeff_map[static_cast<HighsInt>(it->second)] -= prob;
          }
        }
      }

      std::vector<HighsInt> cols;
      std::vector<double> vals;
      for (const auto &[c, v] : coeff_map) {
        cols.push_back(c);
        vals.push_back(v);
      }

      highs.addRow(-_lp_infinity, rhs_const, static_cast<HighsInt>(cols.size()),
                   cols.data(), vals.data());
      n_constraints++;
    }
  }

  _nb_lp_variables = n_vars;
  _nb_lp_constraints = n_constraints;

  if (_verbose) {
    Logger::debug("SSPLP primal: " + StringConverter::from(_nb_lp_variables) +
                  " variables, " + StringConverter::from(_nb_lp_constraints) +
                  " constraints");
  }

  highs.run();
  HighsModelStatus model_status = highs.getModelStatus();
  if (model_status != HighsModelStatus::kOptimal) {
    throw std::runtime_error("SSPLP primal LP not optimal: " +
                             highs.modelStatusToString(model_status));
  }

  const std::vector<double> &solution = highs.getSolution().col_value;
  for (auto *sn : _non_goal_states) {
    sn->best_value = solution[index_to_col.at(sn->index)];
  }
}

// Dual LP for SSP (undiscounted, γ=1):
// min  Σ C(s,a) x(s,a)
// s.t. Σ_a x(s,a) - Σ_{s',a'} P(s|s',a') x(s',a') = α(s)  ∀ non-goal s
//      x(s,a) ≥ 0
SK_SSPLP_TEMPLATE_DECL
void SK_SSPLP_CLASS::solve_dual_lp(const State &s0) {
  Highs highs;
  highs.setOptionValue("output_flag", _verbose);

  struct SAVar {
    std::size_t col;
    StateNode *sn;
    ActionNode *an;
  };
  std::vector<SAVar> sa_vars;
  std::map<std::size_t, std::vector<std::size_t>> state_to_vars;

  std::size_t col = 0;
  for (auto *sn : _non_goal_states) {
    for (auto &an : sn->actions) {
      highs.addVar(0.0, _lp_infinity);
      sa_vars.push_back({col, sn, an.get()});
      state_to_vars[sn->index].push_back(col);
      col++;
    }
  }

  std::size_t n_vars = col;

  for (const auto &sa : sa_vars) {
    double cost = 0.0;
    if (!sa.an->outcomes.empty()) {
      cost = std::get<1>(sa.an->outcomes.front());
    }
    highs.changeColCost(static_cast<HighsInt>(sa.col), cost);
  }
  highs.changeObjectiveSense(ObjSense::kMinimize);

  // Aggregate coefficients per column to handle self-loops
  std::size_t n_constraints = 0;
  for (auto *sn : _non_goal_states) {
    std::map<HighsInt, double> coeff_map;

    for (std::size_t var_col : state_to_vars.at(sn->index)) {
      coeff_map[static_cast<HighsInt>(var_col)] += 1.0;
    }

    for (const auto &sa : sa_vars) {
      for (const auto &outcome : sa.an->outcomes) {
        StateNode *ns = std::get<2>(outcome);
        if (ns->index == sn->index) {
          double prob = std::get<0>(outcome);
          coeff_map[static_cast<HighsInt>(sa.col)] -= prob;
        }
      }
    }

    std::vector<HighsInt> cols;
    std::vector<double> vals;
    for (const auto &[c, v] : coeff_map) {
      cols.push_back(c);
      vals.push_back(v);
    }

    double alpha = (typename State::Equal()(sn->state, s0)) ? 1.0 : 0.0;
    highs.addRow(alpha, alpha, static_cast<HighsInt>(cols.size()), cols.data(),
                 vals.data());
    n_constraints++;
  }

  _nb_lp_variables = n_vars;
  _nb_lp_constraints = n_constraints;

  if (_verbose) {
    Logger::debug("SSPLP dual: " + StringConverter::from(_nb_lp_variables) +
                  " variables, " + StringConverter::from(_nb_lp_constraints) +
                  " constraints");
  }

  highs.run();
  HighsModelStatus dual_model_status = highs.getModelStatus();
  if (dual_model_status != HighsModelStatus::kOptimal) {
    throw std::runtime_error("SSPLP dual LP not optimal: " +
                             highs.modelStatusToString(dual_model_status));
  }

  // V*(s) = row dual of flow conservation constraint (LP strong duality)
  const std::vector<double> &row_duals = highs.getSolution().row_dual;
  for (std::size_t i = 0; i < _non_goal_states.size(); ++i) {
    _non_goal_states[i]->best_value = row_duals[i];
  }

  extract_policy_from_values();
}

SK_SSPLP_TEMPLATE_DECL
void SK_SSPLP_CLASS::extract_policy_from_values() {
  for (auto *sn : _non_goal_states) {
    double best_q = std::numeric_limits<double>::infinity();
    for (auto &an : sn->actions) {
      double qval = 0.0;
      bool cost_set = false;
      for (const auto &outcome : an->outcomes) {
        if (!cost_set) {
          qval += std::get<1>(outcome);
          cost_set = true;
        }
        qval += std::get<0>(outcome) * std::get<2>(outcome)->best_value;
      }
      an->value = qval;
      if (qval < best_q) {
        best_q = qval;
        sn->best_action = an.get();
        sn->best_value = qval;
      }
    }
  }
}

SK_SSPLP_TEMPLATE_DECL
void SK_SSPLP_CLASS::solve(const State &s) {
  try {
    Logger::info(std::string("Running SSPLP solver (variant=") +
                 (_variant == LPVariant::Primal ? "primal" : "dual") +
                 ", LP backend=HiGHS)");
    _start_time = std::chrono::high_resolution_clock::now();

    enumerate_reachable_states(s);

    switch (_variant) {
    case LPVariant::Primal:
      solve_primal_lp();
      extract_policy_from_values();
      break;
    case LPVariant::Dual:
      solve_dual_lp(s);
      break;
    }

    Logger::info("SSPLP finished in " +
                 StringConverter::from((double)get_solving_time() / 1e3) +
                 " seconds with " + StringConverter::from(_graph.size()) +
                 " states, " + StringConverter::from(_nb_lp_variables) +
                 " LP variables, " + StringConverter::from(_nb_lp_constraints) +
                 " LP constraints.");
  } catch (const std::exception &e) {
    Logger::error("SSPLP failed: " + std::string(e.what()));
    throw;
  }
}

SK_SSPLP_TEMPLATE_DECL
bool SK_SSPLP_CLASS::is_solution_defined_for(const State &s) const {
  auto it = _graph.find(s);
  if (it == _graph.end())
    return false;
  return it->best_action != nullptr || it->terminal;
}

SK_SSPLP_TEMPLATE_DECL
const typename SK_SSPLP_CLASS::Action &
SK_SSPLP_CLASS::get_best_action(const State &s) const {
  auto it = _graph.find(s);
  if (it != _graph.end() && it->best_action != nullptr) {
    return it->best_action->action;
  }
  throw std::runtime_error(
      "SKDECIDE exception: no best action found in SSPLP policy.");
}

SK_SSPLP_TEMPLATE_DECL
typename SK_SSPLP_CLASS::Value
SK_SSPLP_CLASS::get_best_value(const State &s) const {
  auto it = _graph.find(s);
  if (it != _graph.end()) {
    Value val;
    val.cost(it->best_value);
    return val;
  }
  throw std::runtime_error("SKDECIDE exception: state not found in SSPLP.");
}

SK_SSPLP_TEMPLATE_DECL
std::size_t SK_SSPLP_CLASS::get_nb_states() const { return _graph.size(); }

SK_SSPLP_TEMPLATE_DECL
std::size_t SK_SSPLP_CLASS::get_nb_lp_variables() const {
  return _nb_lp_variables;
}

SK_SSPLP_TEMPLATE_DECL
std::size_t SK_SSPLP_CLASS::get_nb_lp_constraints() const {
  return _nb_lp_constraints;
}

SK_SSPLP_TEMPLATE_DECL
std::size_t SK_SSPLP_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

SK_SSPLP_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_SSPLP_CLASS::State>::Set
SK_SSPLP_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored;
  for (const auto &sn : _graph) {
    explored.insert(sn.state);
  }
  return explored;
}

} // namespace skdecide

#endif // SKDECIDE_MDPLP_IMPL_HH
