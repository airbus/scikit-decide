/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_IDUAL_IMPL_HH
#define SKDECIDE_IDUAL_IMPL_HH

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include "Highs.h"
#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

#define SK_IDUAL_TEMPLATE_DECL                                                 \
  template <typename Tdomain, typename Texecution_policy>

#define SK_IDUAL_CLASS IDualSolver<Tdomain, Texecution_policy>

SK_IDUAL_TEMPLATE_DECL
SK_IDUAL_CLASS::StateNode::StateNode(const State &s)
    : state(s), best_action(nullptr), best_value(0.0), expanded(false),
      goal(false), terminal(false), index(0) {}

SK_IDUAL_TEMPLATE_DECL
SK_IDUAL_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), value(0.0) {}

SK_IDUAL_TEMPLATE_DECL
SK_IDUAL_CLASS::IDualSolver(
    Domain &domain, const GoalCheckerFunctor &goal_checker,
    const HeuristicFunctor &heuristic,
    const TerminalValueFunctor &terminal_value,
    const SecondaryHeuristicFunctor &secondary_heuristic,
    const std::vector<double> &dead_end_costs, double epsilon,
    double lp_infinity, double lp_tolerance, double default_dead_end_cost,
    std::size_t lp_callback_interval, const CallbackFunctor &callback,
    bool verbose)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _terminal_value(terminal_value),
      _secondary_heuristic(secondary_heuristic), _epsilon(epsilon),
      _lp_infinity(lp_infinity), _lp_tolerance(lp_tolerance),
      _default_dead_end_cost(default_dead_end_cost),
      _lp_callback_interval(lp_callback_interval), _callback(callback),
      _verbose(verbose), _n_constraints(0), _dead_end_costs(dead_end_costs),
      _nb_lp_iterations(0) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm IDual");
  }

  if constexpr (has_get_constraints<Domain>::value) {
    auto constraints = _domain.get_constraints();
    _n_constraints = constraints.size();
    for (auto &c : constraints) {
      _cost_bounds.push_back(c.bound);
    }
    if (_dead_end_costs.empty()) {
      _dead_end_costs.resize(_n_constraints, _default_dead_end_cost);
    }
  }
}

SK_IDUAL_TEMPLATE_DECL
void SK_IDUAL_CLASS::clear() {
  _graph.clear();
  _nb_lp_iterations = 0;
  _highs.reset();
  _lp_col_info.clear();
  _lp_state_sa_cols.clear();
  _lp_state_xd_col.clear();
  _lp_flow_row.clear();
  _lp_c9_row = -1;
  _lp_c11_rows.clear();
  _lp_succ_to_cols.clear();
  _lp_col_obj.clear();
  _lp_col_c9.clear();
  _lp_col_c11.clear();
  _lp_initialized = false;
}

SK_IDUAL_TEMPLATE_DECL
void SK_IDUAL_CLASS::expand_states(std::vector<StateNode *> &fr) {
  for (auto *sn : fr) {
    if (sn->expanded || sn->goal || sn->terminal)
      continue;

    auto applicable_actions =
        _domain.get_applicable_actions(sn->state).get_elements();

    std::for_each(
        ExecutionPolicy::policy, applicable_actions.begin(),
        applicable_actions.end(), [this, sn](auto a) {
          _execution_policy.protect([sn, &a] {
            sn->actions.push_back(std::make_unique<ActionNode>(a));
          });
          ActionNode &an = *(sn->actions.back());

          auto next_states =
              _domain.get_next_state_distribution(sn->state, a).get_values();
          for (auto ns : next_states) {
            std::pair<typename Graph::iterator, bool> si;
            _execution_policy.protect(
                [this, &si, &ns] { si = _graph.emplace(ns.state()); });
            StateNode &next_node = const_cast<StateNode &>(*(si.first));

            if (si.second) {
              if (_goal_checker(_domain, ns.state())) {
                next_node.goal = true;
                next_node.terminal = true;
                next_node.best_value = 0.0;
              } else if (_domain.is_terminal(ns.state())) {
                next_node.terminal = true;
                next_node.best_value = _terminal_value(ns.state()).cost();
              }
            }

            double cost =
                _domain.get_transition_value(sn->state, a, next_node.state)
                    .cost();
            _execution_policy.protect([&an, &ns, cost, &next_node] {
              an.outcomes.push_back(
                  std::make_tuple(ns.probability(), cost, &next_node));
            });
          }

          if constexpr (has_get_constraints<Domain>::value) {
            auto constraints = _domain.get_constraints();
            an.secondary_costs.resize(constraints.size());
            for (std::size_t j = 0; j < constraints.size(); ++j) {
              an.secondary_costs[j] =
                  constraints[j].evaluate(sn->state, an.action);
            }
          }
        });

    sn->expanded = true;
  }
}

// --- Helper: compute objective cost for a column x(s,a) ---

SK_IDUAL_TEMPLATE_DECL
double SK_IDUAL_CLASS::compute_sa_obj_cost(ActionNode *an) const {
  double cost = 0.0;
  for (const auto &outcome : an->outcomes) {
    StateNode *ns = std::get<2>(outcome);
    double prob = std::get<0>(outcome);
    double tcost = std::get<1>(outcome);
    cost += prob * tcost;
    if (!ns->expanded || ns->goal) {
      if (!ns->goal) {
        if (ns->terminal) {
          cost += prob * ns->best_value;
        } else {
          cost += prob * _heuristic(_domain, ns->state).cost();
        }
      }
    }
  }
  return cost;
}

// --- Helper: compute C9 coefficient for a column x(s,a) ---

SK_IDUAL_TEMPLATE_DECL
double SK_IDUAL_CLASS::compute_sa_c9_coeff(ActionNode *an) const {
  double c9 = 0.0;
  for (const auto &outcome : an->outcomes) {
    StateNode *ns = std::get<2>(outcome);
    double prob = std::get<0>(outcome);
    if (!ns->expanded || ns->goal) {
      c9 += prob;
    }
  }
  return c9;
}

// --- Helper: add a column for x(s,a) and record tracking data ---

SK_IDUAL_TEMPLATE_DECL
void SK_IDUAL_CLASS::add_sa_column(StateNode *sn, ActionNode *an,
                                   const State &s0) {
  double cost = compute_sa_obj_cost(an);
  double c9_coeff = compute_sa_c9_coeff(an);

  std::map<HighsInt, double> coeff_map;

  // Outflow: +1 in this state's flow row
  coeff_map[_lp_flow_row.at(sn)] += 1.0;

  // Inflow: -prob in each expanded successor's flow row
  for (const auto &outcome : an->outcomes) {
    StateNode *ns = std::get<2>(outcome);
    double prob = std::get<0>(outcome);
    if (ns->expanded && !ns->goal) {
      auto it = _lp_flow_row.find(ns);
      if (it != _lp_flow_row.end()) {
        coeff_map[it->second] -= prob;
      }
    }
  }

  // C9
  if (std::abs(c9_coeff) > _lp_tolerance) {
    coeff_map[_lp_c9_row] += c9_coeff;
  }

  // C11
  if constexpr (has_get_constraints<Domain>::value) {
    for (std::size_t j = 0; j < _n_constraints; ++j) {
      double cj = an->secondary_costs[j];
      for (const auto &outcome : an->outcomes) {
        StateNode *ns = std::get<2>(outcome);
        double prob = std::get<0>(outcome);
        if (!ns->expanded || ns->goal) {
          if (!ns->goal) {
            if (ns->terminal) {
              cj += prob * _dead_end_costs[j];
            } else if (_secondary_heuristic) {
              cj += prob * _secondary_heuristic(_domain, ns->state, j);
            }
          }
        }
      }
      _lp_col_c11[j].push_back(cj);
      if (std::abs(cj) > _lp_tolerance) {
        coeff_map[_lp_c11_rows[j]] += cj;
      }
    }
  }

  std::vector<HighsInt> indices;
  std::vector<double> values;
  for (const auto &[row, val] : coeff_map) {
    if (std::abs(val) > _lp_tolerance) {
      indices.push_back(row);
      values.push_back(val);
    }
  }

  HighsInt new_col = static_cast<HighsInt>(_lp_col_info.size());
  _highs->addCol(cost, 0.0, _lp_infinity, static_cast<HighsInt>(indices.size()),
                 indices.data(), values.data());

  _lp_col_info.push_back({sn, an});
  _lp_state_sa_cols[sn].push_back(new_col);
  _lp_col_obj.push_back(cost);
  _lp_col_c9.push_back(c9_coeff);

  // Update successor index
  for (const auto &outcome : an->outcomes) {
    StateNode *ns = std::get<2>(outcome);
    double prob = std::get<0>(outcome);
    _lp_succ_to_cols[ns].push_back({new_col, prob});
  }
}

// --- Helper: add x_D(s) column ---

SK_IDUAL_TEMPLATE_DECL
void SK_IDUAL_CLASS::add_xd_column(StateNode *sn) {
  double tv = _terminal_value(sn->state).cost();

  std::vector<HighsInt> indices;
  std::vector<double> values;

  // Outflow: +1 in flow row
  indices.push_back(_lp_flow_row.at(sn));
  values.push_back(1.0);

  // C9: +1
  indices.push_back(_lp_c9_row);
  values.push_back(1.0);

  // C11: dead_end_costs[j]
  if constexpr (has_get_constraints<Domain>::value) {
    for (std::size_t j = 0; j < _n_constraints; ++j) {
      indices.push_back(_lp_c11_rows[j]);
      values.push_back(_dead_end_costs[j]);
    }
  }

  HighsInt new_col = static_cast<HighsInt>(_lp_col_info.size());
  _highs->addCol(tv, 0.0, _lp_infinity, static_cast<HighsInt>(indices.size()),
                 indices.data(), values.data());

  _lp_col_info.push_back({sn, nullptr});
  _lp_state_xd_col[sn] = new_col;
  _lp_col_obj.push_back(tv);
  _lp_col_c9.push_back(1.0);

  if constexpr (has_get_constraints<Domain>::value) {
    for (std::size_t j = 0; j < _n_constraints; ++j) {
      _lp_col_c11[j].push_back(_dead_end_costs[j]);
    }
  }
}

// --- init_lp: build full LP on first iteration ---

SK_IDUAL_TEMPLATE_DECL
void SK_IDUAL_CLASS::init_lp(const State &s0) {
  _highs = std::make_unique<Highs>();
  _highs->setOptionValue("output_flag", _verbose);
  _highs->changeObjectiveSense(ObjSense::kMinimize);

  _lp_col_info.clear();
  _lp_state_sa_cols.clear();
  _lp_state_xd_col.clear();
  _lp_flow_row.clear();
  _lp_c11_rows.clear();
  _lp_succ_to_cols.clear();
  _lp_col_obj.clear();
  _lp_col_c9.clear();
  _lp_col_c11.clear();

  if constexpr (has_get_constraints<Domain>::value) {
    _lp_col_c11.resize(_n_constraints);
  }

  // Identify expanded states
  std::vector<StateNode *> expanded_states;
  for (auto &sn_ref : _graph) {
    StateNode &sn = const_cast<StateNode &>(sn_ref);
    if (sn.expanded && !sn.goal) {
      expanded_states.push_back(&sn);
    }
  }

  // Phase 1: Add flow conservation rows (empty initially, columns will
  // reference them)
  for (auto *sn : expanded_states) {
    double alpha = (typename State::Equal()(sn->state, s0)) ? 1.0 : 0.0;
    HighsInt row = static_cast<HighsInt>(_lp_flow_row.size());
    _highs->addRow(alpha, alpha, 0, nullptr, nullptr);
    _lp_flow_row[sn] = row;
  }

  // Phase 2: Add C9 row (empty, columns will populate it)
  _lp_c9_row = static_cast<HighsInt>(_lp_flow_row.size());
  _highs->addRow(1.0, 1.0, 0, nullptr, nullptr);

  // Phase 3: Add C11 rows (empty)
  if constexpr (has_get_constraints<Domain>::value) {
    for (std::size_t j = 0; j < _n_constraints; ++j) {
      HighsInt c11_row = static_cast<HighsInt>(_lp_flow_row.size() + 1 + j);
      _highs->addRow(-_lp_infinity, _cost_bounds[j], 0, nullptr, nullptr);
      _lp_c11_rows.push_back(c11_row);
    }
  }

  // Phase 4: Add columns (x(s,a) and x_D(s)) — these reference the rows above
  for (auto *sn : expanded_states) {
    for (auto &an_ptr : sn->actions) {
      add_sa_column(sn, an_ptr.get(), s0);
    }
    add_xd_column(sn);
  }

  if (_verbose) {
    Logger::debug(
        "IDual init_lp: " + StringConverter::from(_lp_col_info.size()) +
        " variables, " + StringConverter::from(expanded_states.size()) +
        " expanded states");
  }
}

// --- update_lp: incremental LP update ---

SK_IDUAL_TEMPLATE_DECL
void SK_IDUAL_CLASS::update_lp(const State &s0,
                               const std::vector<StateNode *> &newly_expanded) {

  // Phase 1: Update existing columns' objective coefficients
  for (auto *T : newly_expanded) {
    double h_T;
    if (T->goal) {
      h_T = 0.0;
    } else if (T->terminal) {
      h_T = T->best_value;
    } else {
      h_T = _heuristic(_domain, T->state).cost();
    }

    auto it = _lp_succ_to_cols.find(T);
    if (it != _lp_succ_to_cols.end()) {
      for (const auto &entry : it->second) {
        _lp_col_obj[entry.col] -= entry.prob * h_T;
        _highs->changeColCost(entry.col, _lp_col_obj[entry.col]);
      }
    }
  }

  // Phase 2: Update existing columns' C9 coefficients
  for (auto *T : newly_expanded) {
    auto it = _lp_succ_to_cols.find(T);
    if (it != _lp_succ_to_cols.end()) {
      for (const auto &entry : it->second) {
        _lp_col_c9[entry.col] -= entry.prob;
        _highs->changeCoeff(_lp_c9_row, entry.col, _lp_col_c9[entry.col]);
      }
    }
  }

  // Phase 3: Update existing columns' C11 coefficients
  if constexpr (has_get_constraints<Domain>::value) {
    for (auto *T : newly_expanded) {
      auto it = _lp_succ_to_cols.find(T);
      if (it != _lp_succ_to_cols.end()) {
        for (const auto &entry : it->second) {
          for (std::size_t j = 0; j < _n_constraints; ++j) {
            double sec_h;
            if (T->goal) {
              sec_h = 0.0;
            } else if (T->terminal) {
              sec_h = _dead_end_costs[j];
            } else if (_secondary_heuristic) {
              sec_h = _secondary_heuristic(_domain, T->state, j);
            } else {
              sec_h = 0.0;
            }
            _lp_col_c11[j][entry.col] -= entry.prob * sec_h;
            _highs->changeCoeff(_lp_c11_rows[j], entry.col,
                                _lp_col_c11[j][entry.col]);
          }
        }
      }
    }
  }

  // Phase 4: Add flow conservation rows for newly expanded states
  for (auto *T : newly_expanded) {
    std::vector<HighsInt> indices;
    std::vector<double> values;

    // Inflow from existing columns that transition to T
    auto it = _lp_succ_to_cols.find(T);
    if (it != _lp_succ_to_cols.end()) {
      for (const auto &entry : it->second) {
        indices.push_back(entry.col);
        values.push_back(-entry.prob);
      }
    }

    double alpha = (typename State::Equal()(T->state, s0)) ? 1.0 : 0.0;
    HighsInt new_row = _highs->getNumRow();
    _highs->addRow(alpha, alpha, static_cast<HighsInt>(indices.size()),
                   indices.data(), values.data());
    _lp_flow_row[T] = new_row;
  }

  // Phase 5: Add new columns for newly expanded states
  for (auto *T : newly_expanded) {
    for (auto &an_ptr : T->actions) {
      add_sa_column(T, an_ptr.get(), s0);
    }
    add_xd_column(T);
  }

  if (_verbose) {
    Logger::debug(
        "IDual update_lp: " + StringConverter::from(newly_expanded.size()) +
        " newly expanded, " + StringConverter::from(_lp_col_info.size()) +
        " total variables");
  }
}

// --- extract_solution: get values, policy, fringe from solved LP ---

SK_IDUAL_TEMPLATE_DECL
void SK_IDUAL_CLASS::extract_solution() {
  const std::vector<double> &col_values = _highs->getSolution().col_value;
  const std::vector<double> &row_duals = _highs->getSolution().row_dual;
  double mu = row_duals[_lp_c9_row];

  // Extract V*(s) from row duals
  for (const auto &[sn, flow_row] : _lp_flow_row) {
    sn->best_value = row_duals[flow_row] + mu;
  }

  // Extract policy
  for (const auto &[sn, sa_cols] : _lp_state_sa_cols) {
    double out_s = 0.0;
    for (HighsInt col : sa_cols) {
      out_s += col_values[col];
    }
    out_s += col_values[_lp_state_xd_col.at(sn)];

    sn->action_probabilities.clear();
    sn->best_action = nullptr;
    double best_x = 0.0;

    if (out_s > _epsilon) {
      for (HighsInt col : sa_cols) {
        double x_val = col_values[col];
        if (x_val > _epsilon) {
          ActionNode *an = _lp_col_info[col].an;
          double prob = x_val / out_s;
          sn->action_probabilities.push_back({an, prob});
          if (x_val > best_x) {
            best_x = x_val;
            sn->best_action = an;
          }
        }
      }
    }
  }

  // Compute fringe reachable: sink states with positive inflow
  _fringe_reachable.clear();
  for (auto &sn_ref : _graph) {
    StateNode &sn = const_cast<StateNode &>(sn_ref);
    if (sn.expanded || sn.goal || sn.terminal)
      continue;

    auto it = _lp_succ_to_cols.find(&sn);
    if (it == _lp_succ_to_cols.end())
      continue;

    double inflow = 0.0;
    for (const auto &entry : it->second) {
      inflow += col_values[entry.col] * entry.prob;
    }
    if (inflow > _epsilon) {
      _fringe_reachable.push_back(&sn);
    }
  }
}

SK_IDUAL_TEMPLATE_DECL
void SK_IDUAL_CLASS::solve(const State &s) {
  try {
    Logger::info("Running IDual solver");
    _start_time = std::chrono::high_resolution_clock::now();
    _nb_lp_iterations = 0;

    // Initialize: Ŝ = {s0}, F = {s0}, FR = {s0}
    auto si = _graph.emplace(s);
    StateNode &root = const_cast<StateNode &>(*(si.first));
    if (si.second) {
      if (_goal_checker(_domain, s)) {
        root.goal = true;
        root.terminal = true;
        root.best_value = 0.0;
      } else if (_domain.is_terminal(s)) {
        root.terminal = true;
        root.best_value = _terminal_value(s).cost();
      }
    }

    if (root.goal) {
      Logger::info("IDual: initial state is a goal");
      return;
    }

    _fringe_reachable.clear();
    _fringe_reachable.push_back(&root);
    _lp_initialized = false;

    struct LPInterruptData {
      std::size_t interval;
      std::size_t last_interrupt_iter = 0;
    };

    while (!_fringe_reachable.empty()) {
      auto to_expand = _fringe_reachable;
      expand_states(_fringe_reachable);

      std::vector<StateNode *> newly_expanded;
      for (auto *sn : to_expand)
        if (sn->expanded)
          newly_expanded.push_back(sn);

      if (!_lp_initialized) {
        init_lp(s);
        _lp_initialized = true;
      } else {
        update_lp(s, newly_expanded);
      }

      if (_lp_callback_interval > 0) {
        LPInterruptData cb_data{_lp_callback_interval};
        _highs->setCallback(
            [](const int, const std::string &,
               const HighsCallbackDataOut *data_out,
               HighsCallbackDataIn *data_in, void *user_data) {
              auto *d = static_cast<LPInterruptData *>(user_data);
              auto iter =
                  static_cast<std::size_t>(data_out->simplex_iteration_count);
              if (iter - d->last_interrupt_iter >= d->interval) {
                d->last_interrupt_iter = iter;
                data_in->user_interrupt = 1;
              }
            },
            &cb_data);
        _highs->startCallback(kCallbackSimplexInterrupt);
      }

      bool user_stopped = false;
      auto run_lp_with_callbacks = [this, &user_stopped]() -> bool {
        while (true) {
          _highs->run();
          auto ms = _highs->getModelStatus();
          if (ms == HighsModelStatus::kOptimal)
            return true;

          if (_lp_callback_interval == 0 ||
              ms == HighsModelStatus::kInfeasible ||
              ms == HighsModelStatus::kUnbounded ||
              ms == HighsModelStatus::kSolveError) {
            return false;
          }

          extract_solution();
          _last_callback_event = LPCallbackEvent::LPProgress;
          if (_callback(*this, _domain)) {
            user_stopped = true;
            return true;
          }
        }
      };

      if (!run_lp_with_callbacks() && !user_stopped) {
        _highs->clearSolver();
        if (!run_lp_with_callbacks() && !user_stopped) {
          init_lp(s);
          if (!run_lp_with_callbacks() && !user_stopped) {
            throw std::runtime_error(
                "IDual LP not optimal: " +
                _highs->modelStatusToString(_highs->getModelStatus()));
          }
        }
      }

      if (_lp_callback_interval > 0) {
        _highs->stopCallback(kCallbackSimplexInterrupt);
      }

      if (user_stopped) {
        Logger::info("IDual interrupted by callback during LP solve");
        break;
      }

      extract_solution();
      _nb_lp_iterations++;

      _last_callback_event = LPCallbackEvent::SolverIteration;
      if (_callback(*this, _domain)) {
        Logger::info("IDual interrupted by callback");
        break;
      }

      if (_verbose) {
        Logger::debug(
            "IDual iteration " + StringConverter::from(_nb_lp_iterations) +
            ": " + StringConverter::from(_graph.size()) + " states explored, " +
            StringConverter::from(_fringe_reachable.size()) +
            " fringe reachable");
      }
    }

    Logger::info("IDual finished in " +
                 StringConverter::from((double)get_solving_time() / 1e3) +
                 " seconds with " + StringConverter::from(_nb_lp_iterations) +
                 " LP iterations and " + StringConverter::from(_graph.size()) +
                 " states explored.");
  } catch (const std::exception &e) {
    Logger::error("IDual failed: " + std::string(e.what()));
    throw;
  }
}

SK_IDUAL_TEMPLATE_DECL
bool SK_IDUAL_CLASS::is_solution_defined_for(const State &s) const {
  auto it = _graph.find(s);
  if (it == _graph.end())
    return false;
  return it->best_action != nullptr || it->goal;
}

SK_IDUAL_TEMPLATE_DECL
typename SK_IDUAL_CLASS::Value
SK_IDUAL_CLASS::get_best_value(const State &s) const {
  auto it = _graph.find(s);
  if (it != _graph.end()) {
    Value val;
    val.cost(it->best_value);
    return val;
  }
  throw std::runtime_error("SKDECIDE exception: state not found in IDual.");
}

SK_IDUAL_TEMPLATE_DECL
template <typename D, std::enable_if_t<!has_get_constraints<D>::value, int>>
const typename SK_IDUAL_CLASS::Action &
SK_IDUAL_CLASS::get_best_action(const State &s) const {
  auto it = _graph.find(s);
  if (it != _graph.end() && it->best_action != nullptr) {
    return it->best_action->action;
  }
  throw std::runtime_error(
      "SKDECIDE exception: no best action found in IDual policy.");
}

SK_IDUAL_TEMPLATE_DECL
template <typename D, std::enable_if_t<has_get_constraints<D>::value, int>>
std::vector<std::pair<typename SK_IDUAL_CLASS::Action, double>>
SK_IDUAL_CLASS::get_action_distribution(const State &s) const {
  auto it = _graph.find(s);
  if (it != _graph.end() && !it->action_probabilities.empty()) {
    std::vector<std::pair<Action, double>> dist;
    for (const auto &[an, prob] : it->action_probabilities) {
      dist.push_back({an->action, prob});
    }
    return dist;
  }
  throw std::runtime_error(
      "SKDECIDE exception: no action distribution found in IDual policy.");
}

SK_IDUAL_TEMPLATE_DECL
std::size_t SK_IDUAL_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_IDUAL_TEMPLATE_DECL
std::size_t SK_IDUAL_CLASS::get_nb_lp_iterations() const {
  return _nb_lp_iterations;
}

SK_IDUAL_TEMPLATE_DECL
std::size_t SK_IDUAL_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

SK_IDUAL_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_IDUAL_CLASS::State>::Set
SK_IDUAL_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored;
  for (const auto &sn : _graph) {
    explored.insert(sn.state);
  }
  return explored;
}

SK_IDUAL_TEMPLATE_DECL
template <typename D, std::enable_if_t<!has_get_constraints<D>::value, int>>
typename MapTypeDeducer<typename SK_IDUAL_CLASS::State,
                        std::pair<typename SK_IDUAL_CLASS::Action,
                                  typename SK_IDUAL_CLASS::Value>>::Map
SK_IDUAL_CLASS::get_policy() const {
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map policy;
  for (const auto &sn : _graph) {
    if (sn.best_action != nullptr) {
      Value value;
      value.cost(sn.best_value);
      policy.insert({sn.state, {sn.best_action->action, value}});
    }
  }
  return policy;
}

SK_IDUAL_TEMPLATE_DECL
template <typename D, std::enable_if_t<has_get_constraints<D>::value, int>>
typename MapTypeDeducer<
    typename SK_IDUAL_CLASS::State,
    std::pair<std::vector<std::pair<typename SK_IDUAL_CLASS::Action, double>>,
              typename SK_IDUAL_CLASS::Value>>::Map
SK_IDUAL_CLASS::get_policy() const {
  typename MapTypeDeducer<
      State, std::pair<std::vector<std::pair<Action, double>>, Value>>::Map
      policy;
  for (const auto &sn : _graph) {
    if (!sn.action_probabilities.empty()) {
      std::vector<std::pair<Action, double>> dist;
      for (const auto &[an, prob] : sn.action_probabilities) {
        dist.push_back({an->action, prob});
      }
      Value value;
      value.cost(sn.best_value);
      policy.insert({sn.state, {dist, value}});
    }
  }
  return policy;
}

SK_IDUAL_TEMPLATE_DECL
template <typename Params>
std::unique_ptr<SK_IDUAL_CLASS> SK_IDUAL_CLASS::create_from_params(
    Domain &domain,
    std::function<Predicate(Domain &, const State &)> goal_checker,
    std::function<Value(Domain &, const State &)> heuristic,
    std::function<Value(const State &)> terminal_value, const Params &params,
    bool verbose) {
  return std::make_unique<IDualSolver>(
      domain, goal_checker, heuristic, terminal_value,
      SecondaryHeuristicFunctor(nullptr), std::vector<double>{},
      params.template get<double>("epsilon", 0.001),
      params.template get<double>("lp_infinity", 1e20),
      params.template get<double>("lp_tolerance", 1e-15),
      params.template get<double>("default_dead_end_cost", 1000.0),
      std::size_t(0),
      CallbackFunctor([](const IDualSolver &, Domain &) { return false; }),
      params.template get<bool>("verbose", verbose));
}

} // namespace skdecide

#endif // SKDECIDE_IDUAL_IMPL_HH
