/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "temporal_simulator.hh"
#include "goal_checker.hh"
#include "successor_generator.hh"
#include "task.hh"

#include "../aggregation_formula.hh"
#include "../comparison_formula.hh"
#include "../domain.hh"
#include "../duration_expression.hh"
#include "../operator.hh"
#include "../predicate.hh"
#include "../timed_effect.hh"
#include "../timed_formula.hh"
#include "../variable.hh"

#include <algorithm>
#include <clingo.hh>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace skdecide {

namespace pddl {

TemporalSimulator::TemporalSimulator(const Task &task, double epsilon,
                                     double max_event_lookahead,
                                     int max_cascade_iterations,
                                     EventTimeFinderFn event_time_finder)
    : _task(task), _epsilon(epsilon), _max_event_lookahead(max_event_lookahead),
      _max_cascade_iterations(max_cascade_iterations),
      _event_time_finder(std::move(event_time_finder)) {

  _asp_program = _task.generate_asp_program();

  for (int pid = 0; pid < _task.num_predicates(); ++pid) {
    std::string pname = _task.predicate_name(pid);
    std::string safe_pname = pname;
    std::replace(safe_pname.begin(), safe_pname.end(), '-', '_');
    auto &pred = _task.domain()->get_predicate(pname);
    int arity = static_cast<int>(pred->get_variables().size());
    _pred_externals.push_back({safe_pname, arity});
  }

  _ctl = std::make_unique<Clingo::Control>();
  _ctl->add("base", {}, _asp_program.c_str());
  _ctl->ground({{"base", {}}});
}

TemporalSimulator::~TemporalSimulator() = default;

void TemporalSimulator::set_state(const State &state) const {
  for (int pid = 0; pid < static_cast<int>(_pred_externals.size()); ++pid) {
    auto &info = _pred_externals[pid];
    if (pid >= static_cast<int>(state.atoms.size()))
      continue;
    auto &atom_set = state.atoms[pid];

    for (auto &tuple : atom_set) {
      if (info.arity == 0) {
        auto sym = Clingo::Id(info.safe_name.c_str());
        _ctl->assign_external(sym, Clingo::TruthValue::True);
      } else {
        std::vector<Clingo::Symbol> args;
        args.reserve(tuple.size());
        for (int obj_id : tuple) {
          args.push_back(Clingo::Number(obj_id));
        }
        auto sym = Clingo::Function(info.safe_name.c_str(),
                                    {args.data(), args.size()});
        _ctl->assign_external(sym, Clingo::TruthValue::True);
      }
    }
  }
}

void TemporalSimulator::clear_state() const {
  for (auto &sa : _ctl->symbolic_atoms()) {
    if (sa.is_external()) {
      _ctl->assign_external(sa.symbol(), Clingo::TruthValue::False);
    }
  }
}

State TemporalSimulator::apply_action(const State &state,
                                      const GroundAction &action) const {
  auto &act = _task.actions()[action.action_id];
  auto &effect = act->get_effect();
  if (!effect) {
    return state.copy();
  }
  auto outcomes = effect->apply(state, _task, action.binding);
  if (outcomes.empty()) {
    return state.copy();
  }
  return std::move(outcomes[0].second);
}

State TemporalSimulator::integrate_processes(const State &state,
                                             double dt) const {
  auto active_procs = get_active_processes(state);
  if (active_procs.empty()) {
    return state.copy();
  }

  State s = state.copy();
  s.dt = dt;

  for (auto &gp : active_procs) {
    auto &proc = _task.processes()[gp.action_id];
    auto &effect = proc->get_effect();
    if (!effect) {
      continue;
    }
    auto outcomes = effect->apply(s, _task, gp.binding);
    if (!outcomes.empty()) {
      s = std::move(outcomes[0].second);
      s.dt = dt;
    }
  }

  s.dt = 0.0;
  return s;
}

std::pair<State, bool>
TemporalSimulator::fire_events(const State &state) const {
  State s = state.copy();
  bool any_fired = false;

  for (int iter = 0; iter < _max_cascade_iterations; ++iter) {
    auto triggered = get_triggered_events(s);
    if (triggered.empty()) {
      break;
    }

    any_fired = true;
    for (auto &ge : triggered) {
      auto &event = _task.events()[ge.action_id];
      auto &effect = event->get_effect();
      if (!effect) {
        continue;
      }
      auto outcomes = effect->apply(s, _task, ge.binding);
      if (!outcomes.empty()) {
        s = std::move(outcomes[0].second);
      }
    }
  }

  return {std::move(s), any_fired};
}

double TemporalSimulator::evaluate_duration(const State &state, int da_id,
                                            const Binding &binding) const {
  auto &da = _task.durative_actions()[da_id];
  auto &dc = da->get_duration_constraint();
  if (!dc) {
    return 1.0;
  }

  auto try_extract = [&](const Expression::Ptr &lhs,
                         const Expression::Ptr &rhs) -> double {
    if (dynamic_cast<DurationExpression *>(lhs.get())) {
      return rhs->evaluate(state, _task, binding);
    }
    if (dynamic_cast<DurationExpression *>(rhs.get())) {
      return lhs->evaluate(state, _task, binding);
    }
    return -1.0;
  };

  if (auto *eq = dynamic_cast<EqFormula *>(dc.get())) {
    double v =
        try_extract(eq->get_left_expression(), eq->get_right_expression());
    if (v >= 0.0)
      return v;
  }
  if (auto *geq = dynamic_cast<GreaterEqFormula *>(dc.get())) {
    double v =
        try_extract(geq->get_left_expression(), geq->get_right_expression());
    if (v >= 0.0)
      return v;
  }
  if (auto *leq = dynamic_cast<LessEqFormula *>(dc.get())) {
    double v =
        try_extract(leq->get_left_expression(), leq->get_right_expression());
    if (v >= 0.0)
      return v;
  }

  // Try ConjunctionFormula wrapping comparisons
  if (auto *conj = dynamic_cast<ConjunctionFormula *>(dc.get())) {
    for (auto &sub : conj->get_formulas()) {
      if (auto *eq2 = dynamic_cast<EqFormula *>(sub.get())) {
        double v = try_extract(eq2->get_left_expression(),
                               eq2->get_right_expression());
        if (v >= 0.0)
          return v;
      }
      if (auto *geq2 = dynamic_cast<GreaterEqFormula *>(sub.get())) {
        double v = try_extract(geq2->get_left_expression(),
                               geq2->get_right_expression());
        if (v >= 0.0)
          return v;
      }
    }
  }

  return 1.0;
}

State TemporalSimulator::start_durative_action(
    const State &state, const GroundAction &da_action) const {
  auto &da = _task.durative_actions()[da_action.action_id];
  double dur = evaluate_duration(state, da_action.action_id, da_action.binding);

  State s = state.copy();

  // Apply at-start effects
  auto &effect = da->get_effect();
  if (effect) {
    s.duration = dur;
    auto outcomes = effect->apply(s, _task, da_action.binding);
    if (!outcomes.empty()) {
      s = std::move(outcomes[0].second);
    }
    s.duration = 0.0;
  }

  ActiveDurativeAction ada;
  ada.action_id = da_action.action_id;
  ada.binding = da_action.binding;
  ada.start_time = s.time;
  ada.end_time = s.time + dur;
  s.active_durative_actions.push_back(std::move(ada));

  return s;
}

State TemporalSimulator::end_durative_action(const State &state,
                                             int active_index) const {
  if (active_index < 0 ||
      active_index >= static_cast<int>(state.active_durative_actions.size())) {
    return state.copy();
  }

  auto &ada = state.active_durative_actions[active_index];
  auto &da = _task.durative_actions()[ada.action_id];

  State s = state.copy();

  // Apply at-end effects
  auto &effect = da->get_effect();
  if (effect) {
    s.duration = ada.end_time - ada.start_time;
    auto outcomes = effect->apply(s, _task, ada.binding);
    if (!outcomes.empty()) {
      s = std::move(outcomes[0].second);
    }
    s.duration = 0.0;
  }

  s.active_durative_actions.erase(s.active_durative_actions.begin() +
                                  active_index);
  return s;
}

bool TemporalSimulator::check_invariants(const State &state) const {
  for (auto &ada : state.active_durative_actions) {
    auto &da = _task.durative_actions()[ada.action_id];
    auto &precond = da->get_condition();
    if (precond) {
      State s_check = state.copy();
      s_check.duration = ada.end_time - ada.start_time;
      if (!precond->holds(s_check, _task, ada.binding)) {
        return false;
      }
    }
  }
  return true;
}

State TemporalSimulator::time_step(const State &state, double dt,
                                   const GroundAction *action) const {
  State s = state.copy();

  if (action) {
    s = apply_action(s, *action);
  }

  // End durative actions whose end_time <= state.time + dt
  double end_time = s.time + dt;
  bool ended_any = true;
  while (ended_any) {
    ended_any = false;
    for (int i = 0; i < static_cast<int>(s.active_durative_actions.size());
         ++i) {
      if (s.active_durative_actions[i].end_time <= end_time + _epsilon) {
        s = end_durative_action(s, i);
        ended_any = true;
        break;
      }
    }
  }

  s = integrate_processes(s, dt);

  auto [s2, fired] = fire_events(s);
  s = std::move(s2);

  s.time += dt;

  return s;
}

State TemporalSimulator::event_step(const State &state,
                                    const GroundAction *action) const {
  State s = state.copy();

  if (action) {
    s = apply_action(s, *action);
  }

  double dt = get_next_event_time(s);

  // Also consider durative action endpoints
  for (auto &ada : s.active_durative_actions) {
    double remaining = ada.end_time - s.time;
    if (remaining > _epsilon && remaining < dt) {
      dt = remaining;
    }
  }

  if (dt <= _epsilon) {
    dt = _epsilon;
  }

  // End durative actions that expire at or before s.time + dt
  double end_time = s.time + dt;
  bool ended_any = true;
  while (ended_any) {
    ended_any = false;
    for (int i = 0; i < static_cast<int>(s.active_durative_actions.size());
         ++i) {
      if (s.active_durative_actions[i].end_time <= end_time + _epsilon) {
        s = end_durative_action(s, i);
        ended_any = true;
        break;
      }
    }
  }

  s = integrate_processes(s, dt);

  auto [s2, fired] = fire_events(s);
  s = std::move(s2);

  s.time += dt;

  return s;
}

std::vector<GroundAction>
TemporalSimulator::get_applicable_actions(const State &state) const {
  set_state(state);

  std::vector<GroundAction> result;

  auto handle = _ctl->solve();
  for (auto &model : handle) {
    auto symbols = model.symbols(Clingo::ShowType::Shown);
    for (auto &sym : symbols) {
      std::string name = sym.name();
      if (name.substr(0, 11) != "applicable_")
        continue;
      if (name.find("applicable_da_") == 0)
        continue;

      int action_id = std::stoi(name.substr(11));
      auto &act = _task.actions()[action_id];
      auto &params = act->get_variables();

      GroundAction ga;
      ga.action_id = action_id;

      auto args = sym.arguments();
      ga.arguments.reserve(args.size());
      for (std::size_t i = 0; i < args.size(); ++i) {
        int obj_id = args[i].number();
        ga.arguments.push_back(obj_id);
        ga.binding[params[i]->get_name()] = obj_id;
      }

      auto &precond = act->get_condition();
      if (precond && !precond->holds(state, _task, ga.binding)) {
        continue;
      }

      result.push_back(std::move(ga));
    }
  }

  clear_state();
  return result;
}

std::vector<GroundAction>
TemporalSimulator::get_applicable_durative_actions(const State &state) const {
  set_state(state);

  std::vector<GroundAction> result;

  auto handle = _ctl->solve();
  for (auto &model : handle) {
    auto symbols = model.symbols(Clingo::ShowType::Shown);
    for (auto &sym : symbols) {
      std::string name = sym.name();
      if (name.find("applicable_da_") != 0)
        continue;

      int da_id = std::stoi(name.substr(14));
      auto &da = _task.durative_actions()[da_id];
      auto &params = da->get_variables();

      GroundAction ga;
      ga.action_id = da_id;

      auto args = sym.arguments();
      ga.arguments.reserve(args.size());
      for (std::size_t i = 0; i < args.size(); ++i) {
        int obj_id = args[i].number();
        ga.arguments.push_back(obj_id);
        ga.binding[params[i]->get_name()] = obj_id;
      }

      auto &precond = da->get_condition();
      if (precond && !precond->holds(state, _task, ga.binding)) {
        continue;
      }

      result.push_back(std::move(ga));
    }
  }

  clear_state();
  return result;
}

std::vector<GroundAction>
TemporalSimulator::get_active_processes(const State &state) const {
  set_state(state);

  std::vector<GroundAction> result;

  auto handle = _ctl->solve();
  for (auto &model : handle) {
    auto symbols = model.symbols(Clingo::ShowType::Shown);
    for (auto &sym : symbols) {
      std::string name = sym.name();
      if (name.find("active_process_") != 0)
        continue;

      int proc_id = std::stoi(name.substr(15));
      auto &proc = _task.processes()[proc_id];
      auto &params = proc->get_variables();

      GroundAction ga;
      ga.action_id = proc_id;

      auto args = sym.arguments();
      ga.arguments.reserve(args.size());
      for (std::size_t i = 0; i < args.size(); ++i) {
        int obj_id = args[i].number();
        ga.arguments.push_back(obj_id);
        ga.binding[params[i]->get_name()] = obj_id;
      }

      auto &precond = proc->get_condition();
      if (precond && !precond->holds(state, _task, ga.binding)) {
        continue;
      }

      result.push_back(std::move(ga));
    }
  }

  clear_state();
  return result;
}

std::vector<GroundAction>
TemporalSimulator::get_triggered_events(const State &state) const {
  set_state(state);

  std::vector<GroundAction> result;

  auto handle = _ctl->solve();
  for (auto &model : handle) {
    auto symbols = model.symbols(Clingo::ShowType::Shown);
    for (auto &sym : symbols) {
      std::string name = sym.name();
      if (name.find("event_trigger_") != 0)
        continue;

      int event_id = std::stoi(name.substr(14));
      auto &event = _task.events()[event_id];
      auto &params = event->get_variables();

      GroundAction ga;
      ga.action_id = event_id;

      auto args = sym.arguments();
      ga.arguments.reserve(args.size());
      for (std::size_t i = 0; i < args.size(); ++i) {
        int obj_id = args[i].number();
        ga.arguments.push_back(obj_id);
        ga.binding[params[i]->get_name()] = obj_id;
      }

      auto &precond = event->get_condition();
      if (precond && !precond->holds(state, _task, ga.binding)) {
        continue;
      }

      result.push_back(std::move(ga));
    }
  }

  clear_state();
  return result;
}

bool TemporalSimulator::is_goal(const State &state) const {
  auto &goal = _task.goal();
  if (!goal)
    return false;
  Binding empty;
  return goal->holds(state, _task, empty);
}

double TemporalSimulator::get_next_event_time(const State &state) const {
  if (_event_time_finder) {
    return _event_time_finder(state);
  }
  return find_next_event_time_binary(state);
}

double
TemporalSimulator::find_next_event_time_binary(const State &state) const {
  // Check if events already trigger at current state
  auto current_events = get_triggered_events(state);
  if (!current_events.empty()) {
    return _epsilon;
  }

  double dt_lo = 0.0;
  double dt_hi = _max_event_lookahead;

  // Check if any events trigger at max lookahead
  State s_hi = integrate_processes(state, dt_hi);
  s_hi.time = state.time + dt_hi;
  auto events_hi = get_triggered_events(s_hi);
  if (events_hi.empty()) {
    return dt_hi;
  }

  // Binary search for the earliest event time
  while (dt_hi - dt_lo > _epsilon) {
    double dt_mid = (dt_lo + dt_hi) / 2.0;
    State s_mid = integrate_processes(state, dt_mid);
    s_mid.time = state.time + dt_mid;
    auto events_mid = get_triggered_events(s_mid);
    if (events_mid.empty()) {
      dt_lo = dt_mid;
    } else {
      dt_hi = dt_mid;
    }
  }

  return dt_hi;
}

bool TemporalSimulator::has_z3() {
#ifdef SKDECIDE_HAS_Z3
  return true;
#else
  return false;
#endif
}

} // namespace pddl

} // namespace skdecide
