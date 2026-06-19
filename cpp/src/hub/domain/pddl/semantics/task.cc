/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "task.hh"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "../domain.hh"
#include "../function.hh"
#include "../object.hh"
#include "../operator.hh"
#include "../predicate.hh"
#include "../predicate_effect.hh"
#include "../problem.hh"
#include "../type.hh"
#include "../variable.hh"

#include "../aggregation_formula.hh"
#include "../assignment_effect.hh"
#include "../equality_formula.hh"
#include "../function_expression.hh"
#include "../negation_effect.hh"
#include "../negation_formula.hh"
#include "../numerical_expression.hh"
#include "../predicate_formula.hh"

#include "../impl/aggregation_formula_impl.hh"

namespace skdecide {

namespace pddl {

Task::Task(const DomainPtr &domain, const ProblemPtr &problem)
    : _domain(domain), _problem(problem) {
  assign_object_ids();
  assign_predicate_ids();
  assign_function_ids();
  build_type_index();

  // Collect actions into ordered vector
  for (auto &a : _domain->get_actions()) {
    _actions.push_back(a);
  }
  for (auto &e : _domain->get_events()) {
    _events.push_back(e);
  }
  for (auto &p : _domain->get_processes()) {
    _processes.push_back(p);
  }
  for (auto &da : _domain->get_durative_actions()) {
    _durative_actions.push_back(da);
  }

  build_initial_state();
}

Task::Task(const Task &other,
           std::vector<std::shared_ptr<Action>> custom_actions)
    : _domain(other._domain), _problem(other._problem),
      _object_names(other._object_names), _object_ids(other._object_ids),
      _predicate_names(other._predicate_names), _pred_ids(other._pred_ids),
      _function_names(other._function_names), _func_ids(other._func_ids),
      _total_cost_func(other._total_cost_func),
      _reward_func(other._reward_func), _type_objects(other._type_objects),
      _type_parent(other._type_parent), _actions(std::move(custom_actions)),
      _events(other._events), _processes(other._processes),
      _durative_actions(other._durative_actions),
      _initial_state(other._initial_state) {}

int Task::num_objects() const { return static_cast<int>(_object_names.size()); }

int Task::object_id(const std::string &name) const {
  auto it = _object_ids.find(name);
  if (it == _object_ids.end()) {
    throw std::runtime_error("Unknown object: " + name);
  }
  return it->second;
}

const std::string &Task::object_name(int id) const {
  return _object_names.at(id);
}

int Task::num_predicates() const {
  return static_cast<int>(_predicate_names.size());
}

int Task::predicate_id(const std::string &name) const {
  auto it = _pred_ids.find(name);
  if (it == _pred_ids.end()) {
    throw std::runtime_error("Unknown predicate: " + name);
  }
  return it->second;
}

const std::string &Task::predicate_name(int id) const {
  return _predicate_names.at(id);
}

int Task::num_functions() const {
  return static_cast<int>(_function_names.size());
}

int Task::function_id(const std::string &name) const {
  auto it = _func_ids.find(name);
  if (it == _func_ids.end()) {
    throw std::runtime_error("Unknown function: " + name);
  }
  return it->second;
}

const std::string &Task::function_name(int id) const {
  return _function_names.at(id);
}

int Task::total_cost_function() const { return _total_cost_func; }

int Task::reward_function() const { return _reward_func; }

const std::vector<int> &
Task::objects_of_type(const std::string &type_name) const {
  auto it = _type_objects.find(type_name);
  if (it == _type_objects.end()) {
    static const std::vector<int> empty;
    return empty;
  }
  return it->second;
}

int Task::resolve_term(const std::shared_ptr<Term> &term,
                       const Binding &binding) const {
  const std::string &name = term->get_name();
  if (!name.empty() && name[0] == '?') {
    auto it = binding.find(name);
    if (it == binding.end()) {
      throw std::runtime_error("Unbound variable: " + name);
    }
    return it->second;
  }
  return object_id(name);
}

const std::vector<std::shared_ptr<Action>> &Task::actions() const {
  return _actions;
}

const std::string &Task::action_name(int action_id) const {
  return _actions.at(action_id)->get_name();
}

int Task::num_events() const { return static_cast<int>(_events.size()); }

const std::vector<std::shared_ptr<Event>> &Task::events() const {
  return _events;
}

const std::string &Task::event_name(int event_id) const {
  return _events.at(event_id)->get_name();
}

int Task::num_processes() const { return static_cast<int>(_processes.size()); }

const std::vector<std::shared_ptr<Process>> &Task::processes() const {
  return _processes;
}

const std::string &Task::process_name(int process_id) const {
  return _processes.at(process_id)->get_name();
}

int Task::num_durative_actions() const {
  return static_cast<int>(_durative_actions.size());
}

const std::vector<std::shared_ptr<DurativeAction>> &
Task::durative_actions() const {
  return _durative_actions;
}

const std::string &Task::durative_action_name(int da_id) const {
  return _durative_actions.at(da_id)->get_name();
}

const State &Task::initial_state() const { return _initial_state; }

const std::shared_ptr<Formula> &Task::goal() const {
  return _problem->get_goal();
}

void Task::assign_object_ids() {
  // Domain constants
  for (auto &obj : _domain->get_objects()) {
    int id = static_cast<int>(_object_names.size());
    _object_names.push_back(obj->get_name());
    _object_ids[obj->get_name()] = id;
  }
  // Problem objects
  for (auto &obj : _problem->get_objects()) {
    if (_object_ids.count(obj->get_name()) == 0) {
      int id = static_cast<int>(_object_names.size());
      _object_names.push_back(obj->get_name());
      _object_ids[obj->get_name()] = id;
    }
  }
}

void Task::assign_predicate_ids() {
  for (auto &pred : _domain->get_predicates()) {
    int id = static_cast<int>(_predicate_names.size());
    _predicate_names.push_back(pred->get_name());
    _pred_ids[pred->get_name()] = id;
  }
}

void Task::assign_function_ids() {
  for (auto &func : _domain->get_functions()) {
    int id = static_cast<int>(_function_names.size());
    _function_names.push_back(func->get_name());
    _func_ids[func->get_name()] = id;
    if (func->get_name() == "total-cost") {
      _total_cost_func = id;
    } else if (func->get_name() == "reward") {
      _reward_func = id;
    }
  }
}

void Task::collect_type_objects(const std::string &type_name,
                                std::vector<int> &result) const {
  // Add direct objects
  auto it = _type_objects.find(type_name);
  if (it != _type_objects.end()) {
    for (int id : it->second) {
      result.push_back(id);
    }
  }
}

void Task::build_type_index() {
  // Build type parent map from domain types
  for (auto &type : _domain->get_types()) {
    auto &parents = type->get_types();
    if (!parents.empty()) {
      _type_parent[type->get_name()] = (*parents.begin())->get_name();
    }
  }

  // Assign objects to their direct types
  auto assign_types = [this](const auto &objects) {
    for (auto &obj : objects) {
      auto &types = obj->get_types();
      if (types.empty()) {
        _type_objects["object"].push_back(_object_ids[obj->get_name()]);
      } else {
        for (auto &t : types) {
          _type_objects[t->get_name()].push_back(_object_ids[obj->get_name()]);
        }
      }
    }
  };
  assign_types(_domain->get_objects());
  assign_types(_problem->get_objects());

  // Propagate objects up the type hierarchy
  // For each type, add its objects to all ancestor types
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &[child, parent] : _type_parent) {
      auto &child_objs = _type_objects[child];
      auto &parent_objs = _type_objects[parent];
      for (int id : child_objs) {
        if (std::find(parent_objs.begin(), parent_objs.end(), id) ==
            parent_objs.end()) {
          parent_objs.push_back(id);
          changed = true;
        }
      }
    }
  }
}

void Task::build_initial_state() {
  _initial_state.atoms.resize(_predicate_names.size());
  _initial_state.fluents.resize(_function_names.size());

  auto &init = _problem->get_initial_effect();
  if (!init)
    return;

  Binding empty_binding;

  for (auto &eff : init->get_effects()) {
    // Positive predicate effects
    if (auto *pe = dynamic_cast<PredicateEffect *>(eff.get())) {
      int pid = predicate_id(pe->get_predicate()->get_name());
      GroundTuple args;
      for (auto &t : pe->get_terms()) {
        args.push_back(resolve_term(t, empty_binding));
      }
      _initial_state.atoms[pid].insert(std::move(args));
    }
    // Assignment effects (function initialization)
    else if (auto *ae = dynamic_cast<AssignEffect *>(eff.get())) {
      int fid = function_id(ae->get_function()->get_function()->get_name());
      GroundTuple args;
      for (auto &t : ae->get_function()->get_terms()) {
        args.push_back(resolve_term(t, empty_binding));
      }
      if (auto *ne =
              dynamic_cast<NumericalExpression *>(ae->get_expression().get())) {
        _initial_state.fluents[fid][std::move(args)] =
            ne->get_number()->as_double();
      }
    }
  }
}

std::string Task::safe_name(const std::string &name) const {
  std::string result = name;
  std::replace(result.begin(), result.end(), '-', '_');
  return result;
}

void Task::generate_formula_asp_body(const std::shared_ptr<Formula> &formula,
                                     const VarMap &var_map,
                                     std::vector<std::string> &body_atoms,
                                     bool &has_numeric) const {

  if (auto *pf = dynamic_cast<PredicateFormula *>(formula.get())) {
    std::ostringstream atom;
    atom << safe_name(pf->get_predicate()->get_name());
    auto &terms = pf->get_terms();
    if (!terms.empty()) {
      atom << "(";
      for (std::size_t i = 0; i < terms.size(); ++i) {
        if (i > 0)
          atom << ",";
        const std::string &tname = terms[i]->get_name();
        if (!tname.empty() && tname[0] == '?') {
          auto it = var_map.find(tname);
          if (it != var_map.end()) {
            atom << it->second;
          } else {
            atom << safe_name(tname.substr(1));
          }
        } else {
          atom << _object_ids.at(tname);
        }
      }
      atom << ")";
    }
    body_atoms.push_back(atom.str());
  } else if (auto *nf = dynamic_cast<NegationFormula *>(formula.get())) {
    std::vector<std::string> inner_atoms;
    generate_formula_asp_body(nf->get_formula(), var_map, inner_atoms,
                              has_numeric);
    for (auto &a : inner_atoms) {
      body_atoms.push_back("not " + a);
    }
  } else if (auto *cf = dynamic_cast<ConjunctionFormula *>(formula.get())) {
    for (auto &sub : cf->get_formulas()) {
      generate_formula_asp_body(sub, var_map, body_atoms, has_numeric);
    }
  } else if (auto *ef = dynamic_cast<EqualityFormula *>(formula.get())) {
    auto &terms = ef->get_terms();
    if (terms.size() >= 2) {
      auto resolve_asp_term =
          [&](const std::shared_ptr<Term> &t) -> std::string {
        const std::string &n = t->get_name();
        if (!n.empty() && n[0] == '?') {
          auto it = var_map.find(n);
          if (it != var_map.end())
            return it->second;
          return safe_name(n.substr(1));
        }
        return std::to_string(_object_ids.at(n));
      };
      body_atoms.push_back(resolve_asp_term(terms[0]) + " = " +
                           resolve_asp_term(terms[1]));
    }
  } else {
    has_numeric = true;
  }
}

std::string Task::generate_asp_program() const {
  std::ostringstream asp;

  // Type hierarchy facts
  for (auto &[type_name, obj_ids] : _type_objects) {
    for (int id : obj_ids) {
      asp << "type_" << safe_name(type_name) << "(" << id << ").\n";
    }
  }
  asp << "\n";

  // External state predicates
  for (int pid = 0; pid < num_predicates(); ++pid) {
    const std::string &pname = _predicate_names[pid];
    std::string spname = safe_name(pname);
    auto &pred = _domain->get_predicate(pname);
    auto &vars = pred->get_variables();

    if (vars.empty()) {
      asp << "#external " << spname << ".\n";
    } else {
      asp << "#external " << spname << "(";
      for (std::size_t i = 0; i < vars.size(); ++i) {
        if (i > 0)
          asp << ",";
        asp << "V" << i;
      }
      asp << ") : ";
      for (std::size_t i = 0; i < vars.size(); ++i) {
        if (i > 0)
          asp << ", ";
        auto &types = vars[i]->get_types();
        std::string tname =
            types.empty() ? "object" : (*types.begin())->get_name();
        asp << "type_" << safe_name(tname) << "(V" << i << ")";
      }
      asp << ".\n";
    }
  }
  asp << "\n";

  // Action applicability rules
  for (std::size_t aid = 0; aid < _actions.size(); ++aid) {
    auto &action = _actions[aid];
    auto &params = action->get_variables();

    // Build variable name → ASP variable name mapping
    VarMap var_map;
    for (std::size_t i = 0; i < params.size(); ++i) {
      var_map[params[i]->get_name()] = "V" + std::to_string(i);
    }

    asp << "% Action " << aid << ": " << action->get_name() << "\n";
    asp << "applicable_" << aid;
    if (!params.empty()) {
      asp << "(";
      for (std::size_t i = 0; i < params.size(); ++i) {
        if (i > 0)
          asp << ",";
        asp << "V" << i;
      }
      asp << ")";
    }
    asp << " :- ";

    // Type constraints for parameters
    std::vector<std::string> body_atoms;
    for (std::size_t i = 0; i < params.size(); ++i) {
      auto &types = params[i]->get_types();
      std::string tname =
          types.empty() ? "object" : (*types.begin())->get_name();
      body_atoms.push_back("type_" + safe_name(tname) + "(V" +
                           std::to_string(i) + ")");
    }

    // Precondition atoms (logical predicates encoded in ASP)
    auto &precond = action->get_condition();
    bool has_numeric = false;
    if (precond) {
      generate_formula_asp_body(precond, var_map, body_atoms, has_numeric);
    }

    for (std::size_t i = 0; i < body_atoms.size(); ++i) {
      if (i > 0)
        asp << ", ";
      asp << body_atoms[i];
    }

    if (body_atoms.empty()) {
      asp << "#true";
    }

    asp << ".\n";
  }
  asp << "\n";

  // Process activation rules
  for (std::size_t pid = 0; pid < _processes.size(); ++pid) {
    auto &process = _processes[pid];
    auto &params = process->get_variables();

    VarMap var_map;
    for (std::size_t i = 0; i < params.size(); ++i) {
      var_map[params[i]->get_name()] = "V" + std::to_string(i);
    }

    asp << "% Process " << pid << ": " << process->get_name() << "\n";
    asp << "active_process_" << pid;
    if (!params.empty()) {
      asp << "(";
      for (std::size_t i = 0; i < params.size(); ++i) {
        if (i > 0)
          asp << ",";
        asp << "V" << i;
      }
      asp << ")";
    }
    asp << " :- ";

    std::vector<std::string> body_atoms;
    for (std::size_t i = 0; i < params.size(); ++i) {
      auto &types = params[i]->get_types();
      std::string tname =
          types.empty() ? "object" : (*types.begin())->get_name();
      body_atoms.push_back("type_" + safe_name(tname) + "(V" +
                           std::to_string(i) + ")");
    }

    auto &precond = process->get_condition();
    bool has_numeric = false;
    if (precond) {
      generate_formula_asp_body(precond, var_map, body_atoms, has_numeric);
    }

    for (std::size_t i = 0; i < body_atoms.size(); ++i) {
      if (i > 0)
        asp << ", ";
      asp << body_atoms[i];
    }
    if (body_atoms.empty()) {
      asp << "#true";
    }
    asp << ".\n";
  }
  asp << "\n";

  // Event trigger rules
  for (std::size_t eid = 0; eid < _events.size(); ++eid) {
    auto &event = _events[eid];
    auto &params = event->get_variables();

    VarMap var_map;
    for (std::size_t i = 0; i < params.size(); ++i) {
      var_map[params[i]->get_name()] = "V" + std::to_string(i);
    }

    asp << "% Event " << eid << ": " << event->get_name() << "\n";
    asp << "event_trigger_" << eid;
    if (!params.empty()) {
      asp << "(";
      for (std::size_t i = 0; i < params.size(); ++i) {
        if (i > 0)
          asp << ",";
        asp << "V" << i;
      }
      asp << ")";
    }
    asp << " :- ";

    std::vector<std::string> body_atoms;
    for (std::size_t i = 0; i < params.size(); ++i) {
      auto &types = params[i]->get_types();
      std::string tname =
          types.empty() ? "object" : (*types.begin())->get_name();
      body_atoms.push_back("type_" + safe_name(tname) + "(V" +
                           std::to_string(i) + ")");
    }

    auto &precond = event->get_condition();
    bool has_numeric = false;
    if (precond) {
      generate_formula_asp_body(precond, var_map, body_atoms, has_numeric);
    }

    for (std::size_t i = 0; i < body_atoms.size(); ++i) {
      if (i > 0)
        asp << ", ";
      asp << body_atoms[i];
    }
    if (body_atoms.empty()) {
      asp << "#true";
    }
    asp << ".\n";
  }
  asp << "\n";

  // Durative action applicability rules (at-start preconditions)
  for (std::size_t daid = 0; daid < _durative_actions.size(); ++daid) {
    auto &da = _durative_actions[daid];
    auto &params = da->get_variables();

    VarMap var_map;
    for (std::size_t i = 0; i < params.size(); ++i) {
      var_map[params[i]->get_name()] = "V" + std::to_string(i);
    }

    asp << "% DurativeAction " << daid << ": " << da->get_name() << "\n";
    asp << "applicable_da_" << daid;
    if (!params.empty()) {
      asp << "(";
      for (std::size_t i = 0; i < params.size(); ++i) {
        if (i > 0)
          asp << ",";
        asp << "V" << i;
      }
      asp << ")";
    }
    asp << " :- ";

    std::vector<std::string> body_atoms;
    for (std::size_t i = 0; i < params.size(); ++i) {
      auto &types = params[i]->get_types();
      std::string tname =
          types.empty() ? "object" : (*types.begin())->get_name();
      body_atoms.push_back("type_" + safe_name(tname) + "(V" +
                           std::to_string(i) + ")");
    }

    auto &precond = da->get_condition();
    bool has_numeric = false;
    if (precond) {
      generate_formula_asp_body(precond, var_map, body_atoms, has_numeric);
    }

    for (std::size_t i = 0; i < body_atoms.size(); ++i) {
      if (i > 0)
        asp << ", ";
      asp << body_atoms[i];
    }
    if (body_atoms.empty()) {
      asp << "#true";
    }
    asp << ".\n";
  }
  asp << "\n";

  return asp.str();
}

} // namespace pddl

} // namespace skdecide
