/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "applicable_actions_generator.hh"
#include "task.hh"

#include "../domain.hh"
#include "../operator.hh"
#include "../predicate.hh"
#include "../variable.hh"

#include <clingo.hh>
#include <sstream>
#include <stdexcept>

namespace skdecide {

namespace pddl {

ApplicableActionsGenerator::ApplicableActionsGenerator(const Task &task)
    : _task(task), _has_numeric_preconditions(false) {

  _asp_program = _task.generate_asp_program();

  // Build predicate external info for state toggling
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

ApplicableActionsGenerator::~ApplicableActionsGenerator() = default;

void ApplicableActionsGenerator::set_state(const State &state) const {
  for (int pid = 0; pid < static_cast<int>(_pred_externals.size()); ++pid) {
    auto &info = _pred_externals[pid];
    auto &atom_set = state.atoms[pid];

    // Set all ground atoms for this predicate
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

void ApplicableActionsGenerator::clear_state() const {
  for (int pid = 0; pid < static_cast<int>(_pred_externals.size()); ++pid) {
    auto &info = _pred_externals[pid];
    auto &atom_set = _task.initial_state().atoms.size() > 0
                         ? _task.initial_state().atoms[pid]
                         : _task.initial_state().atoms[pid];

    // We need to clear ALL atoms that were set, not just initial state ones.
    // Use symbolic_atoms to find all external atoms and set them to false.
  }

  // More robust: iterate over symbolic atoms and set all externals to false
  for (auto &sa : _ctl->symbolic_atoms()) {
    if (sa.is_external()) {
      _ctl->assign_external(sa.symbol(), Clingo::TruthValue::False);
    }
  }
}

std::vector<GroundAction>
ApplicableActionsGenerator::get_applicable_actions(const State &state,
                                                   bool check_numeric) const {
  set_state(state);

  std::vector<GroundAction> result;

  auto handle = _ctl->solve();
  for (auto &model : handle) {
    auto symbols = model.symbols(Clingo::ShowType::Shown);
    for (auto &sym : symbols) {
      // Match applicable_N(...) atoms
      std::string name = sym.name();
      if (name.substr(0, 11) != "applicable_")
        continue;

      int action_id = std::stoi(name.substr(11));
      auto &action = _task.actions()[action_id];
      auto &params = action->get_variables();

      GroundAction ga;
      ga.action_id = action_id;

      auto args = sym.arguments();
      ga.arguments.reserve(args.size());
      for (std::size_t i = 0; i < args.size(); ++i) {
        int obj_id = args[i].number();
        ga.arguments.push_back(obj_id);
        ga.binding[params[i]->get_name()] = obj_id;
      }

      // Post-filter: check numeric preconditions via polymorphic holds()
      if (check_numeric) {
        auto &precond = action->get_condition();
        if (precond) {
          if (!precond->holds(state, _task, ga.binding)) {
            continue;
          }
        }
      }

      result.push_back(std::move(ga));
    }
  }

  clear_state();
  return result;
}

} // namespace pddl

} // namespace skdecide
