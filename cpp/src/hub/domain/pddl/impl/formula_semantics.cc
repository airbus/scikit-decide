/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "../semantics/task.hh"

#include "../comparison_formula.hh"
#include "../constraint_formula.hh"
#include "../duration_formula.hh"
#include "../equality_formula.hh"
#include "../imply_formula.hh"
#include "../negation_formula.hh"
#include "../predicate_formula.hh"
#include "../preference.hh"
#include "../quantified_formula.hh"
#include "../timed_formula.hh"
#include "../variable.hh"

// Include template implementations needed for explicit instantiation
#include "aggregation_formula_impl.hh"
#include "binary_formula_impl.hh"
#include "comparison_formula_impl.hh"
#include "quantified_formula_impl.hh"

#include <cmath>
#include <stdexcept>

namespace skdecide {

namespace pddl {

// === collect_positive_atoms overrides ===

// --- PredicateFormula ---
void PredicateFormula::collect_positive_atoms(
    const Task &task, const Binding &binding,
    const AtomCallback &callback) const {
  int pid = task.predicate_id(get_predicate()->get_name());
  GroundTuple gt;
  gt.reserve(get_terms().size());
  for (auto &term : get_terms()) {
    gt.push_back(task.resolve_term(term, binding));
  }
  callback(pid, gt);
}

// --- ConjunctionFormula ---
void ConjunctionFormula::collect_positive_atoms(
    const Task &task, const Binding &binding,
    const AtomCallback &callback) const {
  for (auto &sub : get_formulas()) {
    sub->collect_positive_atoms(task, binding, callback);
  }
}

// === holds() implementations ===

// --- PredicateFormula ---
bool PredicateFormula::holds(const State &state, const Task &task,
                             const Binding &binding) const {
  int pid = task.predicate_id(get_predicate()->get_name());
  GroundTuple args;
  for (auto &t : get_terms()) {
    args.push_back(task.resolve_term(t, binding));
  }
  return state.atoms[pid].count(args) > 0;
}

// --- NegationFormula ---
bool NegationFormula::holds(const State &state, const Task &task,
                            const Binding &binding) const {
  return !get_formula()->holds(state, task, binding);
}

// --- ConjunctionFormula ---
bool ConjunctionFormula::holds(const State &state, const Task &task,
                               const Binding &binding) const {
  for (auto &f : get_formulas()) {
    if (!f->holds(state, task, binding)) {
      return false;
    }
  }
  return true;
}

// --- DisjunctionFormula ---
bool DisjunctionFormula::holds(const State &state, const Task &task,
                               const Binding &binding) const {
  for (auto &f : get_formulas()) {
    if (f->holds(state, task, binding)) {
      return true;
    }
  }
  return false;
}

// --- ImplyFormula ---
bool ImplyFormula::holds(const State &state, const Task &task,
                         const Binding &binding) const {
  return !get_left_formula()->holds(state, task, binding) ||
         get_right_formula()->holds(state, task, binding);
}

// --- EqualityFormula ---
bool EqualityFormula::holds(const State &state, const Task &task,
                            const Binding &binding) const {
  auto &terms = get_terms();
  if (terms.size() < 2) {
    return true;
  }
  int first = task.resolve_term(terms[0], binding);
  int second = task.resolve_term(terms[1], binding);
  return first == second;
}

// --- UniversalFormula ---
bool UniversalFormula::holds(const State &state, const Task &task,
                             const Binding &binding) const {
  auto &vars = get_variables();

  std::function<bool(std::size_t, Binding)> check = [&](std::size_t vi,
                                                        Binding b) -> bool {
    if (vi >= vars.size()) {
      return get_formula()->holds(state, task, b);
    }
    auto &types = vars[vi]->get_types();
    std::string type_name =
        types.empty() ? "object" : (*types.begin())->get_name();
    for (int obj_id : task.objects_of_type(type_name)) {
      b[vars[vi]->get_name()] = obj_id;
      if (!check(vi + 1, b)) {
        return false;
      }
    }
    return true;
  };

  Binding b = binding;
  return check(0, b);
}

// --- ExistentialFormula ---
bool ExistentialFormula::holds(const State &state, const Task &task,
                               const Binding &binding) const {
  auto &vars = get_variables();

  std::function<bool(std::size_t, Binding)> check = [&](std::size_t vi,
                                                        Binding b) -> bool {
    if (vi >= vars.size()) {
      return get_formula()->holds(state, task, b);
    }
    auto &types = vars[vi]->get_types();
    std::string type_name =
        types.empty() ? "object" : (*types.begin())->get_name();
    for (int obj_id : task.objects_of_type(type_name)) {
      b[vars[vi]->get_name()] = obj_id;
      if (check(vi + 1, b)) {
        return true;
      }
    }
    return false;
  };

  Binding b = binding;
  return check(0, b);
}

// --- GreaterFormula ---
bool GreaterFormula::holds(const State &state, const Task &task,
                           const Binding &binding) const {
  return get_left_expression()->evaluate(state, task, binding) >
         get_right_expression()->evaluate(state, task, binding);
}

// --- GreaterEqFormula ---
bool GreaterEqFormula::holds(const State &state, const Task &task,
                             const Binding &binding) const {
  return get_left_expression()->evaluate(state, task, binding) >=
         get_right_expression()->evaluate(state, task, binding);
}

// --- LessFormula ---
bool LessFormula::holds(const State &state, const Task &task,
                        const Binding &binding) const {
  return get_left_expression()->evaluate(state, task, binding) <
         get_right_expression()->evaluate(state, task, binding);
}

// --- LessEqFormula ---
bool LessEqFormula::holds(const State &state, const Task &task,
                          const Binding &binding) const {
  return get_left_expression()->evaluate(state, task, binding) <=
         get_right_expression()->evaluate(state, task, binding);
}

// --- EqFormula ---
bool EqFormula::holds(const State &state, const Task &task,
                      const Binding &binding) const {
  double lhs = get_left_expression()->evaluate(state, task, binding);
  double rhs = get_right_expression()->evaluate(state, task, binding);
  return std::abs(lhs - rhs) < 1e-9;
}

// --- AtStartFormula ---
bool AtStartFormula::holds(const State &state, const Task &task,
                           const Binding &binding) const {
  return get_formula()->holds(state, task, binding);
}

// --- AtEndFormula ---
bool AtEndFormula::holds(const State &state, const Task &task,
                         const Binding &binding) const {
  return get_formula()->holds(state, task, binding);
}

// --- OverAllFormula ---
bool OverAllFormula::holds(const State &state, const Task &task,
                           const Binding &binding) const {
  return get_formula()->holds(state, task, binding);
}

// --- AlwaysFormula ---
bool AlwaysFormula::holds(const State &state, const Task &task,
                          const Binding &binding) const {
  return get_formula()->holds(state, task, binding);
}

// --- SometimeFormula ---
bool SometimeFormula::holds(const State &state, const Task &task,
                            const Binding &binding) const {
  return get_formula()->holds(state, task, binding);
}

// --- AtMostOnceFormula ---
bool AtMostOnceFormula::holds(const State &state, const Task &task,
                              const Binding &binding) const {
  return get_formula()->holds(state, task, binding);
}

// --- WithinFormula ---
bool WithinFormula::holds(const State &state, const Task &task,
                          const Binding &binding) const {
  return get_formula()->holds(state, task, binding);
}

// --- HoldAfterFormula ---
bool HoldAfterFormula::holds(const State &state, const Task &task,
                             const Binding &binding) const {
  return get_formula()->holds(state, task, binding);
}

// --- HoldDuringFormula ---
bool HoldDuringFormula::holds(const State &state, const Task &task,
                              const Binding &binding) const {
  return get_formula()->holds(state, task, binding);
}

// --- SometimeAfterFormula ---
bool SometimeAfterFormula::holds(const State &state, const Task &task,
                                 const Binding &binding) const {
  return get_left_formula()->holds(state, task, binding) &&
         get_right_formula()->holds(state, task, binding);
}

// --- SometimeBeforeFormula ---
bool SometimeBeforeFormula::holds(const State &state, const Task &task,
                                  const Binding &binding) const {
  return get_left_formula()->holds(state, task, binding) &&
         get_right_formula()->holds(state, task, binding);
}

// --- AlwaysWithinFormula ---
bool AlwaysWithinFormula::holds(const State &state, const Task &task,
                                const Binding &binding) const {
  return get_left_formula()->holds(state, task, binding) &&
         get_right_formula()->holds(state, task, binding);
}

// --- DurationFormula (private inheritance — provide minimal impl) ---
bool DurationFormula::holds(const State & /*state*/, const Task & /*task*/,
                            const Binding & /*binding*/) const {
  throw std::runtime_error("DurationFormula::holds() not supported");
}

// --- Preference ---
bool Preference::holds(const State &state, const Task &task,
                       const Binding &binding) const {
  if (_formula) {
    return _formula->holds(state, task, binding);
  }
  return true;
}

} // namespace pddl

} // namespace skdecide
