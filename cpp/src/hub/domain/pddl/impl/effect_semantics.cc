/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <random>

#include "../semantics/task.hh"

#include "../aggregation_effect.hh"
#include "../assignment_effect.hh"
#include "../conditional_effect.hh"
#include "../duration_effect.hh"
#include "../function_expression.hh"
#include "../negation_effect.hh"
#include "../predicate_effect.hh"
#include "../probabilistic_effect.hh"
#include "../quantified_effect.hh"
#include "../timed_effect.hh"
#include "../variable.hh"

// Include template implementations needed for explicit instantiation
#include "aggregation_effect_impl.hh"
#include "assignment_effect_impl.hh"
#include "quantified_effect_impl.hh"

namespace skdecide {

namespace pddl {

// === collect_add_atoms / collect_cost_increase overrides ===

// --- PredicateEffect ---
void PredicateEffect::collect_add_atoms(const Task &task,
                                        const Binding &binding,
                                        const AtomCallback &callback) const {
  int pid = task.predicate_id(get_predicate()->get_name());
  GroundTuple gt;
  gt.reserve(get_terms().size());
  for (auto &term : get_terms()) {
    gt.push_back(task.resolve_term(term, binding));
  }
  callback(pid, gt);
}

// --- ConditionalEffect ---
void ConditionalEffect::collect_add_atoms(const Task &task,
                                          const Binding &binding,
                                          const AtomCallback &callback) const {
  if (BinaryEffect::get_effect()) {
    BinaryEffect::get_effect()->collect_add_atoms(task, binding, callback);
  }
}

void ConditionalEffect::collect_cost_increase(
    const Task &task, const Binding &binding,
    const CostCallback &callback) const {
  if (BinaryEffect::get_effect()) {
    BinaryEffect::get_effect()->collect_cost_increase(task, binding, callback);
  }
}

// --- ProbabilisticEffect ---
bool ProbabilisticEffect::is_probabilistic() const { return true; }

Effect::Ptr ProbabilisticEffect::determinize(const Effect::Ptr &,
                                             DeterminizationMode mode,
                                             std::mt19937 &rng) const {
  if (_outcomes.empty()) {
    return std::make_shared<ConjunctionEffect>();
  }

  double total_prob = 0.0;
  for (auto &[p, e] : _outcomes) {
    total_prob += p;
  }
  double noop_prob = 1.0 - total_prob;

  if (mode == DeterminizationMode::MostProbable) {
    auto it = std::max_element(
        _outcomes.begin(), _outcomes.end(),
        [](const Outcome &a, const Outcome &b) { return a.first < b.first; });
    if (noop_prob > 0.0 && noop_prob >= it->first) {
      return std::make_shared<ConjunctionEffect>();
    }
    return it->second->determinize(it->second, mode, rng);
  }

  // RandomOutcome: include implicit no-op in sampling
  std::vector<double> probs;
  probs.reserve(_outcomes.size() + 1);
  for (auto &[p, e] : _outcomes) {
    probs.push_back(p);
  }
  if (noop_prob > 0.0) {
    probs.push_back(noop_prob);
  }
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  auto idx = dist(rng);
  if (idx >= static_cast<int>(_outcomes.size())) {
    return std::make_shared<ConjunctionEffect>();
  }
  auto &selected = _outcomes[idx];
  return selected.second->determinize(selected.second, mode, rng);
}

void ProbabilisticEffect::collect_add_atoms(
    const Task &task, const Binding &binding,
    const AtomCallback &callback) const {
  for (auto &[prob, effect] : get_outcomes()) {
    effect->collect_add_atoms(task, binding, callback);
  }
}

void ProbabilisticEffect::collect_cost_increase(
    const Task &task, const Binding &binding,
    const CostCallback &callback) const {
  for (auto &[prob, effect] : get_outcomes()) {
    effect->collect_cost_increase(task, binding, callback);
  }
}

// --- IncreaseEffect ---
void IncreaseEffect::collect_cost_increase(const Task &task,
                                           const Binding &binding,
                                           const CostCallback &callback) const {
  auto &func = get_function();
  if (func && func->get_function()) {
    int fid = task.function_id(func->get_function()->get_name());
    if (fid == task.total_cost_function()) {
      callback(get_expression()->evaluate(task.initial_state(), task, binding));
    }
  }
}

// === determinize() implementations for compound effects ===

// --- ConjunctionEffect ---
Effect::Ptr ConjunctionEffect::determinize(const Effect::Ptr &,
                                           DeterminizationMode mode,
                                           std::mt19937 &rng) const {
  auto result = std::make_shared<ConjunctionEffect>();
  for (auto &child : get_effects()) {
    result->append_effect(child->determinize(child, mode, rng));
  }
  return result;
}

// --- DisjunctionEffect ---
Effect::Ptr DisjunctionEffect::determinize(const Effect::Ptr &,
                                           DeterminizationMode mode,
                                           std::mt19937 &rng) const {
  auto result = std::make_shared<DisjunctionEffect>();
  for (auto &child : get_effects()) {
    result->append_effect(child->determinize(child, mode, rng));
  }
  return result;
}

// --- ConditionalEffect ---
Effect::Ptr ConditionalEffect::determinize(const Effect::Ptr &,
                                           DeterminizationMode mode,
                                           std::mt19937 &rng) const {
  auto inner = BinaryEffect::get_effect();
  if (!inner) {
    return std::make_shared<ConditionalEffect>();
  }
  auto det_inner = inner->determinize(inner, mode, rng);
  return std::make_shared<ConditionalEffect>(BinaryEffect::get_condition(),
                                             det_inner);
}

// --- UniversalEffect ---
Effect::Ptr UniversalEffect::determinize(const Effect::Ptr &,
                                         DeterminizationMode mode,
                                         std::mt19937 &rng) const {
  auto inner = QuantifiedEffect<UniversalEffect>::get_effect();
  if (!inner) {
    return std::make_shared<UniversalEffect>();
  }
  auto det_inner = inner->determinize(inner, mode, rng);
  return std::make_shared<UniversalEffect>(
      det_inner,
      static_cast<const VariableContainer<UniversalEffect> &>(*this));
}

// --- ExistentialEffect ---
Effect::Ptr ExistentialEffect::determinize(const Effect::Ptr &,
                                           DeterminizationMode mode,
                                           std::mt19937 &rng) const {
  auto inner = QuantifiedEffect<ExistentialEffect>::get_effect();
  if (!inner) {
    return std::make_shared<ExistentialEffect>();
  }
  auto det_inner = inner->determinize(inner, mode, rng);
  return std::make_shared<ExistentialEffect>(
      det_inner,
      static_cast<const VariableContainer<ExistentialEffect> &>(*this));
}

// === all_determinizations() implementations ===

// --- ProbabilisticEffect ---
std::vector<Effect::Ptr>
ProbabilisticEffect::all_determinizations(const Effect::Ptr &,
                                          std::mt19937 &rng) const {
  std::vector<Effect::Ptr> result;
  for (auto &[prob, sub_effect] : _outcomes) {
    auto inner_dets = sub_effect->all_determinizations(sub_effect, rng);
    for (auto &d : inner_dets) {
      result.push_back(d);
    }
  }
  double total_prob = 0.0;
  for (auto &[p, e] : _outcomes) {
    total_prob += p;
  }
  if (total_prob < 1.0 - 1e-9) {
    result.push_back(std::make_shared<ConjunctionEffect>());
  }
  return result;
}

// --- ConjunctionEffect ---
std::vector<Effect::Ptr>
ConjunctionEffect::all_determinizations(const Effect::Ptr &,
                                        std::mt19937 &rng) const {
  std::vector<std::vector<Effect::Ptr>> child_dets;
  for (auto &child : get_effects()) {
    child_dets.push_back(child->all_determinizations(child, rng));
  }

  // Cartesian product of children's determinizations
  std::vector<std::vector<Effect::Ptr>> combos = {{}};
  for (auto &cd : child_dets) {
    std::vector<std::vector<Effect::Ptr>> new_combos;
    for (auto &combo : combos) {
      for (auto &d : cd) {
        auto extended = combo;
        extended.push_back(d);
        new_combos.push_back(std::move(extended));
      }
    }
    combos = std::move(new_combos);
  }

  std::vector<Effect::Ptr> result;
  result.reserve(combos.size());
  for (auto &combo : combos) {
    auto conj = std::make_shared<ConjunctionEffect>();
    for (auto &eff : combo) {
      conj->append_effect(eff);
    }
    result.push_back(std::move(conj));
  }
  return result;
}

// --- DisjunctionEffect ---
std::vector<Effect::Ptr>
DisjunctionEffect::all_determinizations(const Effect::Ptr &,
                                        std::mt19937 &rng) const {
  std::vector<Effect::Ptr> result;
  for (auto &child : get_effects()) {
    auto inner = child->all_determinizations(child, rng);
    for (auto &d : inner) {
      result.push_back(d);
    }
  }
  return result;
}

// --- ConditionalEffect ---
std::vector<Effect::Ptr>
ConditionalEffect::all_determinizations(const Effect::Ptr &,
                                        std::mt19937 &rng) const {
  auto inner = BinaryEffect::get_effect();
  if (!inner) {
    return {std::make_shared<ConditionalEffect>()};
  }
  auto inner_dets = inner->all_determinizations(inner, rng);
  std::vector<Effect::Ptr> result;
  result.reserve(inner_dets.size());
  for (auto &d : inner_dets) {
    result.push_back(
        std::make_shared<ConditionalEffect>(BinaryEffect::get_condition(), d));
  }
  return result;
}

// --- UniversalEffect ---
std::vector<Effect::Ptr>
UniversalEffect::all_determinizations(const Effect::Ptr &,
                                      std::mt19937 &rng) const {
  auto inner = QuantifiedEffect<UniversalEffect>::get_effect();
  if (!inner) {
    return {std::make_shared<UniversalEffect>()};
  }
  auto inner_dets = inner->all_determinizations(inner, rng);
  std::vector<Effect::Ptr> result;
  result.reserve(inner_dets.size());
  for (auto &d : inner_dets) {
    result.push_back(std::make_shared<UniversalEffect>(
        d, static_cast<const VariableContainer<UniversalEffect> &>(*this)));
  }
  return result;
}

// --- ExistentialEffect ---
std::vector<Effect::Ptr>
ExistentialEffect::all_determinizations(const Effect::Ptr &,
                                        std::mt19937 &rng) const {
  auto inner = QuantifiedEffect<ExistentialEffect>::get_effect();
  if (!inner) {
    return {std::make_shared<ExistentialEffect>()};
  }
  auto inner_dets = inner->all_determinizations(inner, rng);
  std::vector<Effect::Ptr> result;
  result.reserve(inner_dets.size());
  for (auto &d : inner_dets) {
    result.push_back(std::make_shared<ExistentialEffect>(
        d, static_cast<const VariableContainer<ExistentialEffect> &>(*this)));
  }
  return result;
}

// === apply() implementations ===

// --- PredicateEffect ---
Effect::Outcomes PredicateEffect::apply(const State &state, const Task &task,
                                        const Binding &binding) const {
  State s = state.copy();
  int pid = task.predicate_id(get_predicate()->get_name());
  GroundTuple args;
  for (auto &t : get_terms()) {
    args.push_back(task.resolve_term(t, binding));
  }
  s.atoms[pid].insert(std::move(args));
  return {{1.0, std::move(s)}};
}

// --- NegationEffect ---
Effect::Outcomes NegationEffect::apply(const State &state, const Task &task,
                                       const Binding &binding) const {
  State s = state.copy();
  auto &inner = get_effect();
  int pid = task.predicate_id(inner->get_predicate()->get_name());
  GroundTuple args;
  for (auto &t : inner->get_terms()) {
    args.push_back(task.resolve_term(t, binding));
  }
  s.atoms[pid].erase(args);
  return {{1.0, std::move(s)}};
}

// --- ConjunctionEffect ---
Effect::Outcomes ConjunctionEffect::apply(const State &state, const Task &task,
                                          const Binding &binding) const {
  Outcomes current = {{1.0, state.copy()}};
  for (auto &child : get_effects()) {
    Outcomes next;
    for (auto &[p, s] : current) {
      for (auto &[cp, cs] : child->apply(s, task, binding)) {
        next.push_back({p * cp, std::move(cs)});
      }
    }
    current = std::move(next);
  }
  return current;
}

// --- DisjunctionEffect (ONEOF) ---
Effect::Outcomes DisjunctionEffect::apply(const State &state, const Task &task,
                                          const Binding &binding) const {
  auto &effects = get_effects();
  if (effects.empty()) {
    return {{1.0, state.copy()}};
  }
  double weight = 1.0 / static_cast<double>(effects.size());
  Outcomes result;
  for (auto &child : effects) {
    for (auto &[cp, cs] : child->apply(state, task, binding)) {
      result.push_back({weight * cp, std::move(cs)});
    }
  }
  return result;
}

// --- ConditionalEffect ---
Effect::Outcomes ConditionalEffect::apply(const State &state, const Task &task,
                                          const Binding &binding) const {
  if (BinaryEffect::get_condition() &&
      BinaryEffect::get_condition()->holds(state, task, binding)) {
    return BinaryEffect::get_effect()->apply(state, task, binding);
  }
  return {{1.0, state.copy()}};
}

// --- UniversalEffect ---
Effect::Outcomes UniversalEffect::apply(const State &state, const Task &task,
                                        const Binding &binding) const {
  auto &vars = get_variables();
  Outcomes current = {{1.0, state.copy()}};

  // Recursively iterate over all variable bindings
  std::function<void(std::size_t, Binding)> expand = [&](std::size_t vi,
                                                         Binding b) {
    if (vi >= vars.size()) {
      Outcomes next;
      for (auto &[p, s] : current) {
        for (auto &[cp, cs] :
             QuantifiedEffect<UniversalEffect>::get_effect()->apply(s, task,
                                                                    b)) {
          next.push_back({p * cp, std::move(cs)});
        }
      }
      current = std::move(next);
      return;
    }
    auto &types = vars[vi]->get_types();
    std::string type_name =
        types.empty() ? "object" : (*types.begin())->get_name();
    for (int obj_id : task.objects_of_type(type_name)) {
      b[vars[vi]->get_name()] = obj_id;
      expand(vi + 1, b);
    }
  };

  Binding b = binding;
  expand(0, b);
  return current;
}

// --- ExistentialEffect ---
Effect::Outcomes ExistentialEffect::apply(const State &state, const Task &task,
                                          const Binding &binding) const {
  auto &vars = get_variables();

  // Apply for the first matching binding that changes state
  std::function<Outcomes(std::size_t, Binding)> try_expand =
      [&](std::size_t vi, Binding b) -> Outcomes {
    if (vi >= vars.size()) {
      return QuantifiedEffect<ExistentialEffect>::get_effect()->apply(state,
                                                                      task, b);
    }
    auto &types = vars[vi]->get_types();
    std::string type_name =
        types.empty() ? "object" : (*types.begin())->get_name();
    for (int obj_id : task.objects_of_type(type_name)) {
      b[vars[vi]->get_name()] = obj_id;
      auto result = try_expand(vi + 1, b);
      if (!result.empty()) {
        return result;
      }
    }
    return {};
  };

  Binding b = binding;
  auto result = try_expand(0, b);
  if (result.empty()) {
    return {{1.0, state.copy()}};
  }
  return result;
}

// --- ProbabilisticEffect ---
Effect::Outcomes ProbabilisticEffect::apply(const State &state,
                                            const Task &task,
                                            const Binding &binding) const {
  Outcomes result;
  double total_prob = 0.0;

  for (auto &[prob, effect] : get_outcomes()) {
    total_prob += prob;
    for (auto &[cp, cs] : effect->apply(state, task, binding)) {
      result.push_back({prob * cp, std::move(cs)});
    }
  }

  // Remainder probability: unchanged state
  if (total_prob < 1.0 - 1e-9) {
    result.push_back({1.0 - total_prob, state.copy()});
  }

  return result;
}

// --- AssignEffect ---
Effect::Outcomes AssignEffect::apply(const State &state, const Task &task,
                                     const Binding &binding) const {
  State s = state.copy();
  int fid = task.function_id(get_function()->get_function()->get_name());
  GroundTuple args;
  for (auto &t : get_function()->get_terms()) {
    args.push_back(task.resolve_term(t, binding));
  }
  double val = get_expression()->evaluate(state, task, binding);
  s.fluents[fid][std::move(args)] = val;
  return {{1.0, std::move(s)}};
}

// --- IncreaseEffect ---
Effect::Outcomes IncreaseEffect::apply(const State &state, const Task &task,
                                       const Binding &binding) const {
  State s = state.copy();
  int fid = task.function_id(get_function()->get_function()->get_name());
  GroundTuple args;
  for (auto &t : get_function()->get_terms()) {
    args.push_back(task.resolve_term(t, binding));
  }
  double val = get_expression()->evaluate(state, task, binding);
  s.fluents[fid][args] += val;
  return {{1.0, std::move(s)}};
}

// --- DecreaseEffect ---
Effect::Outcomes DecreaseEffect::apply(const State &state, const Task &task,
                                       const Binding &binding) const {
  State s = state.copy();
  int fid = task.function_id(get_function()->get_function()->get_name());
  GroundTuple args;
  for (auto &t : get_function()->get_terms()) {
    args.push_back(task.resolve_term(t, binding));
  }
  double val = get_expression()->evaluate(state, task, binding);
  s.fluents[fid][args] -= val;
  return {{1.0, std::move(s)}};
}

// --- ScaleUpEffect ---
Effect::Outcomes ScaleUpEffect::apply(const State &state, const Task &task,
                                      const Binding &binding) const {
  State s = state.copy();
  int fid = task.function_id(get_function()->get_function()->get_name());
  GroundTuple args;
  for (auto &t : get_function()->get_terms()) {
    args.push_back(task.resolve_term(t, binding));
  }
  double val = get_expression()->evaluate(state, task, binding);
  s.fluents[fid][args] *= val;
  return {{1.0, std::move(s)}};
}

// --- ScaleDownEffect ---
Effect::Outcomes ScaleDownEffect::apply(const State &state, const Task &task,
                                        const Binding &binding) const {
  State s = state.copy();
  int fid = task.function_id(get_function()->get_function()->get_name());
  GroundTuple args;
  for (auto &t : get_function()->get_terms()) {
    args.push_back(task.resolve_term(t, binding));
  }
  double val = get_expression()->evaluate(state, task, binding);
  s.fluents[fid][args] /= val;
  return {{1.0, std::move(s)}};
}

// --- AtStartEffect ---
Effect::Outcomes AtStartEffect::apply(const State &state, const Task &task,
                                      const Binding &binding) const {
  return get_effect()->apply(state, task, binding);
}

// --- AtEndEffect ---
Effect::Outcomes AtEndEffect::apply(const State &state, const Task &task,
                                    const Binding &binding) const {
  return get_effect()->apply(state, task, binding);
}

// --- AtTimeEffect ---
Effect::Outcomes AtTimeEffect::apply(const State &state, const Task &task,
                                     const Binding &binding) const {
  return get_effect()->apply(state, task, binding);
}

// --- DurationEffect (private inheritance — provide minimal impl) ---
Effect::Outcomes DurationEffect::apply(const State &state,
                                       const Task & /*task*/,
                                       const Binding & /*binding*/) const {
  throw std::runtime_error("DurationEffect::apply() not supported");
}

} // namespace pddl

} // namespace skdecide
