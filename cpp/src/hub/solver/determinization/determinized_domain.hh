/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_DETERMINIZED_DOMAIN_HH
#define SKDECIDE_DETERMINIZED_DOMAIN_HH

#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace skdecide {

/** @brief Strategy tag: enumerate all outcomes of each action as separate
 *  deterministic actions (one DeterminizedAction per outcome). */
struct AllOutcomesStrategy {};

/** @brief Strategy tag: pick the most probable outcome for each action. */
struct MostProbableOutcomeStrategy {};

/** @brief Strategy tag: pick a random outcome for each action (sampled
 *  uniformly from the transition distribution). */
struct RandomOutcomeStrategy {};

/**
 * @brief A determinized action pairing an original stochastic action with a
 * specific outcome index. Used with AllOutcomesStrategy to create one
 * deterministic action per stochastic outcome.
 *
 * @tparam ToriginalAction The original action type from the stochastic domain.
 */
template <typename ToriginalAction> struct DeterminizedAction {
  ToriginalAction original_action;
  std::size_t outcome_index;

  struct Hash {
    std::size_t operator()(const DeterminizedAction &a) const {
      auto h = typename ToriginalAction::Hash()(a.original_action);
      h ^= std::hash<std::size_t>()(a.outcome_index) + 0x9e3779b9 + (h << 6) +
           (h >> 2);
      return h;
    }
  };

  struct Equal {
    bool operator()(const DeterminizedAction &a,
                    const DeterminizedAction &b) const {
      return typename ToriginalAction::Equal()(a.original_action,
                                               b.original_action) &&
             a.outcome_index == b.outcome_index;
    }
  };

  std::string print() const {
    return original_action.print() + "[" + std::to_string(outcome_index) + "]";
  }
};

/**
 * @brief Wraps a stochastic domain to present a deterministic interface.
 *
 * Depending on the strategy tag, transitions are determinized differently:
 * - AllOutcomesStrategy: each stochastic outcome becomes a separate
 *   DeterminizedAction; get_next_state returns the corresponding outcome.
 * - MostProbableOutcomeStrategy: the original action is kept; get_next_state
 *   returns the most probable successor.
 * - RandomOutcomeStrategy: the original action is kept; get_next_state
 *   samples a random successor from the transition distribution.
 *
 * @tparam Tdomain The original stochastic domain type.
 * @tparam TstrategyTag Determinization strategy (AllOutcomesStrategy,
 *   MostProbableOutcomeStrategy, or RandomOutcomeStrategy).
 * @tparam Texec Execution policy type.
 */
template <typename Tdomain, typename TstrategyTag, typename Texec>
class DeterminizedDomain {
public:
  using OriginalDomain = Tdomain;
  using OrigState = typename Tdomain::State;
  using OrigAction = typename Tdomain::Action;
  using Value = typename Tdomain::Value;
  using Predicate = typename Tdomain::Predicate;

  using State = OrigState;

  using Action =
      std::conditional_t<std::is_same_v<TstrategyTag, AllOutcomesStrategy>,
                         DeterminizedAction<OrigAction>, OrigAction>;

  struct ActionSpace {
    std::vector<Action> _actions;
    const std::vector<Action> &get_elements() const { return _actions; }
  };

  /**
   * @brief Construct a determinized domain wrapping a stochastic domain.
   *
   * @param domain The original stochastic domain to determinize.
   */
  DeterminizedDomain(Tdomain &domain);

  ActionSpace get_applicable_actions(const State &s);
  State get_next_state(const State &s, const Action &a);
  Value get_transition_value(const State &s, const Action &a, const State &ns);

private:
  Tdomain &_domain;
  mutable std::mt19937 _rng;
  mutable Texec _execution_policy;
};

/**
 * @brief Adapter for transition-level determinization of a stochastic domain.
 *
 * Owns a DeterminizedDomain and provides helper methods to convert between
 * determinized and original actions, query expected successors, and signal
 * whether the adapter state must be refreshed between replanning episodes.
 *
 * @tparam Tdomain The original stochastic domain type.
 * @tparam TstrategyTag Determinization strategy (AllOutcomesStrategy,
 *   MostProbableOutcomeStrategy, or RandomOutcomeStrategy).
 * @tparam Texec Execution policy type.
 */
template <typename Tdomain, typename TstrategyTag, typename Texec>
class TransitionDeterminizationAdapter {
public:
  using DeterminizedDomainType =
      DeterminizedDomain<Tdomain, TstrategyTag, Texec>;
  using State = typename Tdomain::State;
  using Action = typename Tdomain::Action;
  using DetAction = typename DeterminizedDomainType::Action;

  /**
   * @brief Construct a transition determinization adapter.
   *
   * @param domain The original stochastic domain to adapt.
   */
  TransitionDeterminizationAdapter(Tdomain &domain);

  void update();
  DeterminizedDomainType &domain();
  Action to_original(const DetAction &a) const;
  State expected_next(const State &s, const DetAction &a);
  bool needs_update_each_replan() const;

private:
  Tdomain &_domain;
  std::unique_ptr<DeterminizedDomainType> _det_domain;
};

} // namespace skdecide

#endif // SKDECIDE_DETERMINIZED_DOMAIN_HH
