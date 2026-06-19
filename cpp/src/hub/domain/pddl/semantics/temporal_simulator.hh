/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_SEMANTICS_TEMPORAL_SIMULATOR_HH
#define SKDECIDE_PDDL_SEMANTICS_TEMPORAL_SIMULATOR_HH

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "applicable_actions_generator.hh"
#include "state.hh"

namespace Clingo {
class Control;
}

namespace skdecide {

namespace pddl {

class Task;

class TemporalSimulator {
public:
  using EventTimeFinderFn = std::function<double(const State &)>;

  TemporalSimulator(const Task &task, double epsilon = 1e-9,
                    double max_event_lookahead = 1e6,
                    int max_cascade_iterations = 100,
                    EventTimeFinderFn event_time_finder = nullptr);
  ~TemporalSimulator();

  State apply_action(const State &state, const GroundAction &action) const;
  State integrate_processes(const State &state, double dt) const;
  std::pair<State, bool> fire_events(const State &state) const;
  State start_durative_action(const State &state,
                              const GroundAction &da_action) const;
  State end_durative_action(const State &state, int active_index) const;
  bool check_invariants(const State &state) const;

  State time_step(const State &state, double dt,
                  const GroundAction *action = nullptr) const;
  State event_step(const State &state,
                   const GroundAction *action = nullptr) const;

  std::vector<GroundAction> get_applicable_actions(const State &state) const;
  std::vector<GroundAction>
  get_applicable_durative_actions(const State &state) const;
  std::vector<GroundAction> get_active_processes(const State &state) const;
  std::vector<GroundAction> get_triggered_events(const State &state) const;
  bool is_goal(const State &state) const;
  double get_next_event_time(const State &state) const;

  static bool has_z3();

private:
  const Task &_task;
  std::unique_ptr<Clingo::Control> _ctl;
  std::string _asp_program;

  double _epsilon;
  double _max_event_lookahead;
  int _max_cascade_iterations;

  EventTimeFinderFn _event_time_finder;

  struct PredicateExternalInfo {
    std::string safe_name;
    int arity;
  };
  std::vector<PredicateExternalInfo> _pred_externals;

  void set_state(const State &state) const;
  void clear_state() const;

  double evaluate_duration(const State &state, int da_id,
                           const Binding &binding) const;
  double find_next_event_time_binary(const State &state) const;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_SEMANTICS_TEMPORAL_SIMULATOR_HH
