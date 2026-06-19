/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_SEMANTICS_TASK_HH
#define SKDECIDE_PDDL_SEMANTICS_TASK_HH

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "state.hh"

namespace skdecide {

namespace pddl {

class Domain;
class Problem;
class Term;
class Action;
class DurativeAction;
class Event;
class Process;
class Formula;

class Task {
public:
  typedef std::shared_ptr<Domain> DomainPtr;
  typedef std::shared_ptr<Problem> ProblemPtr;

  Task(const DomainPtr &domain, const ProblemPtr &problem);
  Task(const Task &other, std::vector<std::shared_ptr<Action>> custom_actions);

  int num_objects() const;
  int object_id(const std::string &name) const;
  const std::string &object_name(int id) const;

  int num_predicates() const;
  int predicate_id(const std::string &name) const;
  const std::string &predicate_name(int id) const;

  int num_functions() const;
  int function_id(const std::string &name) const;
  const std::string &function_name(int id) const;
  int total_cost_function() const;
  int reward_function() const;

  const std::vector<int> &objects_of_type(const std::string &type_name) const;

  int resolve_term(const std::shared_ptr<Term> &term,
                   const Binding &binding) const;

  const std::vector<std::shared_ptr<Action>> &actions() const;
  const std::string &action_name(int action_id) const;

  int num_events() const;
  const std::vector<std::shared_ptr<Event>> &events() const;
  const std::string &event_name(int event_id) const;

  int num_processes() const;
  const std::vector<std::shared_ptr<Process>> &processes() const;
  const std::string &process_name(int process_id) const;

  int num_durative_actions() const;
  const std::vector<std::shared_ptr<DurativeAction>> &durative_actions() const;
  const std::string &durative_action_name(int da_id) const;

  const State &initial_state() const;

  const std::shared_ptr<Formula> &goal() const;

  std::string generate_asp_program() const;

  const DomainPtr &domain() const { return _domain; }
  const ProblemPtr &problem() const { return _problem; }

private:
  using VarMap = std::unordered_map<std::string, std::string>;
  void generate_formula_asp_body(const std::shared_ptr<Formula> &formula,
                                 const VarMap &var_map,
                                 std::vector<std::string> &body_atoms,
                                 bool &has_numeric) const;
  std::string safe_name(const std::string &name) const;
  void assign_object_ids();
  void assign_predicate_ids();
  void assign_function_ids();
  void build_type_index();
  void build_initial_state();
  void collect_type_objects(const std::string &type_name,
                            std::vector<int> &result) const;

  DomainPtr _domain;
  ProblemPtr _problem;

  std::vector<std::string> _object_names;
  std::unordered_map<std::string, int> _object_ids;

  std::vector<std::string> _predicate_names;
  std::unordered_map<std::string, int> _pred_ids;

  std::vector<std::string> _function_names;
  std::unordered_map<std::string, int> _func_ids;
  int _total_cost_func = -1;
  int _reward_func = -1;

  std::unordered_map<std::string, std::vector<int>> _type_objects;
  std::unordered_map<std::string, std::string> _type_parent;

  std::vector<std::shared_ptr<Action>> _actions;
  std::vector<std::shared_ptr<Event>> _events;
  std::vector<std::shared_ptr<Process>> _processes;
  std::vector<std::shared_ptr<DurativeAction>> _durative_actions;

  State _initial_state;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_SEMANTICS_TASK_HH
