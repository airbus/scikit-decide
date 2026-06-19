/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/operator.hh"

namespace skdecide {

namespace pddl {

// === Action implementation ===

Action::Action(const std::string &name) : Operator<Action>(name) {}

Action::Action(const std::string &name, const Formula::Ptr &precondition,
               const Effect::Ptr &effect)
    : Operator<Action>(name, precondition, effect) {}

Action::Action(const Action &other) : Operator<Action>(other) {}

Action &Action::operator=(const Action &other) {
  dynamic_cast<Operator<Action> &>(*this) = other;
  return *this;
}

Action::~Action() {}

std::ostream &operator<<(std::ostream &o, const Action &a) {
  return a.print(o);
}

// === DurativeAction implementation ===

DurativeAction::DurativeAction(const std::string &name)
    : Operator<DurativeAction>(name) {}

DurativeAction::DurativeAction(const std::string &name,
                               const Formula::Ptr &duration_constraint,
                               const Formula::Ptr &precondition,
                               const Effect::Ptr &effect)
    : Operator<DurativeAction>(name, precondition, effect),
      _duration_constraint(duration_constraint) {}

DurativeAction::DurativeAction(const DurativeAction &other)
    : Operator<DurativeAction>(other),
      _duration_constraint(other._duration_constraint) {}

DurativeAction &DurativeAction::operator=(const DurativeAction &other) {
  dynamic_cast<Operator<DurativeAction> &>(*this) = other;
  this->_duration_constraint = other._duration_constraint;
  return *this;
}

DurativeAction::~DurativeAction() {}

void DurativeAction::set_duration_constraint(
    const Formula::Ptr &duration_constraint) {
  _duration_constraint = duration_constraint;
}

const Formula::Ptr &DurativeAction::get_duration_constraint() const {
  return _duration_constraint;
}

std::ostream &DurativeAction::print(std::ostream &o) const {
  o << "(:durative-action " << this->get_name() << std::endl;
  o << ":parameters (";
  for (const auto &v : get_variables()) {
    o << " " << *v;
  }
  o << " )" << std::endl;
  o << ":duration " << *(this->get_duration_constraint()) << std::endl;
  o << ":precondition " << *(this->get_condition()) << std::endl;
  o << ":effect " << *(this->get_effect()) << std::endl;
  o << ")";
  return o;
}

std::ostream &operator<<(std::ostream &o, const DurativeAction &da) {
  return da.print(o);
}

// === Event implementation ===

Event::Event(const std::string &name) : Operator<Event>(name) {}

Event::Event(const std::string &name, const Formula::Ptr &precondition,
             const Effect::Ptr &effect)
    : Operator<Event>(name, precondition, effect) {}

Event::Event(const Event &other) : Operator<Event>(other) {}

Event &Event::operator=(const Event &other) {
  dynamic_cast<Operator<Event> &>(*this) = other;
  return *this;
}

Event::~Event() {}

std::ostream &operator<<(std::ostream &o, const Event &e) { return e.print(o); }

// === Process implementation ===

Process::Process(const std::string &name) : Operator<Process>(name) {}

Process::Process(const std::string &name, const Formula::Ptr &precondition,
                 const Effect::Ptr &effect)
    : Operator<Process>(name, precondition, effect) {}

Process::Process(const Process &other) : Operator<Process>(other) {}

Process &Process::operator=(const Process &other) {
  dynamic_cast<Operator<Process> &>(*this) = other;
  return *this;
}

Process::~Process() {}

std::ostream &operator<<(std::ostream &o, const Process &p) {
  return p.print(o);
}

} // namespace pddl

} // namespace skdecide
