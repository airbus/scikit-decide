/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/object_container.hh"

namespace skdecide {

namespace pddl {

// === OperatorContainer implementation ===

template <typename Derived, typename Operator>
OperatorContainer<Derived, Operator>::OperatorContainer() {}

template <typename Derived, typename Operator>
OperatorContainer<Derived, Operator>::OperatorContainer(
    const OperatorContainer<Derived, Operator> &other)
    : AssociativeContainer<Derived, Operator>(other) {}

template <typename Derived, typename Operator>
OperatorContainer<Derived, Operator> &
OperatorContainer<Derived, Operator>::operator=(
    const OperatorContainer<Derived, Operator> &other) {
  dynamic_cast<AssociativeContainer<Derived, Operator> &>(*this) = other;
  return *this;
}

template <typename Derived, typename Operator>
OperatorContainer<Derived, Operator>::~OperatorContainer() {}

template <typename Derived, typename Operator>
template <typename T>
const typename OperatorContainer<Derived, Operator>::OperatorPtr &
OperatorContainer<Derived, Operator>::add_operator(const T &op) {
  return AssociativeContainer<Derived, Operator>::add(op);
}

template <typename Derived, typename Operator>
template <typename T>
void OperatorContainer<Derived, Operator>::remove_operator(const T &op) {
  AssociativeContainer<Derived, Operator>::remove(op);
}

template <typename Derived, typename Operator>
template <typename T>
const typename OperatorContainer<Derived, Operator>::OperatorPtr &
OperatorContainer<Derived, Operator>::get_operator(const T &op) const {
  return AssociativeContainer<Derived, Operator>::get(op);
}

template <typename Derived, typename Operator>
const typename OperatorContainer<Derived, Operator>::OperatorSet &
OperatorContainer<Derived, Operator>::get_operators() const {
  return AssociativeContainer<Derived, Operator>::get_container();
}

// === ActionContainer implementation ===

template <typename Derived> ActionContainer<Derived>::ActionContainer() {}

template <typename Derived>
ActionContainer<Derived>::ActionContainer(const ActionContainer &other)
    : OperatorContainer<Derived, Action>(other) {}

template <typename Derived>
ActionContainer<Derived> &
ActionContainer<Derived>::operator=(const ActionContainer<Derived> &other) {
  dynamic_cast<OperatorContainer<Derived, Action> &>(*this) = other;
  return *this;
}

template <typename Derived> ActionContainer<Derived>::~ActionContainer() {}

template <typename Derived>
template <typename T>
const typename ActionContainer<Derived>::ActionPtr &
ActionContainer<Derived>::add_action(const T &op) {
  return OperatorContainer<Derived, Action>::add_operator(op);
}

template <typename Derived>
template <typename T>
void ActionContainer<Derived>::remove_action(const T &op) {
  OperatorContainer<Derived, Action>::remove_operator(op);
}

template <typename Derived>
template <typename T>
const typename ActionContainer<Derived>::ActionPtr &
ActionContainer<Derived>::get_action(const T &op) const {
  return OperatorContainer<Derived, Action>::get_operator(op);
}

template <typename Derived>
const typename ActionContainer<Derived>::ActionSet &
ActionContainer<Derived>::get_actions() const {
  return OperatorContainer<Derived, Action>::get_operators();
}

// === DurativeActionContainer implementation ===

template <typename Derived>
DurativeActionContainer<Derived>::DurativeActionContainer() {}

template <typename Derived>
DurativeActionContainer<Derived>::DurativeActionContainer(
    const DurativeActionContainer<Derived> &other)
    : OperatorContainer<Derived, DurativeAction>(other) {}

template <typename Derived>
DurativeActionContainer<Derived> &DurativeActionContainer<Derived>::operator=(
    const DurativeActionContainer<Derived> &other) {
  dynamic_cast<OperatorContainer<Derived, DurativeAction> &>(*this) = other;
  return *this;
}

template <typename Derived>
DurativeActionContainer<Derived>::~DurativeActionContainer() {}

template <typename Derived>
template <typename T>
const typename DurativeActionContainer<Derived>::DurativeActionPtr &
DurativeActionContainer<Derived>::add_durative_action(const T &op) {
  return OperatorContainer<Derived, DurativeAction>::add_operator(op);
}

template <typename Derived>
template <typename T>
void DurativeActionContainer<Derived>::remove_durative_action(const T &op) {
  OperatorContainer<Derived, DurativeAction>::remove_operator(op);
}

template <typename Derived>
template <typename T>
const typename DurativeActionContainer<Derived>::DurativeActionPtr &
DurativeActionContainer<Derived>::get_durative_action(const T &op) const {
  return OperatorContainer<Derived, DurativeAction>::get_operator(op);
}

template <typename Derived>
const typename DurativeActionContainer<Derived>::DurativeActionSet &
DurativeActionContainer<Derived>::get_durative_actions() const {
  return OperatorContainer<Derived, DurativeAction>::get_operators();
}

// === EventContainer implementation ===

template <typename Derived> EventContainer<Derived>::EventContainer() {}

template <typename Derived>
EventContainer<Derived>::EventContainer(const EventContainer<Derived> &other)
    : OperatorContainer<Derived, Event>(other) {}

template <typename Derived>
EventContainer<Derived> &
EventContainer<Derived>::operator=(const EventContainer<Derived> &other) {
  dynamic_cast<OperatorContainer<Derived, Event> &>(*this) = other;
  return *this;
}

template <typename Derived> EventContainer<Derived>::~EventContainer() {}

template <typename Derived>
template <typename T>
const typename EventContainer<Derived>::EventPtr &
EventContainer<Derived>::add_event(const T &op) {
  return OperatorContainer<Derived, Event>::add_operator(op);
}

template <typename Derived>
template <typename T>
void EventContainer<Derived>::remove_event(const T &op) {
  OperatorContainer<Derived, Event>::remove_operator(op);
}

template <typename Derived>
template <typename T>
const typename EventContainer<Derived>::EventPtr &
EventContainer<Derived>::get_event(const T &op) const {
  return OperatorContainer<Derived, Event>::get_operator(op);
}

template <typename Derived>
const typename EventContainer<Derived>::EventSet &
EventContainer<Derived>::get_events() const {
  return OperatorContainer<Derived, Event>::get_operators();
}

// === ProcessContainer implementation ===

template <typename Derived> ProcessContainer<Derived>::ProcessContainer() {}

template <typename Derived>
ProcessContainer<Derived>::ProcessContainer(
    const ProcessContainer<Derived> &other)
    : OperatorContainer<Derived, Process>(other) {}

template <typename Derived>
ProcessContainer<Derived> &
ProcessContainer<Derived>::operator=(const ProcessContainer<Derived> &other) {
  dynamic_cast<OperatorContainer<Derived, Process> &>(*this) = other;
  return *this;
}

template <typename Derived> ProcessContainer<Derived>::~ProcessContainer() {}

template <typename Derived>
template <typename T>
const typename ProcessContainer<Derived>::ProcessPtr &
ProcessContainer<Derived>::add_process(const T &op) {
  return OperatorContainer<Derived, Process>::add_operator(op);
}

template <typename Derived>
template <typename T>
void ProcessContainer<Derived>::remove_process(const T &op) {
  OperatorContainer<Derived, Process>::remove_operator(op);
}

template <typename Derived>
template <typename T>
const typename ProcessContainer<Derived>::ProcessPtr &
ProcessContainer<Derived>::get_process(const T &op) const {
  return OperatorContainer<Derived, Process>::get_operator(op);
}

template <typename Derived>
const typename ProcessContainer<Derived>::ProcessSet &
ProcessContainer<Derived>::get_processes() const {
  return OperatorContainer<Derived, Process>::get_operators();
}

} // namespace pddl

} // namespace skdecide
