/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_OPERATOR_HH
#define SKDECIDE_PDDL_OPERATOR_HH

#include "identifier.hh"
#include "variable_container.hh"
#include "binary_effect.hh"

namespace skdecide {

namespace pddl {

template <typename Derived>
class Operator : public Identifier,
                 public VariableContainer<Derived>,
                 public BinaryEffect { // BinaryEffect brings in precondition
                                       // and effect logics
public:
  typedef std::shared_ptr<Operator<Derived>> Ptr;
  typedef typename VariableContainer<Derived>::VariablePtr VariablePtr;
  typedef typename VariableContainer<Derived>::VariableVector VariableVector;

  Operator(const std::string &name);
  Operator(const std::string &name, const Formula::Ptr &precondition,
           const Effect::Ptr &effect);
  Operator(const Operator<Derived> &other);
  Operator<Derived> &operator=(const Operator<Derived> &other);
  virtual ~Operator();

  virtual std::ostream &print(std::ostream &o) const;
};

class Action : public Operator<Action> {
public:
  static constexpr char class_name[] = "action";

  typedef std::shared_ptr<Action> Ptr;
  typedef VariableContainer<Action>::VariablePtr VariablePtr;
  typedef VariableContainer<Action>::VariableVector VariableVector;

  Action(const std::string &name);
  Action(const std::string &name, const Formula::Ptr &precondition,
         const Effect::Ptr &effect);
  Action(const Action &other);
  Action &operator=(const Action &other);
  virtual ~Action();
};

// Action printing operator
std::ostream &operator<<(std::ostream &o, const Action &a);

class DurativeAction : public Operator<DurativeAction> {
public:
  static constexpr char class_name[] = "durative-action";

  typedef std::shared_ptr<DurativeAction> Ptr;
  typedef VariableContainer<DurativeAction>::VariablePtr VariablePtr;
  typedef VariableContainer<DurativeAction>::VariableVector VariableVector;

  DurativeAction(const std::string &name);
  DurativeAction(const std::string &name,
                 const Formula::Ptr &duration_constraint,
                 const Formula::Ptr &precondition, const Effect::Ptr &effect);
  DurativeAction(const DurativeAction &other);
  DurativeAction &operator=(const DurativeAction &other);
  virtual ~DurativeAction();

  void set_duration_constraint(const Formula::Ptr &duration_constraint);
  const Formula::Ptr &get_duration_constraint() const;

  virtual std::ostream &print(std::ostream &o) const;

private:
  Formula::Ptr _duration_constraint;
};

// Durative action printing operator
std::ostream &operator<<(std::ostream &o, const DurativeAction &da);

class Event : public Operator<Event> {
public:
  static constexpr char class_name[] = "event";

  typedef std::shared_ptr<Event> Ptr;
  typedef VariableContainer<Event>::VariablePtr VariablePtr;
  typedef VariableContainer<Event>::VariableVector VariableVector;

  Event(const std::string &name);
  Event(const std::string &name, const Formula::Ptr &precondition,
        const Effect::Ptr &effect);
  Event(const Event &other);
  Event &operator=(const Event &other);
  virtual ~Event();
};

// Event printing operator
std::ostream &operator<<(std::ostream &o, const Event &e);

class Process : public Operator<Process> {
public:
  static constexpr char class_name[] = "process";

  typedef std::shared_ptr<Process> Ptr;
  typedef VariableContainer<Process>::VariablePtr VariablePtr;
  typedef VariableContainer<Process>::VariableVector VariableVector;

  Process(const std::string &name);
  Process(const std::string &name, const Formula::Ptr &precondition,
          const Effect::Ptr &effect);
  Process(const Process &other);
  Process &operator=(const Process &other);
  virtual ~Process();
};

// Process printing operator
std::ostream &operator<<(std::ostream &o, const Process &p);

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/operator_impl.hh"
#endif

#endif // SKDECIDE_PDDL_OPERATOR_HH
