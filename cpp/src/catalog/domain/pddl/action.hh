#ifndef AIRLAPS_PDDL_ACTION_HH
#define AIRLAPS_PDDL_ACTION_HH

#include "identifier.hh"
#include "variable_container.hh"
#include "binary_effect.hh"

namespace airlaps {

    namespace pddl {

        class Action : public Identifier,
                       public VariableContainer<Action>,
                       public BinaryEffect { // BinaryEffect brings in precondition and effect logics
        public :
            static constexpr char class_name[] = "action";

            typedef std::shared_ptr<Action> Ptr;
            typedef VariableContainer<Action>::VariablePtr VariablePtr;
            typedef VariableContainer<Action>::VariableVector VariableVector;

            Action(const std::string& name)
                : Identifier(name) {}
            
            Action(const std::string& name,
                   const Formula::Ptr& precondition,
                   const Effect::Ptr& effect)
                : Identifier(name), BinaryEffect(precondition, effect) {}

            Action(const Action& other)
                : Identifier(other),
                  VariableContainer<Action>(other),
                  BinaryEffect(other) {}

            Action& operator=(const Action& other) {
                dynamic_cast<Identifier&>(*this) = other;
                dynamic_cast<VariableContainer<Action>&>(*this) = other;
                dynamic_cast<BinaryEffect&>(*this) = other;
                return *this;
            }

            virtual ~Action() {}
        };


        class DurativeAction : public Action {
        public :
            typedef std::shared_ptr<DurativeAction> Ptr;

            DurativeAction(const std::string& name)
                : Action(name) {}
            
            DurativeAction(const std::string& name,
                           const Formula::Ptr& duration_constraint,
                           const Formula::Ptr& precondition,
                           const Effect::Ptr& effect)
                : Action(name, precondition, effect),
                  _duration_constraint(duration_constraint) {}

            DurativeAction(const DurativeAction& other)
                : Action(other),
                  _duration_constraint(other._duration_constraint) {}

            DurativeAction& operator=(const DurativeAction& other) {
                dynamic_cast<Action&>(*this) = other;
                this->_duration_constraint = other._duration_constraint;
                return *this;
            }

            virtual ~DurativeAction() {}
        
        private :
            Formula::Ptr _duration_constraint;
        };

    } // namespace pddl

} // namespace airlaps

// Object printing operator
inline std::ostream& operator<<(std::ostream& o, const airlaps::pddl::Action& p) {
    return p.print(o);
}

#endif // AIRLAPS_PDDL_ACTION_HH
