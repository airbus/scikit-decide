/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_ASSIGNMENT_EFFECT_HH
#define AIRLAPS_PDDL_ASSIGNMENT_EFFECT_HH

#include "effect.hh"
#include "function_effect.hh"
#include "expression.hh"

namespace airlaps {

    namespace pddl {

        template <typename Derived>
        class AssignmentEffect : public Effect {
        public :
            typedef std::shared_ptr<AssignmentEffect<Derived>> Ptr;

            AssignmentEffect() {}

            AssignmentEffect(const FunctionEffect::Ptr& function,
                             const Expression::Ptr& expression)
                : _function(function), _expression(expression) {}
            
            AssignmentEffect(const AssignmentEffect<Derived>& other)
                : _function(other._function), _expression(other._expression) {}
            
            AssignmentEffect<Derived>& operator= (const AssignmentEffect<Derived>& other) {
                this->_function = other._function;
                this->_expression = other._expression;
                return *this;
            }

            void set_function(const FunctionEffect::Ptr& function) {
                _function = function;
            }

            const FunctionEffect::Ptr& get_function() const {
                return _function;
            }

            void set_expression(const Expression::Ptr& expression) {
                _expression = expression;
            }

            const Expression::Ptr& get_expression() const {
                return _expression;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name << " " <<
                     dynamic_cast<const Effect&>(*_function) << " " << *_expression << ")";
                return o;
            }

            std::string print() const {
                return Effect::print();
            }

        private :
            FunctionEffect::Ptr _function;
            Expression::Ptr _expression;
        };


        class AssignEffect : public AssignmentEffect<AssignEffect> {
        public :
            static constexpr char class_name[] = "assign";

            typedef std::shared_ptr<AssignEffect> Ptr;

            AssignEffect() {}

            AssignEffect(const FunctionEffect::Ptr& function,
                         const Expression::Ptr& expression)
                : AssignmentEffect<AssignEffect>(function, expression) {}
            
            AssignEffect(const AssignEffect& other)
                : AssignmentEffect<AssignEffect>(other) {}
            
            AssignEffect& operator= (const AssignEffect& other) {
                dynamic_cast<AssignmentEffect<AssignEffect>&>(*this) = other;
                return *this;
            }

            virtual ~AssignEffect() {}
        };


        class ScaleUpEffect : public AssignmentEffect<ScaleUpEffect> {
        public :
            static constexpr char class_name[] = "scale-up";

            typedef std::shared_ptr<ScaleUpEffect> Ptr;

            ScaleUpEffect() {}

            ScaleUpEffect(const FunctionEffect::Ptr& function,
                          const Expression::Ptr& expression)
                : AssignmentEffect<ScaleUpEffect>(function, expression) {}
            
            ScaleUpEffect(const ScaleUpEffect& other)
                : AssignmentEffect<ScaleUpEffect>(other) {}
            
            ScaleUpEffect& operator= (const ScaleUpEffect& other) {
                dynamic_cast<AssignmentEffect<ScaleUpEffect>&>(*this) = other;
                return *this;
            }

            virtual ~ScaleUpEffect() {}
        };


        class ScaleDownEffect : public AssignmentEffect<ScaleDownEffect> {
        public :
            static constexpr char class_name[] = "scale-down";

            typedef std::shared_ptr<ScaleDownEffect> Ptr;

            ScaleDownEffect() {}

            ScaleDownEffect(const FunctionEffect::Ptr& function,
                            const Expression::Ptr& expression)
                : AssignmentEffect<ScaleDownEffect>(function, expression) {}
            
            ScaleDownEffect(const ScaleDownEffect& other)
                : AssignmentEffect<ScaleDownEffect>(other) {}
            
            ScaleDownEffect& operator= (const ScaleDownEffect& other) {
                dynamic_cast<AssignmentEffect<ScaleDownEffect>&>(*this) = other;
                return *this;
            }

            virtual ~ScaleDownEffect() {}
        };


        class IncreaseEffect : public AssignmentEffect<IncreaseEffect> {
        public :
            static constexpr char class_name[] = "increase";

            typedef std::shared_ptr<IncreaseEffect> Ptr;

            IncreaseEffect() {}

            IncreaseEffect(const FunctionEffect::Ptr& function,
                           const Expression::Ptr& expression)
                : AssignmentEffect<IncreaseEffect>(function, expression) {}
            
            IncreaseEffect(const IncreaseEffect& other)
                : AssignmentEffect<IncreaseEffect>(other) {}
            
            IncreaseEffect& operator= (const IncreaseEffect& other) {
                dynamic_cast<AssignmentEffect<IncreaseEffect>&>(*this) = other;
                return *this;
            }

            virtual ~IncreaseEffect() {}
        };


        class DecreaseEffect : public AssignmentEffect<DecreaseEffect> {
        public :
            static constexpr char class_name[] = "decrease";

            typedef std::shared_ptr<DecreaseEffect> Ptr;

            DecreaseEffect() {}

            DecreaseEffect(const FunctionEffect::Ptr& function,
                           const Expression::Ptr& expression)
                : AssignmentEffect<DecreaseEffect>(function, expression) {}
            
            DecreaseEffect(const DecreaseEffect& other)
                : AssignmentEffect<DecreaseEffect>(other) {}
            
            DecreaseEffect& operator= (const DecreaseEffect& other) {
                dynamic_cast<AssignmentEffect<DecreaseEffect>&>(*this) = other;
                return *this;
            }

            virtual ~DecreaseEffect() {}
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_ASSIGNMENT_EFFECT_HH
