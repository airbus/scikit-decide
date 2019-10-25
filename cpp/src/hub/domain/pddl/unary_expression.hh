/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_UNARY_EXPRESSION_HH
#define AIRLAPS_PDDL_UNARY_EXPRESSION_HH

#include "expression.hh"

namespace airlaps {

    namespace pddl {

        template <typename Derived>
        class UnaryExpression : public Expression {
        public :
            typedef std::shared_ptr<UnaryExpression<Derived>> Ptr;

            UnaryExpression() {}

            UnaryExpression(const Expression::Ptr& expression)
                : _expression(expression) {}
            
            UnaryExpression(const UnaryExpression<Derived>& other)
                : _expression(other._expression) {}
            
            UnaryExpression<Derived>& operator= (const UnaryExpression<Derived>& other) {
                this->_expression = other._expression;
                return *this;
            }

            virtual ~UnaryExpression() {}

            void set_expression(const Expression::Ptr& expression) {
                _expression = expression;
            }

            const Expression::Ptr& get_expression() const {
                return _expression;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name << " " << *_expression << ")";
                return o;
            }

        protected :
            Expression::Ptr _expression;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_UNARY_EXPRESSION_HH
