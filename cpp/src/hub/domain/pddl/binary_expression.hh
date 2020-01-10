/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_BINARY_EXPRESSION_HH
#define SKDECIDE_PDDL_BINARY_EXPRESSION_HH

#include "expression.hh"

namespace skdecide {

    namespace pddl {

        template <typename Derived>
        class BinaryExpression : public Expression {
        public :
            typedef std::shared_ptr<BinaryExpression<Derived>> Ptr;

            BinaryExpression() {}

            BinaryExpression(const Expression::Ptr& left_expression,
                             const Expression::Ptr& right_expression)
                : _left_expression(left_expression), _right_expression(right_expression) {}
            
            BinaryExpression(const BinaryExpression<Derived>& other)
                : _left_expression(other._left_expression),
                  _right_expression(other._right_expression) {}
            
            BinaryExpression<Derived>& operator= (const BinaryExpression<Derived>& other) {
                this->_left_expression = other._left_expression;
                this->_right_expression = other._right_expression;
                return *this;
            }

            virtual ~BinaryExpression() {}

            void set_left_expression(const Expression::Ptr& expression) {
                _left_expression = expression;
            }

            const Expression::Ptr& get_left_expression() const {
                return _left_expression;
            }

            void set_right_expression(const Expression::Ptr& expression) {
                _right_expression = expression;
            }

            const Expression::Ptr& get_right_expression() const {
                return _right_expression;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name << " " << *_left_expression << " " << *_right_expression << ")";
                return o;
            }

        protected :
            Expression::Ptr _left_expression;
            Expression::Ptr _right_expression;
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_BINARY_EXPRESSION_HH
