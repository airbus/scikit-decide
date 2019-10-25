/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_MINUS_EXPRESSION_HH
#define AIRLAPS_PDDL_MINUS_EXPRESSION_HH

#include "unary_expression.hh"

namespace airlaps {

    namespace pddl {

        class MinusExpression : public UnaryExpression<MinusExpression> {
        public :
            static constexpr char class_name[] = "-";

            typedef std::shared_ptr<MinusExpression> Ptr;

            MinusExpression() {}

            MinusExpression(const Expression::Ptr& expression)
                : UnaryExpression<MinusExpression>(expression) {}
            
            MinusExpression(const MinusExpression& other)
                : UnaryExpression<MinusExpression>(other) {}
            
            MinusExpression& operator= (const MinusExpression& other) {
                dynamic_cast<UnaryExpression<MinusExpression>&>(*this) = other;
                return *this;
            }

            virtual ~MinusExpression() {}
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_MINUS_EXPRESSION_HH
