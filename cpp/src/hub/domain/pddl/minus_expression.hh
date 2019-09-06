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
