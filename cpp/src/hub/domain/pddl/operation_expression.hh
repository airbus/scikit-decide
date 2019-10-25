/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_OPERATION_EXPRESSION_HH
#define AIRLAPS_PDDL_OPERATION_EXPRESSION_HH

#include "binary_expression.hh"

namespace airlaps {

    namespace pddl {

        class AddExpression : public BinaryExpression<AddExpression> {
        public :
            static constexpr char class_name[] = "+";

            typedef std::shared_ptr<AddExpression> Ptr;

            AddExpression() {}

            AddExpression(const Expression::Ptr& left_expression,
                          const Expression::Ptr& right_expression)
                : BinaryExpression<AddExpression>(left_expression, right_expression) {}
            
            AddExpression(const AddExpression& other)
                : BinaryExpression<AddExpression>(other) {}
            
            AddExpression& operator= (const AddExpression& other) {
                dynamic_cast<BinaryExpression<AddExpression>&>(*this) = other;
                return *this;
            }

            virtual ~AddExpression() {}
        };


        class SubExpression : public BinaryExpression<SubExpression> {
        public :
            static constexpr char class_name[] = "-";

            typedef std::shared_ptr<SubExpression> Ptr;

            SubExpression() {}

            SubExpression(const Expression::Ptr& left_expression,
                          const Expression::Ptr& right_expression)
                : BinaryExpression<SubExpression>(left_expression, right_expression) {}
            
            SubExpression(const SubExpression& other)
                : BinaryExpression<SubExpression>(other) {}
            
            SubExpression& operator= (const SubExpression& other) {
                dynamic_cast<BinaryExpression<SubExpression>&>(*this) = other;
                return *this;
            }

            virtual ~SubExpression() {}
        };


        class MulExpression : public BinaryExpression<MulExpression> {
        public :
            static constexpr char class_name[] = "*";

            typedef std::shared_ptr<MulExpression> Ptr;

            MulExpression() {}

            MulExpression(const Expression::Ptr& left_expression,
                          const Expression::Ptr& right_expression)
                : BinaryExpression<MulExpression>(left_expression, right_expression) {}
            
            MulExpression(const MulExpression& other)
                : BinaryExpression<MulExpression>(other) {}
            
            MulExpression& operator= (const MulExpression& other) {
                dynamic_cast<BinaryExpression<MulExpression>&>(*this) = other;
                return *this;
            }

            virtual ~MulExpression() {}
        };


        class DivExpression : public BinaryExpression<DivExpression> {
        public :
            static constexpr char class_name[] = "/";

            typedef std::shared_ptr<DivExpression> Ptr;

            DivExpression() {}

            DivExpression(const Expression::Ptr& left_expression,
                          const Expression::Ptr& right_expression)
                : BinaryExpression<DivExpression>(left_expression, right_expression) {}
            
            DivExpression(const DivExpression& other)
                : BinaryExpression<DivExpression>(other) {}
            
            DivExpression& operator= (const DivExpression& other) {
                dynamic_cast<BinaryExpression<DivExpression>&>(*this) = other;
                return *this;
            }

            virtual ~DivExpression() {}
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_OPERATION_EXPRESSION_HH
