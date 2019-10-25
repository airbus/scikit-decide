/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_FUNCTION_EXPRESSION_HH
#define AIRLAPS_PDDL_FUNCTION_EXPRESSION_HH

#include "expression.hh"
#include "function.hh"
#include "term_container.hh"

namespace airlaps {

    namespace pddl {

        class FunctionExpression : public Expression,
                                   public TermContainer<FunctionExpression> {
        public :
            static constexpr char class_name[] = "function expression";

            typedef std::shared_ptr<FunctionExpression> Ptr;
            typedef TermContainer<FunctionExpression>::TermPtr TermPtr;
            typedef TermContainer<FunctionExpression>::TermVector TermVector;

            FunctionExpression() {}

            FunctionExpression(const Function::Ptr& function,
                               const TermContainer<FunctionExpression>& terms)
                : TermContainer<FunctionExpression>(terms), _function(function) {}
            
            FunctionExpression(const FunctionExpression& other)
                : TermContainer<FunctionExpression>(other), _function(other._function) {}
            
            FunctionExpression& operator= (const FunctionExpression& other) {
                dynamic_cast<TermContainer<FunctionExpression>&>(*this) = other;
                this->_function = other._function;
                return *this;
            }

            virtual ~FunctionExpression() {}

            void set_function(const Function::Ptr& function) {
                _function = function;
            }

            const Function::Ptr& get_function() const {
                return _function;
            }

            const std::string& get_name() const {
                return _function->get_name();
            }

            virtual std::ostream& print(std::ostream& o) const {
                return TermContainer<FunctionExpression>::print(o);
            }
        
        private :
            Function::Ptr _function;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_FUNCTION_EXPRESSION_HH