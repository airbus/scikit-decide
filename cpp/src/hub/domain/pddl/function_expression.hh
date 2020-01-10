/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_FUNCTION_EXPRESSION_HH
#define SKDECIDE_PDDL_FUNCTION_EXPRESSION_HH

#include "expression.hh"
#include "function.hh"
#include "term_container.hh"

namespace skdecide {

    namespace pddl {

        template <typename Derived = nullptr_t>
        class FunctionExpression : public Expression,
                                   public TermContainer<typename std::conditional<std::is_null_pointer<Derived>::value, FunctionExpression<>, Derived>::type> {
        public :
            static constexpr char class_name[] = "function expression";
            typedef TermContainer<typename std::conditional<std::is_null_pointer<Derived>::value, FunctionExpression<>, Derived>::type> TermContainerType;

            typedef std::shared_ptr<FunctionExpression> Ptr;
            typedef typename TermContainerType::TermPtr TermPtr;
            typedef typename TermContainerType::TermVector TermVector;

            FunctionExpression() {}

            FunctionExpression(const Function::Ptr& function,
                               const TermContainerType& terms)
                : TermContainerType(terms), _function(function) {}
            
            FunctionExpression(const FunctionExpression& other)
                : TermContainerType(other), _function(other._function) {}
            
            FunctionExpression& operator= (const FunctionExpression& other) {
                dynamic_cast<TermContainerType&>(*this) = other;
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
                return TermContainerType::print(o);
            }
        
        private :
            Function::Ptr _function;
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_FUNCTION_EXPRESSION_HH