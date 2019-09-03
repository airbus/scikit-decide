#ifndef AIRLAPS_PDDL_COMPARISON_FORMULA_HH
#define AIRLAPS_PDDL_COMPARISON_FORMULA_HH

#include "formula.hh"
#include "binary_expression.hh"

namespace airlaps {

    namespace pddl {

        template <typename Derived>
        class ComparisonFormula : public Formula,
                                  public BinaryExpression<Derived> {
        public :
            typedef std::shared_ptr<ComparisonFormula<Derived>> Ptr;

            ComparisonFormula() {}

            ComparisonFormula(const Expression::Ptr& left_expression,
                              const Expression::Ptr& right_expression)
                : BinaryExpression<Derived>(left_expression, right_expression) {}
            
            ComparisonFormula(const ComparisonFormula& other)
                : BinaryExpression<Derived>(other) {}
            
            ComparisonFormula& operator= (const ComparisonFormula& other) {
                dynamic_cast<BinaryExpression<Derived>&>(*this) = other;
                return *this;
            }

            virtual std::ostream& print(std::ostream& o) const {
                return BinaryExpression<Derived>::print(o);
            }

            std::string print() const {
                return Formula::print();
            }
        };


        class GreaterFormula : public ComparisonFormula<GreaterFormula> {
        public :
            static constexpr char class_name[] = ">";

            typedef std::shared_ptr<GreaterFormula> Ptr;

            GreaterFormula() {}

            GreaterFormula(const Expression::Ptr& left_expression,
                           const Expression::Ptr& right_expression)
                : ComparisonFormula<GreaterFormula>(left_expression, right_expression) {}
            
            GreaterFormula(const GreaterFormula& other)
                : ComparisonFormula<GreaterFormula>(other) {}
            
            GreaterFormula& operator= (const GreaterFormula& other) {
                dynamic_cast<ComparisonFormula<GreaterFormula>&>(*this) = other;
                return *this;
            }

            virtual ~GreaterFormula() {}
        };


        class GreaterEqFormula : public ComparisonFormula<GreaterEqFormula> {
        public :
            static constexpr char class_name[] = ">=";

            typedef std::shared_ptr<GreaterEqFormula> Ptr;

            GreaterEqFormula() {}

            GreaterEqFormula(const Expression::Ptr& left_expression,
                             const Expression::Ptr& right_expression)
                : ComparisonFormula<GreaterEqFormula>(left_expression, right_expression) {}
            
            GreaterEqFormula(const GreaterEqFormula& other)
                : ComparisonFormula<GreaterEqFormula>(other) {}
            
            GreaterEqFormula& operator= (const GreaterEqFormula& other) {
                dynamic_cast<ComparisonFormula<GreaterEqFormula>&>(*this) = other;
                return *this;
            }

            virtual ~GreaterEqFormula() {}
        };


        class LessEqFormula : public ComparisonFormula<LessEqFormula> {
        public :
            static constexpr char class_name[] = "<=";

            typedef std::shared_ptr<LessEqFormula> Ptr;

            LessEqFormula() {}

            LessEqFormula(const Expression::Ptr& left_expression,
                          const Expression::Ptr& right_expression)
                : ComparisonFormula<LessEqFormula>(left_expression, right_expression) {}
            
            LessEqFormula(const LessEqFormula& other)
                : ComparisonFormula<LessEqFormula>(other) {}
            
            LessEqFormula& operator= (const LessEqFormula& other) {
                dynamic_cast<ComparisonFormula<LessEqFormula>&>(*this) = other;
                return *this;
            }

            virtual ~LessEqFormula() {}
        };


        class LessFormula : public ComparisonFormula<LessFormula> {
        public :
            static constexpr char class_name[] = "<";

            typedef std::shared_ptr<LessFormula> Ptr;

            LessFormula() {}

            LessFormula(const Expression::Ptr& left_expression,
                        const Expression::Ptr& right_expression)
                : ComparisonFormula<LessFormula>(left_expression, right_expression) {}
            
            LessFormula(const LessFormula& other)
                : ComparisonFormula<LessFormula>(other) {}
            
            LessFormula& operator= (const LessFormula& other) {
                dynamic_cast<ComparisonFormula<LessFormula>&>(*this) = other;
                return *this;
            }

            virtual ~LessFormula() {}
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_COMPARISON_FORMULA_HH
