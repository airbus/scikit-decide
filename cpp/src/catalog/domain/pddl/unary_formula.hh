#ifndef AIRLAPS_PDDL_UNARY_FORMULA_HH
#define AIRLAPS_PDDL_UNARY_FORMULA_HH

#include "formula.hh"

namespace airlaps {

    namespace pddl {

        template <typename Derived>
        class UnaryFormula : public Formula {
        public :
            typedef std::shared_ptr<UnaryFormula<Derived>> Ptr;

            UnaryFormula() {}

            UnaryFormula(const Formula::Ptr& formula)
                : _formula(formula) {}
            
            UnaryFormula(const UnaryFormula<Derived>& other)
                : _formula(other._formula) {}
            
            UnaryFormula<Derived>& operator= (const UnaryFormula<Derived>& other) {
                this->_formula = other._formula;
                return *this;
            }

            virtual ~UnaryFormula() {}

            void set_formula(const Formula::Ptr& formula) {
                _formula = formula;
            }

            const Formula::Ptr& get_formula() const {
                return _formula;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name << " " << *_formula << ")";
                return o;
            }

        private :
            Formula::Ptr _formula;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_UNARY_FORMULA_HH
