#ifndef AIRLAPS_PDDL_QUANTIFIED_FORMULA_HH
#define AIRLAPS_PDDL_QUANTIFIED_FORMULA_HH

#include "formula.hh"
#include "variable_container.hh"

namespace airlaps {

    namespace pddl {

        class QuantifiedFormula : public Formula,
                                   public Identifier,
                                   public VariableContainer<QuantifiedFormula> {
        public :
            static constexpr char class_name[] = "quantified formula";

            typedef std::shared_ptr<QuantifiedFormula> Ptr;
            typedef VariableContainer<QuantifiedFormula>::VariablePtr VariablePtr;
            typedef VariableContainer<QuantifiedFormula>::VariableVector VariableVector;
            
            typedef enum {
                E_FORALL,
                E_EXISTS
            } Quantifier;

            QuantifiedFormula() {}

            QuantifiedFormula(const Quantifier& quantifier)
                : _quantifier(quantifier) {}

            QuantifiedFormula(const Quantifier& quantifier,
                              const Formula::Ptr& formula,
                              const VariableContainer<QuantifiedFormula>& variables)
                : VariableContainer<QuantifiedFormula>(variables),
                  _quantifier(quantifier), _formula(formula) {}
            
            QuantifiedFormula(const QuantifiedFormula& other)
                : VariableContainer<QuantifiedFormula>(other),
                  _quantifier(other._quantifier),
                  _formula(other._formula) {}
            
            QuantifiedFormula& operator= (const QuantifiedFormula& other) {
                dynamic_cast<VariableContainer<QuantifiedFormula>&>(*this) = other;
                this->_quantifier = other._quantifier;
                this->_formula = other._formula;
                return *this;
            }

            virtual ~QuantifiedFormula() {}

            std::string get_name() const {
                switch (_quantifier)
                {
                case E_FORALL:
                    return "forall";
                
                case E_EXISTS:
                    return "exists";
                }
                return "";
            }

            QuantifiedFormula& set_quantifier(const Quantifier& quantifier) {
                _quantifier = quantifier;
                return *this;
            }

            const Quantifier& get_quantifier() const {
                return _quantifier;
            }

            QuantifiedFormula& set_formula(const Formula::Ptr& formula) {
                _formula = formula;
                return *this;
            }

            const Formula::Ptr& get_formula() const {
                return _formula;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << get_name() << "(";
                for (const auto& v : get_variables()) {
                    o << " " << *v;
                }
                o << ") " << *_formula << ")";
                return o;
            }
        
        private :
            Quantifier _quantifier;
            Formula::Ptr _formula;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_QUANTIFIED_FORMULA_HH