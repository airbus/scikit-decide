#ifndef AIRLAPS_PDDL_AGGREGATED_FORMULA_HH
#define AIRLAPS_PDDL_AGGREGATED_FORMULA_HH

#include "formula.hh"
#include <vector>

namespace airlaps {

    namespace pddl {

        class AggregatedFormula : public Formula {
        public :
            typedef std::shared_ptr<AggregatedFormula> Ptr;
            typedef Formula::Ptr FormulaPtr;
            typedef std::vector<Formula::Ptr> FormulaVector;

            typedef enum {
                E_AND,
                E_OR
            } Operator;

            AggregatedFormula() {}

            AggregatedFormula(const Operator& op)
                : _operator(op) {}
            
            AggregatedFormula(const AggregatedFormula& other)
                : _operator(other._operator), _formulas(other._formulas) {}
            
            AggregatedFormula& operator= (const AggregatedFormula& other) {
                this->_operator = other._operator;
                this->_formulas = other._formulas;
                return *this;
            }

            AggregatedFormula& set_operator(const Operator& op) {
                _operator = op;
                return *this;
            }

            const Operator& get_operator() const {
                return _operator;
            }

            AggregatedFormula& append_formula(const Formula::Ptr& formula) {
                _formulas.push_back(formula);
                return *this;
            }

            /**
             * Removes the formula at a given index.
             * Throws an exception if the given index exceeds the size of the
             * conjunction formula
             */
            AggregatedFormula& remove_formula(const std::size_t& index) {
                if (index >= _formulas.size()) {
                    throw std::out_of_range("AIRLAPS exception: index " + std::to_string(index) +
                                            " exceeds the size of the conjunction formula");
                } else {
                    _formulas.erase(_formulas.begin() + index);
                    return *this;
                }
            }

            /**
             * Gets the formula at a given index.
             * Throws an exception if the given index exceeds the size of the
             * conjunction formula
             */
            const Formula::Ptr& formula_at(const std::size_t& index) {
                if (index >= _formulas.size()) {
                    throw std::out_of_range("AIRLAPS exception: index " + std::to_string(index) +
                                            " exceeds the size of the conjunction formula");
                } else {
                    return _formulas[index];
                }
            }

            const FormulaVector& get_formulas() const {
                return _formulas;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(";
                switch (_operator) {
                    case E_AND :
                        o << "and";
                        break;
                    case E_OR :
                        o << "or";
                        break;
                }
                for (const auto& f : _formulas) {
                    o << " " << *f;
                }
                o << ")";
                return o;
            }
        
        private :
            Operator _operator;
            FormulaVector _formulas;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_AGGREGATED_FORMULA_HH