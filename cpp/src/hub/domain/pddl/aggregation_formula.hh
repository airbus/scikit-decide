/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_AGGREGATION_FORMULA_HH
#define SKDECIDE_PDDL_AGGREGATION_FORMULA_HH

#include "formula.hh"
#include <vector>

namespace skdecide {

    namespace pddl {

        template <typename Derived>
        class AggregationFormula : public Formula {
        public :
            typedef std::shared_ptr<AggregationFormula<Derived>> Ptr;
            typedef Formula::Ptr FormulaPtr;
            typedef std::vector<Formula::Ptr> FormulaVector;

            AggregationFormula() {}
            
            AggregationFormula(const AggregationFormula& other)
                : _formulas(other._formulas) {}
            
            AggregationFormula& operator= (const AggregationFormula& other) {
                this->_formulas = other._formulas;
                return *this;
            }

            virtual ~AggregationFormula() {}

            AggregationFormula& append_formula(const Formula::Ptr& formula) {
                _formulas.push_back(formula);
                return *this;
            }

            /**
             * Removes the formula at a given index.
             * Throws an exception if the given index exceeds the size of the
             * aggregation formula
             */
            AggregationFormula& remove_formula(const std::size_t& index) {
                if (index >= _formulas.size()) {
                    throw std::out_of_range("SKDECIDE exception: index " + std::to_string(index) +
                                            " exceeds the size of the '" + Derived::class_name + "' formula");
                } else {
                    _formulas.erase(_formulas.begin() + index);
                    return *this;
                }
            }

            /**
             * Gets the formula at a given index.
             * Throws an exception if the given index exceeds the size of the
             * aggregation formula
             */
            const Formula::Ptr& formula_at(const std::size_t& index) {
                if (index >= _formulas.size()) {
                    throw std::out_of_range("SKDECIDE exception: index " + std::to_string(index) +
                                            " exceeds the size of the '" + Derived::class_name + "' formula");
                } else {
                    return _formulas[index];
                }
            }

            const FormulaVector& get_formulas() const {
                return _formulas;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name;
                for (const auto& f : _formulas) {
                    o << " " << *f;
                }
                o << ")";
                return o;
            }
        
        private :
            FormulaVector _formulas;
        };


        class ConjunctionFormula : public AggregationFormula<ConjunctionFormula> {
        public :
            static constexpr char class_name[] = "and";

            typedef std::shared_ptr<ConjunctionFormula> Ptr;
            typedef AggregationFormula<ConjunctionFormula>::FormulaPtr FormulaPtr;
            typedef std::vector<FormulaPtr> FormulaVector;

            ConjunctionFormula() {}
            
            ConjunctionFormula(const ConjunctionFormula& other)
                : AggregationFormula(other) {}
            
            ConjunctionFormula& operator= (const ConjunctionFormula& other) {
                dynamic_cast<AggregationFormula&>(*this) = other;
                return *this;
            }

            virtual ~ConjunctionFormula() {}
        };


        class DisjunctionFormula : public AggregationFormula<DisjunctionFormula> {
        public :
            static constexpr char class_name[] = "or";

            typedef std::shared_ptr<DisjunctionFormula> Ptr;
            typedef AggregationFormula<DisjunctionFormula>::FormulaPtr FormulaPtr;
            typedef std::vector<FormulaPtr> FormulaVector;

            DisjunctionFormula() {}
            
            DisjunctionFormula(const DisjunctionFormula& other)
                : AggregationFormula(other) {}
            
            DisjunctionFormula& operator= (const DisjunctionFormula& other) {
                dynamic_cast<AggregationFormula&>(*this) = other;
                return *this;
            }

            virtual ~DisjunctionFormula() {}
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_AGGREGATION_FORMULA_HH