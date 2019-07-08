#ifndef AIRLAPS_PDDL_PROPOSITION_FORMULA_HH
#define AIRLAPS_PDDL_PROPOSITION_FORMULA_HH

#include "formula.hh"
#include "identifier.hh"
#include "term_container.hh"

namespace airlaps {

    namespace pddl {

        class PropositionFormula : public Formula,
                                   public Identifier,
                                   public TermContainer<PropositionFormula> {
        public :
            static constexpr char class_name[] = "proposition formula";

            typedef std::shared_ptr<PropositionFormula> Ptr;
            typedef TermContainer<PropositionFormula>::TermPtr TermPtr;
            typedef TermContainer<PropositionFormula>::TermVector TermVector;

            PropositionFormula() {}

            PropositionFormula(const std::string& name,
                               const TermContainer<PropositionFormula>& terms)
                : Identifier(name), TermContainer<PropositionFormula>(terms) {}
            
            PropositionFormula(const PropositionFormula& other)
                : Identifier(other), TermContainer<PropositionFormula>(other) {}
            
            PropositionFormula& operator= (const PropositionFormula& other) {
                dynamic_cast<Identifier&>(*this) = other;
                dynamic_cast<TermContainer<PropositionFormula>&>(*this) = other;
                return *this;
            }

            virtual ~PropositionFormula() {}

            virtual std::ostream& print(std::ostream& o) const {
                return TermContainer<PropositionFormula>::print(o);
            }
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_PROPOSITION_FORMULA_HH