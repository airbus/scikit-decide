#ifndef AIRLAPS_PDDL_PREFERENCE_HH
#define AIRLAPS_PDDL_PREFERENCE_HH

#include "formula.hh"
#include "identifier.hh"

namespace airlaps {

    namespace pddl {

        class Preference : public Formula,
                           public Identifier {
        public :
            typedef std::shared_ptr<Preference> Ptr;

            Preference()
                : Identifier("anonymous") {}

            Preference(const Formula::Ptr& formula,
                       const std::string& name = "anonymous")
                : Identifier(name), _formula(formula) {}
            
            Preference(const Preference& other)
                : Identifier(other), _formula(other._formula) {}
            
            Preference& operator= (const Preference& other) {
                dynamic_cast<Identifier&>(*this) = other;
                this->_formula = other._formula;
                return *this;
            }

            virtual ~Preference() {}

            Preference& set_formula(const Formula::Ptr& formula) {
                _formula = formula;
                return *this;
            }

            const Formula::Ptr& get_formula() const {
                return _formula;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(preference " << ((_name != "anonymous ")?_name:"") << *_formula << ")";
                return o;
            }
        
        private :
            Formula::Ptr _formula;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_PREFERENCE_HH