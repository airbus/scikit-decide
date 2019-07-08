#ifndef AIRLAPS_PDDL_CONSTRAINT_FORMULA_HH
#define AIRLAPS_PDDL_CONSTRAINT_FORMULA_HH

#include "formula.hh"

namespace airlaps {

    namespace pddl {

        class ConstraintFormula : public Formula {
        public :
            typedef std::shared_ptr<ConstraintFormula> Ptr;

            typedef enum {
                E_ATEND,
                E_ALWAYS,
                E_SOMETIME,
                E_WITHIN,
                E_ATMOSTONCE,
                E_SOMETIMEAFTER,
                E_SOMETIMEBEFORE,
                E_ALWAYSWITHIN,
                E_HOLDDURING,
                E_HOLDAFTER
            } Sort;

            ConstraintFormula() {}

            ConstraintFormula(const Sort& sort)
                : _sort(sort) {}

            ConstraintFormula(const Sort& sort,
                            const Formula::Ptr& requirement,
                            const Formula::Ptr& trigger,
                            double deadline,
                            double from)
                : _sort(sort), _requirement(requirement), _trigger(trigger),
                _deadline(deadline), _from(from) {}
                
            ConstraintFormula(const ConstraintFormula& other)
                : _sort(other._sort), _requirement(other._requirement), _trigger(other._trigger),
                _deadline(other._deadline), _from(other._from) {}

            ConstraintFormula& operator= (const ConstraintFormula& other) {
                this->_sort = other._sort;
                this->_requirement = other._requirement;
                this->_trigger = other._trigger;
                this->_deadline = other._deadline;
                this->_from = other._from;
                return *this;
            }

            virtual ~ConstraintFormula() {}

            ConstraintFormula& set_sort(const Sort& sort) {
                _sort = sort;
                return *this;
            }

            const Sort& get_sort() const {
                return _sort;
            }

            ConstraintFormula& set_requirement(const Formula::Ptr& requirement) {
                _requirement = requirement;
                return *this;
            }

            const Formula::Ptr& get_requirement() const {
                return _requirement;
            }

            ConstraintFormula& set_trigger(const Formula::Ptr& trigger) {
                _trigger = trigger;
                return *this;
            }

            const Formula::Ptr& get_trigger() const {
                return _trigger;
            }

            ConstraintFormula& set_deadline(double deadline) {
                _deadline = deadline;
                return *this;
            }

            double get_deadline() const {
                return _deadline;
            }

            ConstraintFormula& set_from(double from) {
                _from = from;
                return *this;
            }

            double get_from() const {
                return _from;
            }

            virtual std::ostream& print(std::ostream& o) const {
                switch (_sort) {
                    case E_ATEND :
                        o << "(at end" << *_requirement << ")";
                        return o;
                    
                    case E_ALWAYS :
                        o << "(always" << *_requirement << ")";
                        return o;
                    
                    case E_SOMETIME :
                        o << "(sometime" << *_requirement << ")";
                        return o;
                    
                    case E_WITHIN :
                        o << "(within " << _deadline << " " << *_requirement << ")";
                        return o;
                    
                    case E_ATMOSTONCE :
                        o << "(at-most-once" << *_requirement << ")";
                        return o;
                    
                    case E_SOMETIMEAFTER :
                        o << "(sometime-after " << *_requirement << " " << *_trigger << ")";
                        return o;
                    
                    case E_SOMETIMEBEFORE :
                        o << "(sometime-before " << *_requirement << " " << *_trigger << ")";
                        return o;
                    
                    case E_ALWAYSWITHIN :
                        o << "(always-within " << _deadline << " " << *_requirement << " " << *_trigger << ")";
                        return o;
                    
                    case E_HOLDDURING :
                        o << "(hold-during " << _from << " " << _deadline << " " << *_requirement << ")";
                        return o;
                    
                    case E_HOLDAFTER :
                        o << "(hold-after " << _from << " " << *_requirement << ")";
                        return o;
                }
                return o;
            }

        private :
            Sort _sort;
            Formula::Ptr _requirement;
            Formula::Ptr _trigger;
            double _deadline;
            double _from;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_CONSTRAINT_FORMULA_HH