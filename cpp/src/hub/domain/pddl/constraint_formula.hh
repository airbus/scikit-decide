/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_CONSTRAINT_FORMULA_HH
#define AIRLAPS_PDDL_CONSTRAINT_FORMULA_HH

#include "formula.hh"
#include "unary_formula.hh"
#include "binary_formula.hh"

namespace airlaps {

    namespace pddl {
        
        class AlwaysFormula : public UnaryFormula<AlwaysFormula> {
        public :
            static constexpr char class_name[] = "always";

            typedef std::shared_ptr<AlwaysFormula> Ptr;

            AlwaysFormula() {}

            AlwaysFormula(const Formula::Ptr& formula)
                : UnaryFormula<AlwaysFormula>(formula) {}

            AlwaysFormula(const AlwaysFormula& other)
                : UnaryFormula<AlwaysFormula>(other) {}
            
            AlwaysFormula& operator= (const AlwaysFormula& other) {
                dynamic_cast<AlwaysFormula&>(*this) = other;
                return *this;
            }

            virtual ~AlwaysFormula() {}
        };


        class SometimeFormula : public UnaryFormula<SometimeFormula> {
        public :
            static constexpr char class_name[] = "sometime";

            typedef std::shared_ptr<SometimeFormula> Ptr;

            SometimeFormula() {}

            SometimeFormula(const Formula::Ptr& formula)
                : UnaryFormula<SometimeFormula>(formula) {}

            SometimeFormula(const SometimeFormula& other)
                : UnaryFormula<SometimeFormula>(other) {}
            
            SometimeFormula& operator= (const SometimeFormula& other) {
                dynamic_cast<SometimeFormula&>(*this) = other;
                return *this;
            }

            virtual ~SometimeFormula() {}
        };


        class AtMostOnceFormula : public UnaryFormula<AtMostOnceFormula> {
        public :
            static constexpr char class_name[] = "at-most-once";

            typedef std::shared_ptr<AtMostOnceFormula> Ptr;

            AtMostOnceFormula() {}

            AtMostOnceFormula(const Formula::Ptr& formula)
                : UnaryFormula<AtMostOnceFormula>(formula) {}

            AtMostOnceFormula(const AtMostOnceFormula& other)
                : UnaryFormula<AtMostOnceFormula>(other) {}
            
            AtMostOnceFormula& operator= (const AtMostOnceFormula& other) {
                dynamic_cast<AtMostOnceFormula&>(*this) = other;
                return *this;
            }

            virtual ~AtMostOnceFormula() {}
        };


        class WithinFormula : public UnaryFormula<WithinFormula> {
        public :
            static constexpr char class_name[] = "within";

            typedef std::shared_ptr<WithinFormula> Ptr;

            WithinFormula() {}

            WithinFormula(const Formula::Ptr& formula, const double& deadline)
                : UnaryFormula<WithinFormula>(formula), _deadline(deadline) {}

            WithinFormula(const WithinFormula& other)
                : UnaryFormula<WithinFormula>(other),
                  _deadline(other._deadline) {}
            
            WithinFormula& operator= (const WithinFormula& other) {
                dynamic_cast<WithinFormula&>(*this) = other;
                this->_deadline = other._deadline;
                return *this;
            }

            virtual ~WithinFormula() {}

            void set_deadline(const double& deadline) {
                _deadline = deadline;
            }

            const double& get_deadline() const {
                return _deadline;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(within " << _deadline << " " << *_formula << ")";
                return o;
            }
        
        private :
            double _deadline;
        };


        class HoldAfterFormula : public UnaryFormula<HoldAfterFormula> {
        public :
            static constexpr char class_name[] = "hold-after";

            typedef std::shared_ptr<HoldAfterFormula> Ptr;

            HoldAfterFormula() {}

            HoldAfterFormula(const Formula::Ptr& formula, const double& from)
                : UnaryFormula<HoldAfterFormula>(formula), _from(from) {}

            HoldAfterFormula(const HoldAfterFormula& other)
                : UnaryFormula<HoldAfterFormula>(other),
                  _from(other._from) {}
            
            HoldAfterFormula& operator= (const HoldAfterFormula& other) {
                dynamic_cast<HoldAfterFormula&>(*this) = other;
                this->_from = other._from;
                return *this;
            }

            virtual ~HoldAfterFormula() {}

            void set_from(const double& from) {
                _from = from;
            }

            const double& get_from() const {
                return _from;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(hold-after " << _from << " " << *_formula << ")";
                return o;
            }
        
        private :
            double _from;
        };


        class HoldDuringFormula : public UnaryFormula<HoldDuringFormula> {
        public :
            static constexpr char class_name[] = "hold-during";

            typedef std::shared_ptr<HoldDuringFormula> Ptr;

            HoldDuringFormula() {}

            HoldDuringFormula(const Formula::Ptr& formula,
                              const double& from,
                              const double& deadline)
                : UnaryFormula<HoldDuringFormula>(formula),
                  _from(from), _deadline(deadline) {}

            HoldDuringFormula(const HoldDuringFormula& other)
                : UnaryFormula<HoldDuringFormula>(other),
                  _from(other._from), _deadline(other._deadline) {}
            
            HoldDuringFormula& operator= (const HoldDuringFormula& other) {
                dynamic_cast<HoldDuringFormula&>(*this) = other;
                this->_from = other._from;
                this->_deadline = other._deadline;
                return *this;
            }

            virtual ~HoldDuringFormula() {}

            void set_from(const double& from) {
                _from = from;
            }

            const double& get_from() const {
                return _from;
            }

            void set_deadline(const double& deadline) {
                _deadline = deadline;
            }

            const double& get_deadline() const {
                return _deadline;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(hold-during " << _from << " " << _deadline << " " << *_formula << ")";
                return o;
            }
        
        private :
            double _from;
            double _deadline;
        };


        class SometimeAfterFormula : public BinaryFormula<SometimeAfterFormula> {
        public :
            static constexpr char class_name[] = "sometime-after";

            typedef std::shared_ptr<SometimeAfterFormula> Ptr;

            SometimeAfterFormula() {}

            SometimeAfterFormula(const Formula::Ptr& left_formula,
                                 const Formula::Ptr& right_formula)
                : BinaryFormula<SometimeAfterFormula>(left_formula, right_formula) {}
            
            SometimeAfterFormula(const SometimeAfterFormula& other)
                : BinaryFormula<SometimeAfterFormula>(other) {}
            
            SometimeAfterFormula& operator= (const SometimeAfterFormula& other) {
                dynamic_cast<BinaryFormula<SometimeAfterFormula>&>(*this) = other;
                return *this;
            }

            virtual ~SometimeAfterFormula() {}
        };


        class SometimeBeforeFormula : public BinaryFormula<SometimeBeforeFormula> {
        public :
            static constexpr char class_name[] = "sometime-before";

            typedef std::shared_ptr<SometimeBeforeFormula> Ptr;

            SometimeBeforeFormula() {}

            SometimeBeforeFormula(const Formula::Ptr& left_formula,
                                  const Formula::Ptr& right_formula)
                : BinaryFormula<SometimeBeforeFormula>(left_formula, right_formula) {}
            
            SometimeBeforeFormula(const SometimeBeforeFormula& other)
                : BinaryFormula<SometimeBeforeFormula>(other) {}
            
            SometimeBeforeFormula& operator= (const SometimeBeforeFormula& other) {
                dynamic_cast<BinaryFormula<SometimeBeforeFormula>&>(*this) = other;
                return *this;
            }

            virtual ~SometimeBeforeFormula() {}
        };


        class AlwaysWithinFormula : public BinaryFormula<AlwaysWithinFormula> {
        public :
            static constexpr char class_name[] = "always-within";

            typedef std::shared_ptr<AlwaysWithinFormula> Ptr;

            AlwaysWithinFormula() {}

            AlwaysWithinFormula(const Formula::Ptr& left_formula,
                                const Formula::Ptr& right_formula,
                                const double& deadline)
                : BinaryFormula<AlwaysWithinFormula>(left_formula, right_formula) {}
            
            AlwaysWithinFormula(const AlwaysWithinFormula& other)
                : BinaryFormula<AlwaysWithinFormula>(other), _deadline(other._deadline) {}
            
            AlwaysWithinFormula& operator= (const AlwaysWithinFormula& other) {
                dynamic_cast<BinaryFormula<AlwaysWithinFormula>&>(*this) = other;
                this->_deadline = other._deadline;
                return *this;
            }

            virtual ~AlwaysWithinFormula() {}

            void set_deadline(const double& deadline) {
                _deadline = deadline;
            }

            const double& get_deadline() const {
                return _deadline;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(always-within " << _deadline << " " << *_left_formula << *_right_formula << ")";
                return o;
            }
        
        private :
            double _deadline;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_CONSTRAINT_FORMULA_HH