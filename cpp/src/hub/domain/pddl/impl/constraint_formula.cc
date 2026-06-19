/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/constraint_formula.hh"

namespace skdecide {

namespace pddl {

// === AlwaysFormula implementation ===

AlwaysFormula::AlwaysFormula() {}

AlwaysFormula::AlwaysFormula(const Formula::Ptr &formula)
    : UnaryFormula<AlwaysFormula>(formula) {}

AlwaysFormula::AlwaysFormula(const AlwaysFormula &other)
    : UnaryFormula<AlwaysFormula>(other) {}

AlwaysFormula &AlwaysFormula::operator=(const AlwaysFormula &other) {
  dynamic_cast<UnaryFormula<AlwaysFormula> &>(*this) = other;
  return *this;
}

AlwaysFormula::~AlwaysFormula() {}

// === SometimeFormula implementation ===

SometimeFormula::SometimeFormula() {}

SometimeFormula::SometimeFormula(const Formula::Ptr &formula)
    : UnaryFormula<SometimeFormula>(formula) {}

SometimeFormula::SometimeFormula(const SometimeFormula &other)
    : UnaryFormula<SometimeFormula>(other) {}

SometimeFormula &SometimeFormula::operator=(const SometimeFormula &other) {
  dynamic_cast<UnaryFormula<SometimeFormula> &>(*this) = other;
  return *this;
}

SometimeFormula::~SometimeFormula() {}

// === AtMostOnceFormula implementation ===

AtMostOnceFormula::AtMostOnceFormula() {}

AtMostOnceFormula::AtMostOnceFormula(const Formula::Ptr &formula)
    : UnaryFormula<AtMostOnceFormula>(formula) {}

AtMostOnceFormula::AtMostOnceFormula(const AtMostOnceFormula &other)
    : UnaryFormula<AtMostOnceFormula>(other) {}

AtMostOnceFormula &
AtMostOnceFormula::operator=(const AtMostOnceFormula &other) {
  dynamic_cast<UnaryFormula<AtMostOnceFormula> &>(*this) = other;
  return *this;
}

AtMostOnceFormula::~AtMostOnceFormula() {}

// === WithinFormula implementation ===

WithinFormula::WithinFormula() {}

WithinFormula::WithinFormula(const Formula::Ptr &formula,
                             const Number::Ptr &deadline)
    : UnaryFormula<WithinFormula>(formula), _deadline(deadline) {}

WithinFormula::WithinFormula(const WithinFormula &other)
    : UnaryFormula<WithinFormula>(other), _deadline(other._deadline) {}

WithinFormula &WithinFormula::operator=(const WithinFormula &other) {
  dynamic_cast<UnaryFormula<WithinFormula> &>(*this) = other;
  this->_deadline = other._deadline;
  return *this;
}

WithinFormula::~WithinFormula() {}

void WithinFormula::set_deadline(const Number::Ptr &deadline) {
  _deadline = deadline;
}

const Number::Ptr &WithinFormula::get_deadline() const { return _deadline; }

std::ostream &WithinFormula::print(std::ostream &o) const {
  o << "(within " << *_deadline << " " << *_formula << ")";
  return o;
}

// === HoldAfterFormula implementation ===

HoldAfterFormula::HoldAfterFormula() {}

HoldAfterFormula::HoldAfterFormula(const Formula::Ptr &formula,
                                   const Number::Ptr &from)
    : UnaryFormula<HoldAfterFormula>(formula), _from(from) {}

HoldAfterFormula::HoldAfterFormula(const HoldAfterFormula &other)
    : UnaryFormula<HoldAfterFormula>(other), _from(other._from) {}

HoldAfterFormula &HoldAfterFormula::operator=(const HoldAfterFormula &other) {
  dynamic_cast<UnaryFormula<HoldAfterFormula> &>(*this) = other;
  this->_from = other._from;
  return *this;
}

HoldAfterFormula::~HoldAfterFormula() {}

void HoldAfterFormula::set_from(const Number::Ptr &from) { _from = from; }

const Number::Ptr &HoldAfterFormula::get_from() const { return _from; }

std::ostream &HoldAfterFormula::print(std::ostream &o) const {
  o << "(hold-after " << *_from << " " << *_formula << ")";
  return o;
}

// === HoldDuringFormula implementation ===

HoldDuringFormula::HoldDuringFormula() {}

HoldDuringFormula::HoldDuringFormula(const Formula::Ptr &formula,
                                     const Number::Ptr &from,
                                     const Number::Ptr &deadline)
    : UnaryFormula<HoldDuringFormula>(formula), _from(from),
      _deadline(deadline) {}

HoldDuringFormula::HoldDuringFormula(const HoldDuringFormula &other)
    : UnaryFormula<HoldDuringFormula>(other), _from(other._from),
      _deadline(other._deadline) {}

HoldDuringFormula &
HoldDuringFormula::operator=(const HoldDuringFormula &other) {
  dynamic_cast<UnaryFormula<HoldDuringFormula> &>(*this) = other;
  this->_from = other._from;
  this->_deadline = other._deadline;
  return *this;
}

HoldDuringFormula::~HoldDuringFormula() {}

void HoldDuringFormula::set_from(const Number::Ptr &from) { _from = from; }

const Number::Ptr &HoldDuringFormula::get_from() const { return _from; }

void HoldDuringFormula::set_deadline(const Number::Ptr &deadline) {
  _deadline = deadline;
}

const Number::Ptr &HoldDuringFormula::get_deadline() const { return _deadline; }

std::ostream &HoldDuringFormula::print(std::ostream &o) const {
  o << "(hold-during " << *_from << " " << *_deadline << " " << *_formula
    << ")";
  return o;
}

// === SometimeAfterFormula implementation ===

SometimeAfterFormula::SometimeAfterFormula() {}

SometimeAfterFormula::SometimeAfterFormula(const Formula::Ptr &left_formula,
                                           const Formula::Ptr &right_formula)
    : BinaryFormula<SometimeAfterFormula>(left_formula, right_formula) {}

SometimeAfterFormula::SometimeAfterFormula(const SometimeAfterFormula &other)
    : BinaryFormula<SometimeAfterFormula>(other) {}

SometimeAfterFormula &
SometimeAfterFormula::operator=(const SometimeAfterFormula &other) {
  dynamic_cast<BinaryFormula<SometimeAfterFormula> &>(*this) = other;
  return *this;
}

SometimeAfterFormula::~SometimeAfterFormula() {}

// === SometimeBeforeFormula implementation ===

SometimeBeforeFormula::SometimeBeforeFormula() {}

SometimeBeforeFormula::SometimeBeforeFormula(const Formula::Ptr &left_formula,
                                             const Formula::Ptr &right_formula)
    : BinaryFormula<SometimeBeforeFormula>(left_formula, right_formula) {}

SometimeBeforeFormula::SometimeBeforeFormula(const SometimeBeforeFormula &other)
    : BinaryFormula<SometimeBeforeFormula>(other) {}

SometimeBeforeFormula &
SometimeBeforeFormula::operator=(const SometimeBeforeFormula &other) {
  dynamic_cast<BinaryFormula<SometimeBeforeFormula> &>(*this) = other;
  return *this;
}

SometimeBeforeFormula::~SometimeBeforeFormula() {}

// === AlwaysWithinFormula implementation ===

AlwaysWithinFormula::AlwaysWithinFormula() {}

AlwaysWithinFormula::AlwaysWithinFormula(const Formula::Ptr &left_formula,
                                         const Formula::Ptr &right_formula,
                                         const Number::Ptr &deadline)
    : BinaryFormula<AlwaysWithinFormula>(left_formula, right_formula) {}

AlwaysWithinFormula::AlwaysWithinFormula(const AlwaysWithinFormula &other)
    : BinaryFormula<AlwaysWithinFormula>(other), _deadline(other._deadline) {}

AlwaysWithinFormula &
AlwaysWithinFormula::operator=(const AlwaysWithinFormula &other) {
  dynamic_cast<BinaryFormula<AlwaysWithinFormula> &>(*this) = other;
  this->_deadline = other._deadline;
  return *this;
}

AlwaysWithinFormula::~AlwaysWithinFormula() {}

void AlwaysWithinFormula::set_deadline(const Number::Ptr &deadline) {
  _deadline = deadline;
}

const Number::Ptr &AlwaysWithinFormula::get_deadline() const {
  return _deadline;
}

std::ostream &AlwaysWithinFormula::print(std::ostream &o) const {
  o << "(always-within " << *_deadline << " " << *_left_formula
    << *_right_formula << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
