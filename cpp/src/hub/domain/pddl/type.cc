/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <exception>
#include <sstream>

#include "type.hh"

using namespace skdecide::pddl;

const Type::Ptr Type::_object = std::make_shared<Type>("object");
const Type::Ptr Type::_number = std::make_shared<Type>("number");


Type::Type(const std::string& name)
    : Identifier(name) {
}


Type::Type(const Type& other)
    : Identifier(other), TypeContainer<Type>(other) {

}


Type& Type::operator=(const Type& other) {
    dynamic_cast<Identifier&>(*this) = other;
    dynamic_cast<TypeContainer<Type>&>(*this) = other;
    return *this;
}


std::ostream& operator<< (std::ostream& o, const Type& t) {
    return t.print(o);
}
