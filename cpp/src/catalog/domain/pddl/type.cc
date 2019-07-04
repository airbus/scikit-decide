#include <exception>
#include <sstream>

#include "type.hh"

using namespace airlaps::pddl;

const Type::Ptr Type::_object = std::make_shared<Type>("object");
const Type::Ptr Type::_number = std::make_shared<Type>("number");


Type::Type(const std::string& name)
    : TypeContainer<Type>(name) {
}


Type::Type(const Type& other)
    : TypeContainer<Type>(other) {

}


Type& Type::operator=(const Type& other) {
    dynamic_cast<TypeContainer<Type>&>(*this) = other;
    return *this;
}


std::ostream& operator<< (std::ostream& o, const Type& t) {
    return t.print(o);
}
