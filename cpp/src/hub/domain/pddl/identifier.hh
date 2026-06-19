/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_IDENTIFIER_HH
#define SKDECIDE_PDDL_IDENTIFIER_HH

#include <string>

#include "utils/string_converter.hh"

namespace skdecide {

namespace pddl {

class Identifier {
public:
  Identifier(const Identifier &other);
  Identifier &operator=(const Identifier &other);
  virtual ~Identifier();

  const std::string &get_name() const;

protected:
  std::string _name;

  Identifier();
  Identifier(const std::string &name);

  void set_name(const std::string &name);
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_IDENTIFIER_HH
