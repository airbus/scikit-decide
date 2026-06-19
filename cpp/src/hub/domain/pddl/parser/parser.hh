/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_PARSER_HH
#define SKDECIDE_PDDL_PARSER_HH

#include <string>
#include <list>

#include "domain.hh"
#include "problem.hh"

namespace skdecide {

namespace pddl {

class Parser {
public:
  /**
   * Parse PDDL files
   * @param files List of files containing domain and problem descriptions
   * @param domains List of parsed domain objects
   * @param problems List of parsed problem objects
   * @param verbose Activates parsing traces
   */
  void parse(const std::list<std::string> &files,
             std::list<Domain::Ptr> &domains, std::list<Problem::Ptr> &problems,
             bool verbose = false);
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSER_HH
