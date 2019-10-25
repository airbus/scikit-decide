/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_PARSER_HH
#define AIRLAPS_PDDL_PARSER_HH

#include <string>
#include "domain.hh"

namespace airlaps {

    namespace pddl {

        class Parser {
        public :
            /**
             * Parse PDDL files
             * @param domain_file Domain description file, must also contain the problem definition if the second argument is the empty string
             * @param problem_file Problem description file, can be empty in which case the problem must be described in the domain description file
             * @param domain Domain object containing parsed data
             * @param debug_logs Activates parsing traces
             * @return True in case of successful parsing
             */
            void parse(const std::string& domain_file, const std::string& problem_file,
                       Domain& domain,
                       bool debug_logs = false);
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_PARSER_HH