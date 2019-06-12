#ifndef AIRLAPS_PDDL_PARSER_HH
#define AIRLAPS_PDDL_PARSER_HH

#include <string>

namespace airlaps {

    namespace pddl {

        class Parser {
        public :
            /**
             * Parse PDDL files
             * @param domain Domain description file, must also contain the problem definition if the second argument is the empty string
             * @param problem Problem description file, can be empty in which case the problem must be described in the domain description file
             * @return True in case of successful parsing
             */
            void parse(const std::string& domain, const std::string& problem = "");
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_PARSER_HH