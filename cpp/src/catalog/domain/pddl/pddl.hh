#ifndef AIRLAPS_PDDL_HH
#define AIRLAPS_PDDL_HH

#include "domain.hh"

namespace airlaps {

    namespace pddl {

        class PDDL {
        public :
            /**
             * Constructs a PDDL object (domain and problem) from PDDL files
             * @param domain_file Domain description file, must also contain the problem definition if the second argument is the empty string
             * @param problem_file Problem description file, can be empty in which case the problem must be described in the domain description file
             * @param debug_logs Activates parsing traces
             * @return True in case of successful parsing
             */
            void load(const std::string& domain_file, const std::string& problem_file = "", bool debug_logs = false);
            
            Domain& get_domain();
            
        private :
            Domain _domain;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_HH
