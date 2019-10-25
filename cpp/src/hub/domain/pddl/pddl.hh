/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_HH
#define AIRLAPS_PDDL_HH

#include "domain.hh"
#include "type.hh"
#include "object.hh"
#include "variable.hh"
#include "term.hh"
#include "predicate.hh"
#include "function.hh"
#include "derived_predicate.hh"
#include "class.hh"
#include "operator.hh"
#include "constraint_formula.hh"
#include "preference.hh"
#include "predicate_formula.hh"
#include "quantified_formula.hh"
#include "aggregation_formula.hh"
#include "imply_formula.hh"
#include "negation_formula.hh"
#include "timed_formula.hh"
#include "duration_formula.hh"
#include "comparison_formula.hh"
#include "operation_expression.hh"
#include "minus_expression.hh"
#include "numerical_expression.hh"
#include "function_expression.hh"
#include "predicate_effect.hh"
#include "aggregation_effect.hh"
#include "quantified_effect.hh"
#include "conditional_effect.hh"
#include "negation_effect.hh"
#include "timed_effect.hh"
#include "duration_effect.hh"
#include "function_effect.hh"
#include "assignment_effect.hh"

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
