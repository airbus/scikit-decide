/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "pybind11/pybind11.h"

#include "pddl.hh"
#include "parser.hh"

using namespace airlaps::pddl;

void PDDL::load(const std::string& domain_file, const std::string& problem_file, bool debug_logs) {
    try {
        Parser parser;
        parser.parse(domain_file, problem_file, _domain, debug_logs);
    } catch (const std::exception& e) {
        spdlog::error("Unable to create the PDDL domain and problem. Reason: " + std::string(e.what()));
        throw;
    }
}


Domain& PDDL::get_domain() {
    return _domain;
}
