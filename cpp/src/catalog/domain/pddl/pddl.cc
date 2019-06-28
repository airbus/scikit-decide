#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "pddl.hh"
#include "parser.hh"

using namespace airlaps::pddl;

PDDL::PDDL(const std::string& domain_file, const std::string& problem_file, bool debug_logs) {
    try {
        Parser parser;
        parser.parse(domain_file, problem_file, _domain, debug_logs);
    } catch (const std::exception& e) {
        spdlog::error("Unable to create the PDDL domain and problem");
    }
}


Domain& PDDL::get_domain() {
    return _domain;
}
