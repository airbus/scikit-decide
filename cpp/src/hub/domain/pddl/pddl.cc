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
        spdlog::error("Unable to create the PDDL domain and problem.");
        // throw std::runtime_error("");//"Unable to create the PDDL domain and problem. Reason: " + std::string(e.what()));
    }
}


Domain& PDDL::get_domain() {
    return _domain;
}