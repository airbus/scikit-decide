/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "pybind11/pybind11.h"

#include "pddl.hh"
#include "parser/parser.hh"

using namespace skdecide::pddl;

void PDDL::load(const std::list<std::string> &files, bool verbose) {
  try {
    Parser parser;
    parser.parse(files, _domains, _problems, verbose);
  } catch (const std::exception &e) {
    spdlog::error("Unable to create the PDDL domains and problems. Reason: " +
                  std::string(e.what()));
    throw;
  }
}

const std::list<Domain::Ptr> &PDDL::get_domains() { return _domains; }

const std::list<Problem::Ptr> &PDDL::get_problems() { return _problems; }
