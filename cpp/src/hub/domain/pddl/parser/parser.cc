/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <exception>
#include <list>

#include "pegtl.hpp"
#include "spdlog/spdlog.h"

#include "parser.hh"
#include "parser_pass.hh"

namespace pegtl = tao::pegtl; // NOLINT

namespace skdecide {

namespace pddl {

void Parser::parse(const std::list<std::string> &files,
                   std::list<Domain::Ptr> &domains,
                   std::list<Problem::Ptr> &problems, bool verbose) {
  if (verbose) {
    spdlog::set_level(spdlog::level::debug);
  } else {
    spdlog::set_level(spdlog::level::info);
  }

  parser::state s;

  for (auto &d : domains) {
    s.domains.insert(std::make_pair(d->get_name(), d));
  }

  for (const std::string &f : files) {
    spdlog::info("Parsing " + f);

    auto run_pass = [&](auto grammar_tag) {
      pegtl::text_file_input in(f);
      if (verbose) {
        pegtl::trace_state t;
        parser::ParsePass<decltype(grammar_tag), parser::TracerControlTag>::run(
            in, &t, s);
      } else {
        parser::ParsePass<decltype(grammar_tag), parser::NormalControlTag>::run(
            in, nullptr, s);
      }
    };

    try {
      run_pass(parser::DomainStructureTag{});
      run_pass(parser::DomainOperatorsTag{});
      run_pass(parser::ProblemTag{});
    } catch (const pegtl::parse_error_base &e) {
      throw std::runtime_error("SKDECIDE parsing exception: " +
                               std::string(e.what()));
    } catch (const std::system_error &e) {
      throw std::runtime_error("SKDECIDE exception: error when reading " + f +
                               ": " + e.what());
    } catch (const std::exception &e) {
      throw std::runtime_error("SKDECIDE exception: error when parsing " + f +
                               ": " + e.what());
    }
  }

  for (const auto &d : s.domains) {
    domains.push_back(d.second);
  }

  for (const auto &p : s.problems) {
    problems.push_back(p.second);
  }
}

} // namespace pddl

} // namespace skdecide
