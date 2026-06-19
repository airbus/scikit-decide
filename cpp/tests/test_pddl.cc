/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <catch.hpp>
#include <filesystem>

#include "hub/domain/pddl/pddl.hh"
#include "config.h"

TEST_CASE("PDDL", "[pddl]") {
  for (auto &p : std::filesystem::directory_iterator(
           std::string(SKDECIDE_SOURCE_DIR) + "/tests/data/pddl")) {
    for (auto &pp : std::filesystem::directory_iterator(p.path() / "domains")) {
      // skip bugged domains
      if (pp.path().filename().string() ==
              "mystery-prime-round-1-adl" || // we don't handle :vars
          pp.path().filename().string() ==
              "logistics-round-1-adl" || // we don't handle :domain-axioms
          pp.path().filename().string() ==
              "mystery-round-1-adl" || // we don't handle 'in-package'
          pp.path().filename().string() ==
              "elevator-temporal-satisficing" || // not using numeric fluents
                                                 // requirement
          pp.path().filename().string() ==
              "storage-temporal-satisficing") { // declares type 'area' twice
        continue;
      }

      if (std::filesystem::exists(pp.path() / "domains") &&
          std::filesystem::is_directory(pp.path() / "domains")) {
        // one domain for one instance
        for (auto &domain :
             std::filesystem::directory_iterator(pp.path() / "domains")) {
          std::string instance_number =
              domain.path().filename().string().substr(
                  domain.path().filename().string().find_first_of('-'));
          auto problem = pp.path() / "instances" /
                         (std::string("instance") + instance_number);
          skdecide::pddl::PDDL pddl;
          REQUIRE_NOTHROW(
              pddl.load({domain.path().string(), problem.string()}));
        }
      } else if (std::filesystem::exists(pp.path() / "domain.pddl")) {
        // one domain for many instances
        skdecide::pddl::PDDL pddl;

        auto domain = pp.path() / "domain.pddl";
        REQUIRE_NOTHROW(pddl.load({domain.string()}));

        for (auto &ppp :
             std::filesystem::directory_iterator(pp.path() / "instances")) {
          if ((pp.path().filename().string() ==
                   "woodworking-sequential-multi-core" ||
               pp.path().filename().string() ==
                   "woodworking-sequential-satisficing") &&
              ppp.path().filename().string() ==
                  "instance-10.pddl") { // bugged empty object declaration on
                                        // Line 25
            continue;
          }
          REQUIRE_NOTHROW(pddl.load({ppp.path().string()}));
        }
      }
    }
  }
}

TEST_CASE("PDDL+", "[pddl+]") {
  for (auto &p : std::filesystem::directory_iterator(
           std::string(SKDECIDE_SOURCE_DIR) + "/tests/data/pddl+")) {
    std::list<std::string> domains;
    std::list<std::string> problems;
    for (auto &pp : std::filesystem::directory_iterator(p.path())) {
      if (pp.path().filename().string().find("problem") != std::string::npos) {
        problems.push_back(pp.path().string());
      } else {
        domains.push_back(pp.path().string());
      }
    }
    // problems must be parsed after domains
    domains.insert(domains.end(), problems.begin(), problems.end());
    skdecide::pddl::PDDL pddl;
    REQUIRE_NOTHROW(pddl.load(domains));
  }
}
