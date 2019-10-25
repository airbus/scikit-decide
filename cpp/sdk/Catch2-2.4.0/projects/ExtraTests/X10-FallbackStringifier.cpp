/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// X10-FallbackStringifier.cpp
// Test that defining fallbackStringifier compiles

#include <string>

// A catch-all stringifier
template <typename T>
std::string fallbackStringifier(T const&) {
    return "{ !!! }";
}

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

struct foo {
    explicit operator bool() const {
        return true;
    }
};

TEST_CASE("aa") {
    REQUIRE(foo{});
}
