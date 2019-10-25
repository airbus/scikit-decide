/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// X11-DisableStringification.cpp
// Test that stringification of original expression can be disabled
// this is a workaround for VS 2017 issue with Raw String literal
// and preprocessor token pasting. In other words, hopefully this test
// will be deleted soon :-)

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

struct Hidden {};

bool operator==(Hidden, Hidden) { return true; }

TEST_CASE("DisableStringification") {
    REQUIRE( Hidden{} == Hidden{} );
}
