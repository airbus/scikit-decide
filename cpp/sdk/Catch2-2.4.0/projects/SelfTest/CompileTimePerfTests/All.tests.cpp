/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// include set of usage tests into one file for compiler performance test purposes
// This whole file can now be included multiple times in 10.tests.cpp, and *that*
// file included multiple times (in 100.tests.cpp)

// Note that the intention is only for these files to be compiled. They will
// fail at runtime due to the re-user of test case names

#include "../UsageTests/Approx.tests.cpp"
#include "../UsageTests/BDD.tests.cpp"
#include "../UsageTests/Class.tests.cpp"
#include "../UsageTests/Compilation.tests.cpp"
#include "../UsageTests/Condition.tests.cpp"
#include "../UsageTests/Exception.tests.cpp"
#include "../UsageTests/Matchers.tests.cpp"
#include "../UsageTests/Misc.tests.cpp"
