/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // create two JSON objects
    json j1 = {{"one", "eins"}, {"two", "zwei"}};
    json j2 = {{"eleven", "elf"}, {"seventeen", "siebzehn"}};

    // output objects
    std::cout << j1 << '\n';
    std::cout << j2 << '\n';

    // insert range from j2 to j1
    j1.insert(j2.begin(), j2.end());

    // output result of insert call
    std::cout << j1 << '\n';
}
