/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // create a JSON object
    json j_object = {{"one", 1}, {"two", 2}};

    // call erase
    auto count_one = j_object.erase("one");
    auto count_three = j_object.erase("three");

    // print values
    std::cout << j_object << '\n';
    std::cout << count_one << " " << count_three << '\n';
}
