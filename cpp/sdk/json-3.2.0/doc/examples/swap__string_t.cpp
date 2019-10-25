/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // create a JSON value
    json value = { "the good", "the bad", "the ugly" };

    // create string_t
    json::string_t string = "the fast";

    // swap the object stored in the JSON value
    value[1].swap(string);

    // output the values
    std::cout << "value = " << value << '\n';
    std::cout << "string = " << string << '\n';
}
