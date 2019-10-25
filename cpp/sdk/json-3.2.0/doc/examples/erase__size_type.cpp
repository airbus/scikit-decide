/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // create a JSON array
    json j_array = {0, 1, 2, 3, 4, 5};

    // call erase
    j_array.erase(2);

    // print values
    std::cout << j_array << '\n';
}
