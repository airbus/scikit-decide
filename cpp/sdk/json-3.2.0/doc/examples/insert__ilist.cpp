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
    json v = {1, 2, 3, 4};

    // insert range from v2 before the end of array v
    auto new_pos = v.insert(v.end(), {7, 8, 9});

    // output new array and result of insert call
    std::cout << *new_pos << '\n';
    std::cout << v << '\n';
}
