/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // implicitly create a JSON null value
    json j1;

    // explicitly create a JSON null value
    json j2(nullptr);

    // serialize the JSON null value
    std::cout << j1 << '\n' << j2 << '\n';
}
