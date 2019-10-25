/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // a JSON text given as std::vector
    std::vector<uint8_t> text = {'[', '1', ',', '2', ',', '3', ']', '\0'};

    // parse and serialize JSON
    json j_complete = json::parse(text.begin(), text.end());
    std::cout << std::setw(4) << j_complete << "\n\n";
}
