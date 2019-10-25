/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // create JSON values
    json j_object = {{"one", 1}, {"two", 2}};
    json j_array = {1, 2, 4, 8, 16};

    // example for an object
    for (auto& x : json::iterator_wrapper(j_object))
    {
        std::cout << "key: " << x.key() << ", value: " << x.value() << '\n';
    }

    // example for an array
    for (auto& x : json::iterator_wrapper(j_array))
    {
        std::cout << "key: " << x.key() << ", value: " << x.value() << '\n';
    }
}
