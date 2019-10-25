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
    json value = { {"translation", {{"one", "eins"}, {"two", "zwei"}}} };

    // create an object_t
    json::object_t object = {{"cow", "Kuh"}, {"dog", "Hund"}};

    // swap the object stored in the JSON value
    value["translation"].swap(object);

    // output the values
    std::cout << "value = " << value << '\n';
    std::cout << "object = " << object << '\n';
}
