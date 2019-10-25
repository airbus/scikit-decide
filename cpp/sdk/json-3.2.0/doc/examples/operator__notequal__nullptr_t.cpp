/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // create several JSON values
    json array = {1, 2, 3};
    json object = {{"A", "a"}, {"B", "b"}};
    json number = 17;
    json string = "foo";
    json null;

    // output values and comparisons
    std::cout << std::boolalpha;
    std::cout << array << " != nullptr " << (array != nullptr) << '\n';
    std::cout << object << " != nullptr " << (object != nullptr) << '\n';
    std::cout << number << " != nullptr " << (number != nullptr) << '\n';
    std::cout << string << " != nullptr " << (string != nullptr) << '\n';
    std::cout << null << " != nullptr " << (null != nullptr) << '\n';
}
