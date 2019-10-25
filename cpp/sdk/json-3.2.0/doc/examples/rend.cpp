/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main()
{
    // create an array value
    json array = {1, 2, 3, 4, 5};

    // get an iterator to the reverse-end
    json::reverse_iterator it = array.rend();

    // increment the iterator to point to the first element
    --it;

    // serialize the element that the iterator points to
    std::cout << *it << '\n';
}
