/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <backward/backward.hpp>
#include <iostream>
#include <stdexcept>
#include <sstream>

using namespace backward;

class TracedException : public std::runtime_error
{
public:
    TracedException() :
        std::runtime_error(_get_trace())
    {}

private:
    std::string _get_trace()
    {
        std::ostringstream ss;

        StackTrace stackTrace;
        TraceResolver resolver;
        stackTrace.load_here();
        resolver.load_stacktrace(stackTrace);

        for(std::size_t i = 0; i < stackTrace.size(); ++i)
        {
            const ResolvedTrace trace = resolver.resolve(stackTrace[i]);

            ss << "#" << i << " at " << trace.object_function << "\n";
        }

        return ss.str();
    }
};

void f(int i)
{
    if(i >= 42)
    {
        throw TracedException();
    }
    else
    {
        std::cout << "i=" << i << "\n";
        f(i + 1);
    }
}

int main()
{
    try
    {
        f(0);
    } catch (const TracedException& ex)
    {
        std::cout << ex.what();
    }
}


