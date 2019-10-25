/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * This content is released under the MIT License as specified in https://raw.githubusercontent.com/gabime/spdlog/master/LICENSE
 */
#include "includes.h"

#include <iostream>

class failing_sink : public spdlog::sinks::base_sink<std::mutex>
{
public:
    failing_sink() = default;
    ~failing_sink() final = default;

protected:
    void sink_it_(const spdlog::details::log_msg &) final
    {
        throw std::runtime_error("some error happened during log");
    }

    void flush_() final
    {
        throw std::runtime_error("some error happened during flush");
    }
};

TEST_CASE("default_error_handler", "[errors]]")
{
    prepare_logdir();
    std::string filename = "logs/simple_log.txt";

    auto logger = spdlog::create<spdlog::sinks::basic_file_sink_mt>("test-error", filename, true);
    logger->set_pattern("%v");
    logger->info("Test message {} {}", 1);
    logger->info("Test message {}", 2);
    logger->flush();

    REQUIRE(file_contents(filename) == std::string("Test message 2\n"));
    REQUIRE(count_lines(filename) == 1);
}

struct custom_ex
{
};
TEST_CASE("custom_error_handler", "[errors]]")
{
    prepare_logdir();
    std::string filename = "logs/simple_log.txt";
    auto logger = spdlog::create<spdlog::sinks::basic_file_sink_mt>("logger", filename, true);
    logger->flush_on(spdlog::level::info);
    logger->set_error_handler([=](const std::string &) { throw custom_ex(); });
    logger->info("Good message #1");

    REQUIRE_THROWS_AS(logger->info("Bad format msg {} {}", "xxx"), custom_ex);
    logger->info("Good message #2");
    REQUIRE(count_lines(filename) == 2);
}

TEST_CASE("default_error_handler2", "[errors]]")
{
    spdlog::drop_all();
    auto logger = spdlog::create<failing_sink>("failed_logger");
    logger->set_error_handler([=](const std::string &) { throw custom_ex(); });
    REQUIRE_THROWS_AS(logger->info("Some message"), custom_ex);
}

TEST_CASE("flush_error_handler", "[errors]]")
{
    spdlog::drop_all();
    auto logger = spdlog::create<failing_sink>("failed_logger");
    logger->set_error_handler([=](const std::string &) { throw custom_ex(); });
    REQUIRE_THROWS_AS(logger->flush(), custom_ex);
}

TEST_CASE("async_error_handler", "[errors]]")
{
    prepare_logdir();
    std::string err_msg("log failed with some msg");

    std::string filename = "logs/simple_async_log.txt";
    {
        spdlog::init_thread_pool(128, 1);
        auto logger = spdlog::create_async<spdlog::sinks::basic_file_sink_mt>("logger", filename, true);
        logger->set_error_handler([=](const std::string &) {
            std::ofstream ofs("logs/custom_err.txt");
            if (!ofs)
            {
                throw std::runtime_error("Failed open logs/custom_err.txt");
            }
            ofs << err_msg;
        });
        logger->info("Good message #1");
        logger->info("Bad format msg {} {}", "xxx");
        logger->info("Good message #2");
        spdlog::drop("logger"); // force logger to drain the queue and shutdown
    }
    spdlog::init_thread_pool(128, 1);
    REQUIRE(count_lines(filename) == 2);
    REQUIRE(file_contents("logs/custom_err.txt") == err_msg);
}

// Make sure async error handler is executed
TEST_CASE("async_error_handler2", "[errors]]")
{
    prepare_logdir();
    std::string err_msg("This is async handler error message");
    {
        spdlog::init_thread_pool(128, 1);
        auto logger = spdlog::create_async<failing_sink>("failed_logger");
        logger->set_error_handler([=](const std::string &) {
            std::ofstream ofs("logs/custom_err2.txt");
            if (!ofs)
                throw std::runtime_error("Failed open logs/custom_err2.txt");
            ofs << err_msg;
        });
        logger->info("Hello failure");
        spdlog::drop("failed_logger"); // force logger to drain the queue and shutdown
    }

    spdlog::init_thread_pool(128, 1);
    REQUIRE(file_contents("logs/custom_err2.txt") == err_msg);
}
