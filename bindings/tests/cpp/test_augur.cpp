// SPDX-License-Identifier: GPL-2.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "augur.hpp"

TEST_CASE("library ABI matches header") {
    REQUIRE(augur::abi_compatible());
    REQUIRE(augur::abi_version() == augur::header_abi_version);
}

TEST_CASE("version is non-empty") {
    REQUIRE_FALSE(augur::version().empty());
}

TEST_CASE("status strings") {
    CHECK(augur::to_string(augur::Status::ok) == "ok");
    CHECK(augur::to_string(augur::Status::invalid_argument) == "invalid argument");
}

TEST_CASE("Error carries its status") {
    const augur::Error error{augur::Status::io, "cannot open model"};
    CHECK(error.status() == augur::Status::io);
    CHECK(std::string_view{error.what()} == "cannot open model");
}
