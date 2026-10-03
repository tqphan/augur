// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdio>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "augur.hpp"

#ifndef AUGUR_TEST_MODEL
#error "AUGUR_TEST_MODEL must name the test model file"
#endif

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
    CHECK(augur::to_string(augur::Status::invalid_model) == "invalid model");
}

TEST_CASE("Error carries its status") {
    const augur::Error error{augur::Status::io, "cannot open model"};
    CHECK(error.status() == augur::Status::io);
    CHECK(std::string_view{error.what()} == "cannot open model");
}

TEST_CASE("predicts words") {
    const augur::Engine engine{"en", AUGUR_TEST_MODEL};
    REQUIRE(engine);

    auto out = engine.predict("I live in ", {}, 2);
    REQUIRE(out.size() == 2);
    CHECK(out[0].text == "London");
    CHECK(out[1].text == "Paris");
    CHECK(out[0].prob >= out[1].prob);

    out = engine.predict("I live in ", " London", 1);
    REQUIRE(out.size() == 1);
    CHECK(out[0].text == "Paris");

    out = engine.predict("I live in Lon");
    REQUIRE(out.size() == 1);
    CHECK(out[0].text == "London");

    CHECK(engine.predict("the ", {}, 0).empty());
    CHECK_NOTHROW(engine.learn("I love my dog."));
}

TEST_CASE("an engine can be shared between threads") {
    const augur::Engine engine{"en", AUGUR_TEST_MODEL};
    std::vector<augur::Suggestion> a, b;
    std::thread other{[&] { a = engine.predict("the cat "); }};
    b = engine.predict("the cat ");
    other.join();
    REQUIRE_FALSE(a.empty());
    CHECK(a.size() == b.size());
    CHECK(a[0].text == b[0].text);
}

TEST_CASE("failures throw augur::Error with the library's message") {
    try {
        augur::Engine engine{"en", "no-such-model.bin"};
        FAIL("expected an exception");
    } catch (const augur::Error& e) {
        CHECK(e.status() == augur::Status::io);
        CHECK_FALSE(std::string_view{e.what()}.empty());
    }
    try {
        augur::Engine engine{"xx", AUGUR_TEST_MODEL};
        FAIL("expected an exception");
    } catch (const augur::Error& e) {
        CHECK(e.status() == augur::Status::invalid_argument);
        CHECK(std::string_view{e.what()}.find("unsupported language") != std::string_view::npos);
    }
    const augur::Engine empty;
    CHECK_FALSE(empty);
    CHECK_THROWS_AS(empty.predict("the "), augur::Error);
}

TEST_CASE("learns from the user and saves to a store") {
    std::remove(AUGUR_TEST_STORE);
    augur::Options options;
    options.user_store = AUGUR_TEST_STORE;
    options.weights = augur::Weights{0.6f, 0.3f, 0.1f};
    options.user_limits = augur::UserLimits{};
    options.recency = augur::Recency{};
    {
        const augur::Engine engine{"en", AUGUR_TEST_MODEL, options};
        CHECK(engine.learning());
        for (const char* word : {"I", "like", "Zanzibar", "a", "lot."}) {
            engine.learn(word);
        }
        CHECK(engine.predict("I like ", {}, 1).at(0).text == "Zanzibar");
        try {
            augur::Engine second{"en", AUGUR_TEST_MODEL, options};
            FAIL("expected the store to be locked");
        } catch (const augur::Error& e) {
            CHECK(e.status() == augur::Status::user_store);
        }
    }
    const augur::Engine engine{"en", AUGUR_TEST_MODEL, options};
    CHECK(engine.predict("I like ", {}, 1).at(0).text == "Zanzibar");
    engine.forget("Zanzibar");
    CHECK(engine.predict("I like ", {}, 1).at(0).text != "Zanzibar");

    engine.set_learning(false);
    CHECK_FALSE(engine.learning());
    engine.learn("I like Zanzibar.");
    CHECK(engine.predict("I like ", {}, 1).at(0).text != "Zanzibar");
    engine.set_learning(true);
    engine.reset_learning_context();
    engine.clear();
    engine.flush();
}

TEST_CASE("invalid options throw") {
    augur::Options options;
    options.weights = augur::Weights{-1.0f, 0.0f, 0.0f};
    try {
        augur::Engine engine{"en", AUGUR_TEST_MODEL, options};
        FAIL("expected an exception");
    } catch (const augur::Error& e) {
        CHECK(e.status() == augur::Status::invalid_argument);
    }
}
