// SPDX-License-Identifier: GPL-2.0-or-later
//
// Uses the C++ wrapper with AUGUR_NO_EXCEPTIONS (Catch2 needs exceptions,
// so this is a plain program).

#define AUGUR_NO_EXCEPTIONS
#include "augur.hpp"

#include <cstdio>

#ifndef AUGUR_TEST_MODEL
#error "AUGUR_TEST_MODEL must name the test model file"
#endif

static int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, \
                         #cond);                                                 \
            failures++;                                                          \
        }                                                                        \
    } while (0)

int main() {
    augur::Engine engine;
    CHECK(!engine);
    CHECK(augur::Engine::create("en", "no-such-model.bin", engine) == augur::Status::io);
    CHECK(!augur::last_error().empty());
    CHECK(augur::Engine::create("en", AUGUR_TEST_MODEL, engine) == augur::Status::ok);
    CHECK(static_cast<bool>(engine));

    std::vector<augur::Suggestion> out;
    CHECK(engine.try_predict("I live in ", {}, 2, out) == augur::Status::ok);
    CHECK(out.size() == 2 && out[0].text == "London");
    CHECK(engine.try_learn("I love my dog.") == augur::Status::ok);
    CHECK(engine.try_forget("dog") == augur::Status::ok);
    CHECK(engine.try_set_learning(false) == augur::Status::ok);
    CHECK(!engine.learning());
    CHECK(engine.try_clear() == augur::Status::ok);
    CHECK(engine.try_flush() == augur::Status::ok);

    std::remove(AUGUR_TEST_STORE);
    augur::Options options;
    options.user_store = AUGUR_TEST_STORE;
    options.learning = false;
    augur::Engine stored;
    CHECK(augur::Engine::create("en", AUGUR_TEST_MODEL, options, stored) == augur::Status::ok);
    CHECK(!stored.learning());
    augur::Engine locked;
    CHECK(augur::Engine::create("en", AUGUR_TEST_MODEL, options, locked) ==
          augur::Status::user_store);
    stored = augur::Engine{};
    std::remove(AUGUR_TEST_STORE);
    options.weights = augur::Weights{0.0f, 0.0f, 0.0f};
    CHECK(augur::Engine::create("en", AUGUR_TEST_MODEL, options, locked) ==
          augur::Status::invalid_argument);

    const augur::Engine empty;
    CHECK(empty.try_predict("the ", {}, 3, out) == augur::Status::invalid_argument);
    CHECK(out.empty());

    if (failures == 0) {
        std::printf("no-exceptions wrapper: ok\n");
    }
    return failures == 0 ? 0 : 1;
}
