/* SPDX-License-Identifier: GPL-2.0-or-later */

#include <stdio.h>
#include <string.h>

#include "augur.h"

#ifndef AUGUR_TEST_MODEL
#error "AUGUR_TEST_MODEL must name the test model file"
#endif

static int failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__,         \
                    __LINE__, #cond);                                      \
            failures++;                                                    \
        }                                                                  \
    } while (0)

static void test_predict(void) {
    const char *lang = "en";
    const char *path = AUGUR_TEST_MODEL;
    augur_engine *engine = NULL;
    augur_status status =
        augur_engine_new(lang, strlen(lang), path, strlen(path), &engine);
    CHECK(status == AUGUR_STATUS_OK);
    if (status != AUGUR_STATUS_OK) {
        fprintf(stderr, "augur_engine_new: %s\n", augur_last_error());
        return;
    }

    const char *past = "I live in ";
    augur_suggestions *list = NULL;
    CHECK(augur_predict(engine, past, strlen(past), NULL, 0, 2, &list) ==
          AUGUR_STATUS_OK);
    CHECK(augur_suggestions_len(list) == 2);

    const char *text = NULL;
    size_t len = 0;
    float prob = 0.0f;
    CHECK(augur_suggestions_text(list, 0, &text, &len) == AUGUR_STATUS_OK);
    CHECK(len == 6 && memcmp(text, "London", 6) == 0);
    CHECK(strcmp(text, "London") == 0); /* also NUL-terminated */
    CHECK(augur_suggestions_prob(list, 0, &prob) == AUGUR_STATUS_OK);
    CHECK(prob > 0.0f && prob <= 1.0f);
    CHECK(augur_suggestions_text(list, 5, &text, &len) ==
          AUGUR_STATUS_INVALID_ARGUMENT);
    augur_suggestions_free(list);

    const char *committed = "I love my dog.";
    CHECK(augur_learn(engine, committed, strlen(committed)) == AUGUR_STATUS_OK);
    augur_engine_free(engine);
}

static int top_is(augur_engine *engine, const char *past, const char *word) {
    augur_suggestions *list = NULL;
    const char *text = NULL;
    size_t len = 0;
    int ok = augur_predict(engine, past, strlen(past), NULL, 0, 1, &list) ==
                 AUGUR_STATUS_OK &&
             augur_suggestions_len(list) == 1 &&
             augur_suggestions_text(list, 0, &text, &len) == AUGUR_STATUS_OK &&
             strcmp(text, word) == 0;
    augur_suggestions_free(list);
    return ok;
}

static void test_user_model(void) {
    const char *store = AUGUR_TEST_STORE;
    const char *path = AUGUR_TEST_MODEL;
    remove(store);

    augur_options *options = NULL;
    CHECK(augur_options_new(&options) == AUGUR_STATUS_OK);
    CHECK(augur_options_set_user_store(options, store, strlen(store)) ==
          AUGUR_STATUS_OK);
    CHECK(augur_options_set_weights(options, 0.6f, 0.3f, 0.1f) == AUGUR_STATUS_OK);
    CHECK(augur_options_set_weights(options, -1.0f, 0.0f, 0.0f) ==
          AUGUR_STATUS_INVALID_ARGUMENT);
    CHECK(augur_options_set_user_limits(options, 1000, 0, 256) == AUGUR_STATUS_OK);
    CHECK(augur_options_set_recency(options, 100, 10.0f) == AUGUR_STATUS_OK);
    CHECK(augur_options_set_learning(options, true) == AUGUR_STATUS_OK);

    augur_engine *engine = NULL;
    CHECK(augur_engine_new_with_options("en", 2, path, strlen(path), options,
                                        &engine) == AUGUR_STATUS_OK);
    if (engine == NULL) {
        fprintf(stderr, "augur_engine_new_with_options: %s\n", augur_last_error());
        augur_options_free(options);
        return;
    }
    CHECK(augur_is_learning(engine));
    /* Word by word, as a keyboard would commit them. */
    const char *words[] = {"I", "like", "Zanzibar", "a", "lot."};
    for (size_t i = 0; i < sizeof words / sizeof *words; i++) {
        CHECK(augur_learn(engine, words[i], strlen(words[i])) == AUGUR_STATUS_OK);
    }
    CHECK(top_is(engine, "I like ", "Zanzibar"));

    /* The store is locked while the engine is open. */
    augur_engine *second = NULL;
    CHECK(augur_engine_new_with_options("en", 2, path, strlen(path), options,
                                        &second) == AUGUR_STATUS_USER_STORE);
    CHECK(augur_flush(engine) == AUGUR_STATUS_OK);
    augur_engine_free(engine);

    /* Reopened: still learned. */
    engine = NULL;
    CHECK(augur_engine_new_with_options("en", 2, path, strlen(path), options,
                                        &engine) == AUGUR_STATUS_OK);
    CHECK(top_is(engine, "I like ", "Zanzibar"));
    CHECK(augur_forget(engine, "zanzibar", 8) == AUGUR_STATUS_OK);
    CHECK(!top_is(engine, "I like ", "Zanzibar"));

    CHECK(augur_set_learning(engine, false) == AUGUR_STATUS_OK);
    CHECK(!augur_is_learning(engine));
    CHECK(augur_learn(engine, "I like Zanzibar.", 16) == AUGUR_STATUS_OK);
    CHECK(!top_is(engine, "I like ", "Zanzibar"));
    CHECK(augur_set_learning(engine, true) == AUGUR_STATUS_OK);
    CHECK(augur_reset_learning_context(engine) == AUGUR_STATUS_OK);
    CHECK(augur_clear(engine) == AUGUR_STATUS_OK);
    augur_engine_free(engine);
    augur_options_free(options);
    remove(store);
}

static void test_errors(void) {
    augur_engine *engine = NULL;
    const char *missing = "no-such-model.bin";
    CHECK(augur_engine_new("en", 2, missing, strlen(missing), &engine) ==
          AUGUR_STATUS_IO);
    CHECK(engine == NULL);
    CHECK(strlen(augur_last_error()) > 0);

    const char *path = AUGUR_TEST_MODEL;
    CHECK(augur_engine_new("xx", 2, path, strlen(path), &engine) ==
          AUGUR_STATUS_INVALID_ARGUMENT);
    CHECK(strstr(augur_last_error(), "unsupported language") != NULL);

    augur_engine_free(NULL);
    augur_suggestions_free(NULL);
}

int main(void) {
    CHECK(augur_abi_version() == AUGUR_ABI_VERSION);

    const char *version = augur_version();
    CHECK(version != NULL && strlen(version) > 0);

    CHECK(strcmp(augur_status_str(AUGUR_STATUS_OK), "ok") == 0);
    CHECK(strcmp(augur_status_str(AUGUR_STATUS_INVALID_MODEL), "invalid model") == 0);
    CHECK(strcmp(augur_status_str(AUGUR_STATUS_USER_STORE), "user store error") == 0);
    CHECK(strcmp(augur_status_str(-1), "unknown status") == 0);

    test_predict();
    test_user_model();
    test_errors();

    if (failures == 0) {
        printf("augur %s (ABI %u): ok\n", version, (unsigned)augur_abi_version());
    }
    return failures == 0 ? 0 : 1;
}
