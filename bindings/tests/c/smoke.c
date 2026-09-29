/* SPDX-License-Identifier: GPL-2.0-or-later */

#include <stdio.h>
#include <string.h>

#include "augur.h"

static int failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__,         \
                    __LINE__, #cond);                                      \
            failures++;                                                    \
        }                                                                  \
    } while (0)

int main(void) {
    CHECK(augur_abi_version() == AUGUR_ABI_VERSION);

    const char *version = augur_version();
    CHECK(version != NULL && strlen(version) > 0);

    CHECK(strcmp(augur_status_str(AUGUR_STATUS_OK), "ok") == 0);
    CHECK(strcmp(augur_status_str(-1), "unknown status") == 0);

    if (failures == 0) {
        printf("augur %s (ABI %u): ok\n", version, (unsigned)augur_abi_version());
    }
    return failures == 0 ? 0 : 1;
}
