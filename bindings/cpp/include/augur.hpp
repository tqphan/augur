// SPDX-License-Identifier: GPL-2.0-or-later
//
// augur C++17 API: a header-only wrapper over the C API in augur.h.
//
// Errors are reported as augur::Error exceptions. Define AUGUR_NO_EXCEPTIONS
// before including this header to use status codes instead.

#ifndef AUGUR_HPP
#define AUGUR_HPP

#include <cstdint>
#include <string_view>

#ifndef AUGUR_NO_EXCEPTIONS
#include <stdexcept>
#include <string>
#endif

#include "augur.h"

namespace augur {

/// ABI version the calling code was compiled against.
inline constexpr std::uint32_t header_abi_version = AUGUR_ABI_VERSION;

/// ABI version of the loaded library.
inline std::uint32_t abi_version() noexcept { return augur_abi_version(); }

/// True if the loaded library matches the header this code was compiled with.
inline bool abi_compatible() noexcept { return abi_version() == header_abi_version; }

/// Library version, e.g. "0.1.0".
inline std::string_view version() noexcept { return augur_version(); }

/// Result code returned by fallible operations.
enum class Status : augur_status {
    ok = AUGUR_STATUS_OK,
    invalid_argument = AUGUR_STATUS_INVALID_ARGUMENT,
    io = AUGUR_STATUS_IO,
    internal = AUGUR_STATUS_INTERNAL,
};

/// Human-readable description of a status.
inline std::string_view to_string(Status status) noexcept {
    return augur_status_str(static_cast<augur_status>(status));
}

#ifndef AUGUR_NO_EXCEPTIONS
/// Exception thrown when a C API call fails.
class Error : public std::runtime_error {
public:
    Error(Status status, const std::string& message)
        : std::runtime_error(message), status_(status) {}

    Status status() const noexcept { return status_; }

private:
    Status status_;
};
#endif

}  // namespace augur

#endif  // AUGUR_HPP
