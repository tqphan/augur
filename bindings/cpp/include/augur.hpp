// SPDX-License-Identifier: GPL-2.0-or-later
//
// augur C++17 API: a header-only wrapper over the C API in augur.h.
//
// Errors are reported as augur::Error exceptions. Define AUGUR_NO_EXCEPTIONS
// before including this header to use status codes instead: the throwing
// constructor and member functions are then left out, and the noexcept
// Engine::create, try_predict and try_learn remain.

#ifndef AUGUR_HPP
#define AUGUR_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifndef AUGUR_NO_EXCEPTIONS
#include <stdexcept>
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
    invalid_model = AUGUR_STATUS_INVALID_MODEL,
};

/// Human-readable description of a status.
inline std::string_view to_string(Status status) noexcept {
    return augur_status_str(static_cast<augur_status>(status));
}

/// Description of the most recent failed call on the calling thread (empty
/// if none). Valid until the next failing call on this thread.
inline std::string_view last_error() noexcept { return augur_last_error(); }

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

/// A predicted word.
struct Suggestion {
    /// The word as it should be displayed or inserted (UTF-8).
    std::string text;
    /// Probability in [0, 1].
    float prob = 0.0f;
};

namespace detail {

struct EngineDeleter {
    void operator()(augur_engine* engine) const noexcept { augur_engine_free(engine); }
};

struct SuggestionsDeleter {
    void operator()(augur_suggestions* list) const noexcept { augur_suggestions_free(list); }
};

#ifndef AUGUR_NO_EXCEPTIONS
inline void check(Status status) {
    if (status != Status::ok) {
        std::string message{last_error()};
        if (message.empty()) {
            message = std::string{to_string(status)};
        }
        throw Error{status, message};
    }
}
#endif

}  // namespace detail

/// A prediction engine. Move-only; const member functions may be called
/// from several threads at once.
class Engine {
public:
    /// An empty engine; assign one from Engine::create before use.
    Engine() noexcept = default;

    /// Creates an engine for `language` (a BCP-47 tag such as "en") from the
    /// n-gram model file at `model_path` (UTF-8), storing it in `out`.
    static Status create(std::string_view language, std::string_view model_path,
                         Engine& out) noexcept {
        augur_engine* handle = nullptr;
        const auto status = static_cast<Status>(augur_engine_new(
            language.data(), language.size(), model_path.data(), model_path.size(), &handle));
        if (status == Status::ok) {
            out.handle_.reset(handle);
        }
        return status;
    }

    /// True if the engine was created.
    explicit operator bool() const noexcept { return handle_ != nullptr; }

    /// The C handle, for use with the C API.
    augur_engine* get() const noexcept { return handle_.get(); }

    /// Predicts up to `max` words for the text around the cursor (`past`
    /// before it, `future` after it), replacing the contents of `out`.
    Status try_predict(std::string_view past, std::string_view future, std::size_t max,
                       std::vector<Suggestion>& out) const {
        out.clear();
        augur_suggestions* raw = nullptr;
        const auto status = static_cast<Status>(augur_predict(
            handle_.get(), past.data(), past.size(), future.data(), future.size(), max, &raw));
        if (status != Status::ok) {
            return status;
        }
        const std::unique_ptr<augur_suggestions, detail::SuggestionsDeleter> list{raw};
        const std::size_t len = augur_suggestions_len(list.get());
        out.reserve(len);
        for (std::size_t i = 0; i < len; ++i) {
            const char* text = nullptr;
            std::size_t text_len = 0;
            float prob = 0.0f;
            augur_suggestions_text(list.get(), i, &text, &text_len);
            augur_suggestions_prob(list.get(), i, &prob);
            out.push_back(Suggestion{std::string{text, text_len}, prob});
        }
        return Status::ok;
    }

    /// Feeds committed text to the engine so that predictors can learn.
    Status try_learn(std::string_view text) const noexcept {
        return static_cast<Status>(augur_learn(handle_.get(), text.data(), text.size()));
    }

#ifndef AUGUR_NO_EXCEPTIONS
    /// Creates an engine; throws augur::Error on failure.
    Engine(std::string_view language, std::string_view model_path) {
        detail::check(create(language, model_path, *this));
    }

    /// Predicts up to `max` words; throws augur::Error on failure.
    std::vector<Suggestion> predict(std::string_view past, std::string_view future = {},
                                    std::size_t max = 5) const {
        std::vector<Suggestion> out;
        detail::check(try_predict(past, future, max, out));
        return out;
    }

    /// Feeds committed text to the engine; throws augur::Error on failure.
    void learn(std::string_view text) const { detail::check(try_learn(text)); }
#endif

private:
    std::unique_ptr<augur_engine, detail::EngineDeleter> handle_;
};

}  // namespace augur

#endif  // AUGUR_HPP
