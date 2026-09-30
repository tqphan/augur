# SPDX-License-Identifier: GPL-2.0-or-later

# Catch2 must be built with the same standard as the tests; below C++17 it
# omits its std::string_view support.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.16.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)
include(Catch)

# The small English model in testdata/ (see testdata/README.md).
set(AUGUR_TEST_MODEL "${PROJECT_SOURCE_DIR}/testdata/tiny.bin")

# C smoke test, linked once statically and once against the shared library.
add_executable(augur_c_static c/smoke.c)
target_link_libraries(augur_c_static PRIVATE Augur::augur_static)
target_compile_definitions(augur_c_static PRIVATE AUGUR_TEST_MODEL="${AUGUR_TEST_MODEL}")
add_test(NAME c_smoke_static COMMAND augur_c_static)

add_executable(augur_c_shared c/smoke.c)
target_link_libraries(augur_c_shared PRIVATE Augur::augur_shared)
target_compile_definitions(augur_c_shared PRIVATE AUGUR_TEST_MODEL="${AUGUR_TEST_MODEL}")
add_test(NAME c_smoke_shared COMMAND augur_c_shared)
if(WIN32)
  # Put augur_capi.dll next to the test executable.
  add_custom_command(TARGET augur_c_shared POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:augur_capi-shared>" "$<TARGET_FILE_DIR:augur_c_shared>")
endif()

# C++ wrapper tests.
add_executable(augur_cpp_tests cpp/test_augur.cpp)
target_link_libraries(augur_cpp_tests PRIVATE Augur::augur_cpp Catch2::Catch2WithMain)
target_compile_definitions(augur_cpp_tests PRIVATE AUGUR_TEST_MODEL="${AUGUR_TEST_MODEL}")
catch_discover_tests(augur_cpp_tests TEST_PREFIX "cpp: ")

# C++ wrapper without exceptions.
add_executable(augur_cpp_no_exceptions cpp/no_exceptions.cpp)
target_link_libraries(augur_cpp_no_exceptions PRIVATE Augur::augur_cpp)
target_compile_definitions(augur_cpp_no_exceptions PRIVATE AUGUR_TEST_MODEL="${AUGUR_TEST_MODEL}")
add_test(NAME cpp_no_exceptions COMMAND augur_cpp_no_exceptions)
