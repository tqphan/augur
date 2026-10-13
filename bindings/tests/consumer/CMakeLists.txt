# SPDX-License-Identifier: GPL-2.0-or-later
#
# A separate project that uses an installed augur SDK through
# find_package(Augur), as an application would. `cargo xtask sdk` installs
# the SDK into a prefix under build/ and builds this project against it,
# once with AUGUR_CPP_SHARED OFF and once ON.

cmake_minimum_required(VERSION 3.22)
project(AugurConsumer LANGUAGES C CXX)

find_package(Augur 0.1 REQUIRED CONFIG)

enable_testing()

set(tests_dir "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(model "${CMAKE_CURRENT_SOURCE_DIR}/../../../testdata/tiny.bin")

add_executable(consumer_c_static "${tests_dir}/c/smoke.c")
target_link_libraries(consumer_c_static PRIVATE Augur::augur_static)
target_compile_definitions(consumer_c_static PRIVATE AUGUR_TEST_MODEL="${model}"
  AUGUR_TEST_STORE="consumer_c_static_user.redb")
add_test(NAME consumer_c_static COMMAND consumer_c_static)

add_executable(consumer_c_shared "${tests_dir}/c/smoke.c")
target_link_libraries(consumer_c_shared PRIVATE Augur::augur_shared)
target_compile_definitions(consumer_c_shared PRIVATE AUGUR_TEST_MODEL="${model}"
  AUGUR_TEST_STORE="consumer_c_shared_user.redb")
add_test(NAME consumer_c_shared COMMAND consumer_c_shared)

add_executable(consumer_cpp "${tests_dir}/cpp/no_exceptions.cpp")
target_link_libraries(consumer_cpp PRIVATE Augur::augur_cpp)
target_compile_definitions(consumer_cpp PRIVATE AUGUR_TEST_MODEL="${model}"
  AUGUR_TEST_STORE="consumer_cpp_user.redb")
add_test(NAME consumer_cpp COMMAND consumer_cpp)

if(WIN32)
  # Windows finds DLLs next to the executable.
  foreach(exe consumer_c_shared consumer_cpp)
    add_custom_command(TARGET ${exe} POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "$<TARGET_FILE:Augur::augur_shared>" "$<TARGET_FILE_DIR:${exe}>")
  endforeach()
endif()

# Check that Augur::augur_cpp links the library AUGUR_CPP_SHARED selects.
get_target_property(cpp_links Augur::augur_cpp INTERFACE_LINK_LIBRARIES)
if(AUGUR_CPP_SHARED)
  set(expected Augur::augur_shared)
else()
  set(expected Augur::augur_static)
endif()
if(NOT cpp_links STREQUAL expected)
  message(FATAL_ERROR "Augur::augur_cpp links ${cpp_links}, expected ${expected}")
endif()
message(STATUS "Augur ${Augur_VERSION}: Augur::augur_cpp links ${cpp_links}")
