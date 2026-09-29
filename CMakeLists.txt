# SPDX-License-Identifier: GPL-2.0-or-later
#
# Builds the augur C/C++ SDK. The Rust library is compiled through Corrosion.
#
# Targets:
#   Augur::augur_static  C API, static library
#   Augur::augur_shared  C API, shared library
#   Augur::augur_cpp     header-only C++17 wrapper (links augur_static)

cmake_minimum_required(VERSION 3.22)
project(Augur VERSION 0.1.0 LANGUAGES C CXX)

option(AUGUR_BUILD_TESTS "Build the C/C++ SDK tests" ${PROJECT_IS_TOP_LEVEL})

include(FetchContent)
FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.6.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(
  MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
  CRATES augur-capi
  CRATE_TYPES staticlib cdylib
)

add_library(augur_headers INTERFACE)
target_include_directories(augur_headers INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/crates/augur-capi/include")

add_library(augur_static INTERFACE)
target_link_libraries(augur_static INTERFACE augur_capi-static augur_headers)
add_library(Augur::augur_static ALIAS augur_static)

add_library(augur_shared INTERFACE)
target_link_libraries(augur_shared INTERFACE augur_capi-shared augur_headers)
add_library(Augur::augur_shared ALIAS augur_shared)

add_library(augur_cpp INTERFACE)
target_include_directories(augur_cpp INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/bindings/cpp/include")
target_compile_features(augur_cpp INTERFACE cxx_std_17)
target_link_libraries(augur_cpp INTERFACE augur_static)
add_library(Augur::augur_cpp ALIAS augur_cpp)

if(AUGUR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(bindings/tests)
endif()
