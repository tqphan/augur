# SPDX-License-Identifier: GPL-2.0-or-later
#
# Builds the augur C/C++ SDK. The Rust library is compiled through Corrosion.
#
# Targets:
#   Augur::augur_static  C API, static library
#   Augur::augur_shared  C API, shared library
#   Augur::augur_cpp     header-only C++17 wrapper (links augur_static, or
#                        augur_shared if AUGUR_CPP_SHARED is ON)
#
# `cmake --install` installs the libraries, headers and a CMake package, so
# that other projects can use `find_package(Augur)` without Rust; see
# cmake/AugurConfig.cmake.in.

cmake_minimum_required(VERSION 3.22)
project(Augur VERSION 0.1.0 LANGUAGES C CXX)

option(AUGUR_BUILD_TESTS "Build the C/C++ SDK tests" ${PROJECT_IS_TOP_LEVEL})
option(AUGUR_INSTALL "Generate install rules" ${PROJECT_IS_TOP_LEVEL})
option(AUGUR_CPP_SHARED "Link Augur::augur_cpp against the shared library" OFF)

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
if(AUGUR_CPP_SHARED)
  target_link_libraries(augur_cpp INTERFACE augur_shared)
else()
  target_link_libraries(augur_cpp INTERFACE augur_static)
endif()
add_library(Augur::augur_cpp ALIAS augur_cpp)

if(AUGUR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(bindings/tests)
endif()

if(AUGUR_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # Corrosion's targets are imported, so install their files directly.
  install(FILES
    "crates/augur-capi/include/augur.h"
    "bindings/cpp/include/augur.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  install(FILES "$<TARGET_FILE:augur_capi-static>"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  if(WIN32)
    install(FILES "$<TARGET_FILE:augur_capi-shared>"
      DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(FILES "$<TARGET_LINKER_FILE:augur_capi-shared>"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  else()
    install(FILES "$<TARGET_FILE:augur_capi-shared>"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()
  install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/augur")

  # Values substituted into AugurConfig.cmake.
  get_target_property(AUGUR_STATIC_FILE augur_capi-static COR_FILE_NAME)
  get_target_property(AUGUR_SHARED_FILE augur_capi-shared COR_FILE_NAME)
  get_target_property(AUGUR_IMPLIB_FILE augur_capi-shared COR_IMPLIB_FILE_NAME)
  if(NOT AUGUR_IMPLIB_FILE)
    set(AUGUR_IMPLIB_FILE "")
  endif()
  # System libraries the Rust static library needs, as found by Corrosion.
  get_target_property(AUGUR_STATIC_LINK_LIBRARIES augur_capi-static INTERFACE_LINK_LIBRARIES)
  get_target_property(AUGUR_STATIC_LINK_OPTIONS augur_capi-static INTERFACE_LINK_OPTIONS)
  foreach(var AUGUR_STATIC_LINK_LIBRARIES AUGUR_STATIC_LINK_OPTIONS)
    if(NOT ${var})
      set(${var} "")
    endif()
  endforeach()

  set(AUGUR_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/Augur")
  configure_package_config_file(
    cmake/AugurConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/AugurConfig.cmake"
    INSTALL_DESTINATION "${AUGUR_CMAKE_DIR}"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_BINDIR)
  # Before 1.0, minor versions may break compatibility.
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AugurConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion)
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/AugurConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/AugurConfigVersion.cmake"
    DESTINATION "${AUGUR_CMAKE_DIR}")
endif()
