# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

# Project-level configuration.
set(PROJECT_NAME "sherpa_onnx_linux")
project(${PROJECT_NAME} LANGUAGES CXX)

# List of absolute paths to libraries that should be bundled with the plugin.
# This list could contain prebuilt libraries, or libraries created by an
# external build triggered from this build file.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
  set(sherpa_onnx_linux_bundled_libraries
    "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_SYSTEM_PROCESSOR}/libsherpa-onnx-c-api.so"
    "${CMAKE_CURRENT_SOURCE_DIR}/${CMAKE_SYSTEM_PROCESSOR}/libonnxruntime.so"
    PARENT_SCOPE
  )
else()
  message(FATAL_ERROR "Unsupported processor for Linux: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
