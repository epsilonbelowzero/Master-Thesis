cmake_minimum_required(VERSION 3.13)
project(masterarbeit CXX)

if(NOT (dune-common_DIR OR dune-common_ROOT OR
      "${CMAKE_PREFIX_PATH}" MATCHES ".*dune-common.*"))
    string(REPLACE  ${PROJECT_NAME} dune-common dune-common_DIR
      ${PROJECT_BINARY_DIR})
endif()

#find dune-common and set the module path
find_package(dune-common REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules"
  ${dune-common_MODULE_PATH})

find_package(OpenMP REQUIRED)
#~ find_package(GMP REQUIRED)
find_package(Alberta REQUIRED)

#include the dune macros
include(DuneMacros)

# start a dune project with information from dune.module
dune_project()

dune_enable_all_packages()

add_subdirectory(src)
add_subdirectory(dune)
add_subdirectory(doc)
add_subdirectory(cmake/modules)

add_dune_alberta_flags(WORLDDIM 2 masterarbeit)
if(OpenMP_FOUND)
	target_link_libraries(masterarbeit PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_options(masterarbeit PRIVATE -fconcepts)
target_compile_options(masterarbeit PRIVATE -std=c++20)
target_compile_options(masterarbeit PRIVATE -O2)
target_compile_options(masterarbeit PRIVATE -flto=auto)
target_compile_options(masterarbeit PRIVATE -mtune=native)
target_link_options(masterarbeit PRIVATE -flto=auto)

# finalize the dune project, e.g. generating config.h etc.
finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
