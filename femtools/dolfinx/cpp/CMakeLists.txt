# femtools.dolfinx's C++ extensions, one per build: MODULE names it (source MODULE.cpp,
# extension MODULE_x). Built on first use by femtools/dolfinx/compiled.py against the
# installed DOLFINx and nanobind.
cmake_minimum_required(VERSION 3.21)
set(MODULE "plaza_periodic" CACHE STRING "the extension to build: plaza_periodic or folded_assembly")
project(femtools_${MODULE}_x LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
                OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)
find_package(DOLFINX REQUIRED CONFIG)
nanobind_add_module(${MODULE}_x NOMINSIZE ${MODULE}.cpp)
target_link_libraries(${MODULE}_x PRIVATE dolfinx)
# femtools/cpp/plaza_periodic_rules.h, shared with the dolfin module
target_include_directories(${MODULE}_x PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp)
