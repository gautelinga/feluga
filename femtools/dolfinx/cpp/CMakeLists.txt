# femtools.dolfinx.refine's C++ extension (plaza_periodic.cpp), built on first use by
# femtools/dolfinx/refine.py against the installed DOLFINx and nanobind.
cmake_minimum_required(VERSION 3.21)
project(femtools_plaza_periodic_x LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
                OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)
find_package(DOLFINX REQUIRED CONFIG)
nanobind_add_module(plaza_periodic_x NOMINSIZE plaza_periodic.cpp)
target_link_libraries(plaza_periodic_x PRIVATE dolfinx)
# femtools/cpp/plaza_periodic_rules.h, shared with the dolfin module
target_include_directories(plaza_periodic_x PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp)
