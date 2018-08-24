cmake_minimum_required(VERSION 3.5.0)
project(particle_wrappers)

# Set flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find required packages
find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

pybind11_add_module(particle_wrapper SHARED
  particle_wrapper.cpp
  l2projection.cpp
  advect_particles.cpp
  formutils.cpp
  particles.cpp
  pdestaticcondensation.cpp)


# Add DOLFIN libraries and other config
target_link_libraries(particle_wrapper PRIVATE pybind11::module dolfin)
