################################################################################
cmake_minimum_required(VERSION 3.1)
project(pyaabb)
################################################################################

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Color output
include(UseColors)

# Prepend function
include(PrependCurrentPath)


# Extra warnings
include(Warnings)

# Use C++11/14
include(CXXFeatures)

# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	foreach(config ${CMAKE_CONFIGURATION_TYPES})
		string(TOUPPER ${config} config)
		string(REPLACE /MD /MT CMAKE_C_FLAGS_${config} "${CMAKE_C_FLAGS_${config}}")
		string(REPLACE /MD /MT CMAKE_CXX_FLAGS_${config} "${CMAKE_CXX_FLAGS_${config}}")
	endforeach()
endif()

################################################################################


# Setup dependencies
include(Dependencies)


# pybind11
download_pybind11()
add_subdirectory(${THIRD_PARTY_DIR}/pybind11)


################################################################################
# Subdirectories
################################################################################
add_library(pyaabb MODULE src/binding.cpp src/AABB.cpp src/Predicates_psm.cpp)
target_link_libraries(pyaabb PRIVATE pybind11::module)
set_target_properties(pyaabb PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")



download_eigen()
target_include_directories(pyaabb PUBLIC ${THIRD_PARTY_DIR}/eigen)

