cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(NAME Rin)

project(${NAME})

# Use FindVulkan module added with CMAKE 3.7
if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
    message(STATUS "Using module to find Vulkan")
    find_package(Vulkan)
endif()

IF (NOT Vulkan_FOUND)
	message(FATAL_ERROR "Could not find Vulkan library!")
ELSE()
	message(STATUS ${Vulkan_LIBRARY})
ENDIF()

include_directories(external)
include_directories(external/glm)
include_directories(external/gli)
include_directories(external/glfw/include)
include_directories(external/meshoptimizer)
include_directories(external/volk)
include_directories(base)
include_directories(${Vulkan_INCLUDE_DIRS})

# Set preprocessor defines
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -D_USE_MATH_DEFINES")

IF(WIN32)
ENDIF(WIN32)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")

# Compiler specific stuff
IF(MSVC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
ENDIF(MSVC)

# glfw
set(GLFW_BUILD_EXAMPLES OFF CACHE STRING "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE STRING "" FORCE)
set(GLFW_INSTALL        OFF CACHE STRING "" FORCE)
add_subdirectory(external/glfw)

# glog
# set(BUILD_TESTING OFF CACHE STRING "" FORCE)
# set(WITH_GFLAGS OFF CACHE STRING "" FORCE)
# add_subdirectory(external/glog)

IF(WIN32)
add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
ENDIF(WIN32)

add_subdirectory(external/volk)
add_subdirectory(external/meshoptimizer)

add_subdirectory(base)
add_subdirectory(examples)
