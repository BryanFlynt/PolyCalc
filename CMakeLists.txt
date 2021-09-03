#---------------------------------------------------------------------
# Build System for xstd
#---------------------------------------------------------------------
cmake_minimum_required(VERSION 3.1)
project(polycalc VERSION 1.0.0 LANGUAGES CXX)

#---------------------------------------------------------------------
# User Configure Build Process
#---------------------------------------------------------------------
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_COLOR_MAKEFILE TRUE)
set(DEFAULT_BUILD_TYPE "Release")

#---------------------------------------------------------------------
# Set location of *.cmake modules
#---------------------------------------------------------------------
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#---------------------------------------------------------------------
# Force build in seperate directory
#---------------------------------------------------------------------
include(PreventInSourceBuilds)

#---------------------------------------------------------------------
# Locations of Installation (CMAKE_INSTALL_PREFIX)
#---------------------------------------------------------------------
#set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})

#---------------------------------------------------------------------
# Directory Report
#---------------------------------------------------------------------
message(VERBOSE "")
message(VERBOSE "--------------------- Directory Report -----------------------")
message(VERBOSE " ") # CMAKE directories are from top level CMakeLists.txt
message(VERBOSE "Top Level Directories:")
message(VERBOSE "CMAKE_SOURCE_DIR     = ${CMAKE_SOURCE_DIR}")
message(VERBOSE "CMAKE_BINARY_DIR     = ${CMAKE_BINARY_DIR}")
message(VERBOSE " ") # PROJECT directories are for recent project call
message(VERBOSE "Project Level Directories:")
message(VERBOSE "PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")
message(VERBOSE "PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")
message(VERBOSE " ") # My installation directories
message(VERBOSE "Installation Directories:")
message(VERBOSE "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")


#---------------------------------------------------------------------
# Locations of Installation & Report
#---------------------------------------------------------------------

# Source locations in source tree
set(MY_PROJECT_PREFIX ${PROJECT_SOURCE_DIR})
set(MY_PROJECT_INCDIR ${MY_PROJECT_PREFIX}/include)
set(MY_PROJECT_TSTDIR ${MY_PROJECT_PREFIX}/test)
set(MY_PROJECT_DOCDIR ${MY_PROJECT_PREFIX}/docs)

# Where to place all libraries
set(MY_INSTALL_PREFIX ${CMAKE_BINARY_DIR})
set(MY_INSTALL_INCDIR ${MY_INSTALL_PREFIX}/include)
set(MY_INSTALL_TSTDIR ${MY_INSTALL_PREFIX}/test)
set(MY_INSTALL_DOCDIR ${MY_INSTALL_PREFIX}/docs)

message(STATUS " ")
message(STATUS "Using Directories:")
message(STATUS "--- Original Locations ---")
message(STATUS "MY_PROJECT_PREFIX = ${MY_PROJECT_PREFIX}")
message(STATUS "MY_PROJECT_INCDIR = ${MY_PROJECT_INCDIR}")
message(STATUS "MY_PROJECT_TSTDIR = ${MY_PROJECT_TSTDIR}")
message(STATUS "MY_PROJECT_DOCDIR = ${MY_PROJECT_DOCDIR}")
message(STATUS " ")
message(STATUS "--- Installation Locations ---")
message(STATUS "MY_INSTALL_PREFIX = ${MY_INSTALL_PREFIX}")
message(STATUS "MY_INSTALL_INCDIR = ${MY_INSTALL_INCDIR}")
message(STATUS "MY_INSTALL_TSTDIR = ${MY_INSTALL_TSTDIR}")
message(STATUS "MY_INSTALL_DOCDIR = ${MY_INSTALL_DOCDIR}")

#
#---------------------------------------------------------------------
# Compiler Feature Detection & Selection
#---------------------------------------------------------------------
#
message("")
message("--------------- Compiler Feature Detection -----------------")
message("")
message("CMAKE_CXX_COMPILER         = ${CMAKE_CXX_COMPILER}")
message("CMAKE_CXX_COMPILER_ID      = ${CMAKE_CXX_COMPILER_ID}")
message("CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
message("")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#---------------------------------------------------------------------
# Detect Library includes, flags, etc.
#---------------------------------------------------------------------
message(VERBOSE "")
message(VERBOSE "================================================================")
message(VERBOSE "                    Searching for Libraries                     ")
message(VERBOSE "================================================================")
if(${CMAKE_VERSION} VERSION_GREATER "3.11.999") 
	cmake_policy(SET CMP0074 NEW) # find_package search <name>_ROOT
endif()

find_package(Doxygen)
	
#---------------------------------------------------------------------
# 
#---------------------------------------------------------------------

# Build Libraries & Applications
add_subdirectory(src)

# Build Tests for Libraries
enable_testing()
add_subdirectory(test)

# Build Documentation (DOxygen)
if(DOXYGEN_FOUND)
#	add_subdirectory(docs)
endif()
