cmake_minimum_required(VERSION 3.22)
project(NeuralGreyBoxAmp VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 23)
set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)

include(cmake/get_cpm.cmake)

CPMAddPackage(
        NAME JUCE
        GIT_TAG 7.0.7
        VERSION 7.0.7
        GITHUB_REPOSITORY juce-framework/JUCE
        SOURCE_DIR ${LIB_DIR}/juce
)

CPMAddPackage(
        NAME EIGEN
        GIT_TAG 3.4.0
        VERSION 3.4.0
        GITLAB_REPOSITORY libeigen/eigen
        SOURCE_DIR ${LIB_DIR}/eigen
        DOWNLOAD_ONLY TRUE
)

CPMAddPackage(
        NAME JSON
        VERSION 3.11.2
        GITHUB_REPOSITORY nlohmann/json
)

add_subdirectory(NeuralGreyBoxAmpRT)