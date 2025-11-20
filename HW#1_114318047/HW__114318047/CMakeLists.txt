cmake_minimum_required(VERSION 3.16)
project(HW1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (expects OpenCVConfig.cmake to be discoverable)
find_package(OpenCV REQUIRED)

add_executable(HW1 HW1.cpp)
target_link_libraries(HW1 PRIVATE ${OpenCV_LIBS})
target_include_directories(HW1 PRIVATE ${OpenCV_INCLUDE_DIRS})

# Silence MSVC secure CRT warnings (Windows optional)
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif()
