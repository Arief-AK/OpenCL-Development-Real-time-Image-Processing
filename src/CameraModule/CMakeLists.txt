set(PROJECT CameraModule)

# Print the build type
message(STATUS "CMake Build Type: ${CMAKE_BUILD_TYPE}")

# Set runtime output directory based on build type
if(UNIX)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endif()

# Find OpenCV package
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Operating System: ${CMAKE_SYSTEM_NAME}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# Add the executable
add_executable(${PROJECT} ${SOURCES} CameraModule.cpp)

# Link OpenCL library
target_link_libraries(${PROJECT} PRIVATE ${OpenCL_LIBRARIES} ${OpenCV_LIBS})