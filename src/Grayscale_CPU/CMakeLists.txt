# Find OpenCV package
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Operating System: ${CMAKE_SYSTEM_NAME}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# Add executable to the CMake framework
add_executable(Grayscale_CPU grayscale_cpu.cpp)

# Link the OpenCV library to the executable
target_link_libraries(Grayscale_CPU ${OpenCV_LIBS})

# Move the image into the executable directory
if(WIN32)
    message(STATUS "The executable is located in: " ${CMAKE_CURRENT_BINARY_DIR} "/Debug")
    configure_file(galaxy.jpg ${CMAKE_CURRENT_BINARY_DIR}/Debug/galaxy.jpg COPYONLY)
elseif(UNIX AND NOT APPLE)
    message(STATUS "The executable is located in: " ${CMAKE_CURRENT_BINARY_DIR})
    configure_file(galaxy.jpg ${CMAKE_CURRENT_BINARY_DIR}/galaxy.jpg COPYONLY)
endif()