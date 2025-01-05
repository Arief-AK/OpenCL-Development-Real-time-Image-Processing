# Set variables
set(APPLICATION_NAME RealtimeImageProcessing)
set(HEADERS
    ../../include/Controller.hpp
    ../../include/InfoPlatform.hpp
    ../../include/FileHandler.hpp
    ../../include/Logger.hpp
)
set(IMAGES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/images)
set(IMAGES_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/images)

if(UNIX)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endif()

# Print the application name and build type
message(STATUS "Perfoming CMake configuration for ${APPLICATION_NAME}")
message(STATUS "CMake Build Type: ${CMAKE_BUILD_TYPE}")

# Function to correlate sources to header files
function(collect_sources_from_headers source_var header_dir source_dir)
    set(sources "")
    foreach(header ${ARGN})
        get_filename_component(name ${header} NAME_WE)
        list(APPEND sources ${source_dir}/${name}.cpp)
    endforeach()
    set(${source_var} ${sources} PARENT_SCOPE)
endfunction()

# Find OpenCL package/installation
find_package(OpenCL REQUIRED)
if (OpenCL_FOUND)
    message(STATUS "Operating System: ${CMAKE_SYSTEM_NAME}")
    include_directories(${OpenCL_INCLUDE_DIRS})
    message(STATUS "Found OpenCL: ${OpenCL_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCL not found")
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

# Set the OpenCL library
set(OS_LIB OpenCL::OpenCL)

# Collect matching sources based on the headers
collect_sources_from_headers(SOURCES include src ${HEADERS})

# Add executable to the CMake framework
add_executable(${APPLICATION_NAME} ${SOURCES} RealtimeImageProcessing.cpp)

# Include the directory with kernel file
target_include_directories(${APPLICATION_NAME} PRIVATE ../../include)

# Link the OpenCL and OpenCV library to the executable
target_link_libraries(${APPLICATION_NAME} ${OS_LIB} ${OpenCV_LIBS})

# Move the kernel file(s) into the executable directory
configure_file(kernel/grayscale_base.cl ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/grayscale_base.cl COPYONLY)
message(STATUS "The kernel is located in: " ${CMAKE_CURRENT_BINARY_DIR})

configure_file(kernel/grayscale_images.cl ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/grayscale_images.cl COPYONLY)
message(STATUS "The kernel is located in: " ${CMAKE_CURRENT_BINARY_DIR})