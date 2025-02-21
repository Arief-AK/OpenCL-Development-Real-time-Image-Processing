# Set variables
set(APPLICATION_NAME RealtimeImageProcessing)
set(HEADERS
    ../../include/ProgramHandler.hpp
    ../../include/Comparator.hpp
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
target_include_directories(${APPLICATION_NAME} PRIVATE kernel)

# Link the OpenCL and OpenCV library to the executable
target_link_libraries(${APPLICATION_NAME} ${OS_LIB} ${OpenCV_LIBS})

# Define the source images directory and the target images directory
set(IMAGES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/images)
message(STATUS "Image sources are located in: " ${IMAGES_SOURCE_DIR})

set(KERNEL_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/${APPLICATION_NAME}/kernel)
message(STATUS "Kernels are located in: " ${KERNEL_SOURCE_DIR})

# Find all images in the source images directory
file(GLOB IMAGE_FILES "${IMAGES_SOURCE_DIR}/*.*") # Match all files in the directory
file(GLOB KERNEL_FILES "${KERNEL_SOURCE_DIR}/*.*")

# Define the target image and kernel directory
set(IMAGES_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/images)
set(KERNEL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})

# Ensure the target directory is cleaned up
if(EXISTS ${IMAGES_BINARY_DIR})
    file(REMOVE_RECURSE ${IMAGES_BINARY_DIR})
endif()

# Make an empty directory
file(MAKE_DIRECTORY ${IMAGES_BINARY_DIR})

# Copy each image file to the target images directory
foreach(IMAGE_FILE ${IMAGE_FILES})
    get_filename_component(IMAGE_NAME ${IMAGE_FILE} NAME)
    configure_file(${IMAGE_FILE} ${IMAGES_BINARY_DIR}/${IMAGE_NAME} COPYONLY)
endforeach()
message(STATUS "The images were re-located to: ${IMAGES_BINARY_DIR}")

# Move the kernel file(s) into the executable directory
foreach(KERNEL_FILE ${KERNEL_FILES})
    get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME)
    configure_file(${KERNEL_FILE} ${KERNEL_BINARY_DIR}/${KERNEL_NAME} COPYONLY)
endforeach()
message(STATUS "Kernels are located in: " ${KERNEL_BINARY_DIR})