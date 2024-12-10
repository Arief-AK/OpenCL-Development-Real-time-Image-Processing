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

# List all headers used in this project
set(HEADERS
    include/InfoPlatform.hpp
    include/Controller.hpp
    include/FileHandler.hpp
)

# Collect matching sources based on the headers
collect_sources_from_headers(SOURCES include src ${HEADERS})

# Add executable to the CMake framework
add_executable(Grayscale_OpenCL ${SOURCES} grayscale.cpp)

# Include the directory with kernel file
target_include_directories(Grayscale_OpenCL PRIVATE include)
target_include_directories(Grayscale_OpenCL PRIVATE kernel)

# Link the OpenCL and OpenCV library to the executable
target_link_libraries(Grayscale_OpenCL ${OS_LIB} ${OpenCV_LIBS})

# Define the source images directory and the target images directory
set(IMAGES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/Grayscale/images)
message(STATUS "Image sources are located in: " ${IMAGES_SOURCE_DIR})

# Find all images in the source images directory
file(GLOB IMAGE_FILES "${IMAGES_SOURCE_DIR}/*.*") # Match all files in the directory

# Copy each image file to the target images directory
foreach(IMAGE_FILE ${IMAGE_FILES})
    if(WIN32)
        # Define the target image directory and ensure the target images directory exists
        set(IMAGES_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/Debug/images)
        
        # Ensure the target directory is cleaned up
        if(EXISTS ${IMAGES_BINARY_DIR})
            file(REMOVE_RECURSE ${IMAGES_BINARY_DIR})
        endif()
        
        # Make an empty directory
        file(MAKE_DIRECTORY ${IMAGES_BINARY_DIR})

        get_filename_component(IMAGE_NAME ${IMAGE_FILE} NAME)
        configure_file(${IMAGE_FILE} ${IMAGES_BINARY_DIR}/${IMAGE_NAME} COPYONLY)
    elseif(UNIX AND NOT APPLE)
        # Define the target image directory and ensure the target images directory exists
        set(IMAGES_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/images)
        
        # Ensure the target directory is cleaned up
        if(EXISTS ${IMAGES_BINARY_DIR})
            file(REMOVE_RECURSE ${IMAGES_BINARY_DIR})
        endif()
        
        # Make an empty directory
        file(MAKE_DIRECTORY ${IMAGES_BINARY_DIR})

        get_filename_component(IMAGE_NAME ${IMAGE_FILE} NAME)
        configure_file(${IMAGE_FILE} ${IMAGES_BINARY_DIR}/${IMAGE_NAME} COPYONLY)
    endif()
endforeach()
message(STATUS "Image sources have re-located to: " ${IMAGES_BINARY_DIR})

# Move the kernel file(s) into the executable directory
if(WIN32)
    message(STATUS "The executable is located in: " ${CMAKE_CURRENT_BINARY_DIR} "/Debug")
    configure_file(kernel/grayscale.cl ${CMAKE_CURRENT_BINARY_DIR}/Debug/grayscale.cl COPYONLY)
elseif(UNIX AND NOT APPLE)
    message(STATUS "The executable is located in: " ${CMAKE_CURRENT_BINARY_DIR})
    configure_file(kernel/grayscale.cl ${CMAKE_CURRENT_BINARY_DIR}/grayscale.cl COPYONLY)
endif()
