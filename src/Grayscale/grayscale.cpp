#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include <Controller.hpp>
#include <FileHandler.hpp>

// CONSTANTS
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 0

bool TIME_EXECUTION = true;
bool DISPLAY_IMAGES = false;

std::string test_directory = "images/";

void InitOpenCL(cl_context* context, cl_command_queue* command_queue, cl_program* program, cl_kernel* kernel){
    // Initialise OpenCL variables
    Controller controller;

    // Get OpenCL platforms
    auto platforms = controller.GetPlatforms();
    for (auto && platform : platforms){
        controller.DisplayPlatformInformation(platform);
    }

    // Inform user of chosen indexes for platform and device
    std::cout << "\nApplication will use:\nPLATFORM INDEX:\t" << PLATFORM_INDEX << "\nDEVICE INDEX:\t" << DEVICE_INDEX << "\n" << std::endl;

    // Get intended device
    auto devices = controller.GetDevices(platforms[PLATFORM_INDEX]);

    // Get OpenCL mandatory properties
    cl_int err_num = 0;
    *context = controller.CreateContext(platforms[PLATFORM_INDEX], devices);
    *command_queue = controller.CreateCommandQueue(*context, devices[DEVICE_INDEX]);
    *program = controller.CreateProgram(*context, devices[DEVICE_INDEX], "grayscale.cl");
    *kernel = controller.CreateKernel(*program, "grayscale");
}

void GetImageOpenCL(std::string image_path, std::vector<unsigned char> *input_data, std::vector<unsigned char> *output_data, cl_int* width, cl_int* height){
    // Load the input image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if(image.empty()){
        std::cerr << "Failed to load image" << std::endl;
    }

    // Display image (if necessary)
    if(DISPLAY_IMAGES){
        cv::imshow("Display Window", image);
        cv::waitKey(0);
    }

    // Convert to RGBA and get image dimensions
    cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
    *width = image.cols;
    *height = image.rows;

    // Flatten image into uchar array
    std::vector<unsigned char> _input_data(image.data, image.data + image.total() * 4);
    std::vector<unsigned char> _output_data(*width * *height);

    // Assign parameters
    *input_data = _input_data;
    *output_data = _output_data;
}

void GetImageCPU(cv::Mat* input_image, std::string image_path){
    // Load the input image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if(image.empty()){
        std::cerr << "Failed to load image" << std::endl;
    }
    
    if(DISPLAY_IMAGES){
        cv::imshow("CPU Display Window", image);
        cv::waitKey(0);
    }

    *input_image = image;
}

void PerformCPU(std::string image_path){
    std::cout << "Performing grayscale on the CPU" << std::endl;

    // Initialise variables
    cv::Mat input_image;
    cv::Mat output_image;

    // Get image using OpenCV
    GetImageCPU(&input_image, image_path);

    // Convert the image to grayscale and perform execution time profiling
    auto start = std::chrono::high_resolution_clock::now();
    cv::cvtColor(input_image, output_image, cv::COLOR_RGBA2GRAY);
    auto end = std::chrono::high_resolution_clock::now();

    // Print results
    auto execution_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "\n-------------------- START OF CPU EXEUCTION DETAILS --------------------" << std::endl;
    std::cout << "Grayscale conversion execution time: " << execution_time << " ms" << std::endl;
    std::cout << "-------------------- END OF CPU EXEUCTION DETAILS --------------------" << std::endl;
}

void PrintEndToEndExecutionTime(std::string method, std::chrono::steady_clock::time_point execution_time_start,
    std::chrono::steady_clock::time_point execution_time_end){
    // Print the complete execution time
    auto total_execution_time_ms = std::chrono::duration<double, std::milli>(execution_time_end - execution_time_start).count();
    
    std::cout << "\n-------------------- START OF OpenCL EXECUTION TIME (end-to-end) DETAILS --------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Total execution time (OpenCL end-to-end): " << total_execution_time_ms << " ms" << std::endl;
    std::cout << "-------------------- END OF OpenCL EXECUTION TIME (end-to-end) DETAILS --------------------" << std::endl;
}

void PrintRawKernelExecutionTime(cl_ulong start, cl_ulong end){
    // Print the RAW kernel execution time
    double time_ms = (end - start) * 1e-6;

    std::cout << "\n-------------------- START OF KERNEL EXEUCTION DETAILS --------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(10) << "Kernel execution time: " << time_ms << " ms" << std::endl;
    std::cout << "-------------------- END OF KERNEL EXEUCTION DETAILS --------------------" << std::endl;
    std::cout << std::endl;
}

int main(int, char**){
    std::cout << "Hello, from Grayscale application!\n";

    // Initialise FileHandler
    FileHandler file_handler;

    // Initialise OpenCL variables
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err_num = 0;

    // Initialise OpenCL platforms and devices
    InitOpenCL(&context, &command_queue, &program, &kernel);

    // Load the images
    auto image_paths = file_handler.LoadImages(test_directory);
    
    // Iterate through the images
    for(const auto& image_path: image_paths){
        // Initialise image variables
        std::vector<unsigned char> input_data;
        std::vector<unsigned char> output_data;
        cl_int width, height;

        // Get image
        GetImageOpenCL(image_path, &input_data, &output_data, &width, &height);

        // Initialise the global work size for kernel execution
        size_t global_work_size[] = {static_cast<size_t>(width), static_cast<size_t>(height)};

        // Start profiling execution time
        auto opencl_execution_time_start = std::chrono::high_resolution_clock::now();

        // Create buffers
        auto input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * input_data.size(), input_data.data(), &err_num);
        if(err_num != CL_SUCCESS){
            std::cerr << "Failed to create input buffer" << std::endl;
        }
        auto output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * output_data.size(), nullptr, &err_num);
        if(err_num != CL_SUCCESS){
            std::cerr << "Failed to create output buffer" << std::endl;
        }

        // Assign the kernel arguments
        err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
        err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
        err_num |= clSetKernelArg(kernel, 2, sizeof(int), &width);
        err_num |= clSetKernelArg(kernel, 3, sizeof(int), &height);

        // Create an event
        cl_event event;

        err_num = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, &event);
        if(err_num != CL_SUCCESS){
            std::cerr << "Failed when executing kernel" << std::endl;
        }

        // Wait for the event to complete
        clWaitForEvents(1, &event);

        // Read the buffer
        err_num  = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, sizeof(unsigned char) * output_data.size(), output_data.data(), 0, nullptr, nullptr);
        if(err_num != CL_SUCCESS){
            std::cerr << "Failed to read buffer" << std::endl;
        }

        // End profiling execution time
        auto opencl_execution_time_end = std::chrono::high_resolution_clock::now();

        // Get the RAW kernel timing using OpenCL events
        cl_ulong start, end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

        // Convert output to OpenCV matrix
        auto new_image_path = "images/grayscale_" + std::filesystem::path(image_path).filename().string();
        
        cv::Mat output_image(height, width, CV_8UC1, output_data.data());
        cv::imwrite(new_image_path, output_image);
        std::cout << "Grayscale conversion complete. Displaying Grayscale window." << std::endl;

        // Display grayscale window
        if(DISPLAY_IMAGES){
            cv::Mat grayscale_image = cv::imread(new_image_path, cv::IMREAD_COLOR);
            cv::imshow("Grayscale window", grayscale_image);
            cv::waitKey(0);
        }

        // Print the RAW kernel duration
        if(TIME_EXECUTION){
            // OpenCL
            PrintEndToEndExecutionTime( "OpenCL", opencl_execution_time_start, opencl_execution_time_end);
            PrintRawKernelExecutionTime(start, end);

            // CPU
            PerformCPU(image_path);
        }
    }

    return 0;
}
