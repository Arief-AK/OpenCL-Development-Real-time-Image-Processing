#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include <Controller.hpp>
#include <FileHandler.hpp>

// CONSTANTS
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 0

bool PERFORM_COMP = true;
bool SAVE_IMAGES = true;
bool DISPLAY_IMAGES = true;

std::string TEST_DIRECTORY = "images/";
std::string OUTPUT_FILE = "results.csv";

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

void GetImageOpenCL(std::string image_path, std::vector<unsigned char> *input_data, std::vector<unsigned char> *output_data,
    cl_int* width, cl_int* height){
    // Load the input image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if(image.empty()){
        std::cerr << "Failed to load image" << std::endl;
    }

    // Display image (if necessary)
    if(DISPLAY_IMAGES){
        cv::imshow("Reference Image Window", image);
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

    *input_image = image;
}

std::vector<uchar> PerformOpenCL(std::string image_path, cl_context* context, cl_command_queue* command_queue,
    cl_kernel* kernel, double* opencl_execution_time, cl_ulong* opencl_event_start,
    cl_ulong* opencl_event_end, cl_int& width, cl_int& height){
    // Initialise image variables
    std::vector<unsigned char> input_data;
    std::vector<unsigned char> output_data;
    cl_int err_num;

    // Get image
    GetImageOpenCL(image_path, &input_data, &output_data, &width, &height);

    // Initialise the global work size for kernel execution
    size_t global_work_size[] = {static_cast<size_t>(width), static_cast<size_t>(height)};

    // Start profiling execution time
    auto opencl_execution_time_start = std::chrono::high_resolution_clock::now();

    // Create buffers
    auto input_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * input_data.size(), input_data.data(), &err_num);
    if(err_num != CL_SUCCESS){
        std::cerr << "Failed to create input buffer" << std::endl;
    }
    auto output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * output_data.size(), nullptr, &err_num);
    if(err_num != CL_SUCCESS){
        std::cerr << "Failed to create output buffer" << std::endl;
    }

    // Assign the kernel arguments
    err_num = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &input_buffer);
    err_num |= clSetKernelArg(*kernel, 1, sizeof(cl_mem), &output_buffer);
    err_num |= clSetKernelArg(*kernel, 2, sizeof(int), &width);
    err_num |= clSetKernelArg(*kernel, 3, sizeof(int), &height);

    // Create an event
    cl_event event;

    err_num = clEnqueueNDRangeKernel(*command_queue, *kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, &event);
    if(err_num != CL_SUCCESS){
        std::cerr << "Failed when executing kernel" << std::endl;
    }

    // Wait for the event to complete
    clWaitForEvents(1, &event);

    // Read the buffer
    err_num  = clEnqueueReadBuffer(*command_queue, output_buffer, CL_TRUE, 0, sizeof(unsigned char) * output_data.size(), output_data.data(), 0, nullptr, nullptr);
    if(err_num != CL_SUCCESS){
        std::cerr << "Failed to read buffer" << std::endl;
    }

    // End profiling execution time
    auto opencl_execution_time_end = std::chrono::high_resolution_clock::now();

    // Get the RAW kernel timing using OpenCL events
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), opencl_event_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), opencl_event_end, NULL);

    // Calculate the execution time
    *opencl_execution_time = std::chrono::duration<double, std::milli>(opencl_execution_time_end - opencl_execution_time_start).count();

    return output_data;
}

cv::Mat PerformCPU(std::string image_path, double* execution_time){
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
    *execution_time = std::chrono::duration<double, std::milli>(end - start).count();

    return output_image;
}

void PrintEndToEndExecutionTime(std::string method, double total_execution_time_ms){
    std::cout << "\n-------------------- START OF " << method << " EXECUTION TIME (end-to-end) DETAILS --------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Total execution time (OpenCL end-to-end): " << total_execution_time_ms << " ms" << std::endl;
    std::cout << "-------------------- END OF " << method << " EXECUTION TIME (end-to-end) DETAILS --------------------" << std::endl;
}

void PrintRawKernelExecutionTime(cl_ulong start, cl_ulong end){
    // Print the RAW kernel execution time
    double time_ms = (end - start) * 1e-6;

    std::cout << "\n-------------------- START OF KERNEL EXEUCTION DETAILS --------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(10) << "Kernel execution time: " << time_ms << " ms" << std::endl;
    std::cout << "-------------------- END OF KERNEL EXEUCTION DETAILS --------------------" << std::endl;
    std::cout << std::endl;
}

void PrintSummary(cl_ulong& opencl_event_start, cl_ulong& opencl_event_end,
    double& opencl_execution_time, double& cpu_execution_time){
    std::cout << "\n **************************************** START OF OpenCL SUMMARY **************************************** " << std::endl;
    PrintEndToEndExecutionTime("OpenCL", opencl_execution_time);
    PrintRawKernelExecutionTime(opencl_event_start, opencl_event_end);
    std::cout << " **************************************** END OF OpenCL SUMMARY **************************************** " << std::endl;

    std::cout << "\n **************************************** START OF CPU SUMMARY **************************************** " << std::endl;
    PrintEndToEndExecutionTime("CPU", cpu_execution_time);
    std::cout << "\n **************************************** END OF CPU SUMMARY **************************************** " << std::endl;
}

double ComputeMAE(const cv::Mat& reference, const cv::Mat& result){
    cv::Mat difference;
    cv::absdiff(reference, result, difference);
    return cv::mean(difference)[0];
}

void SaveImages(std::string image_path, cv::Mat& opencl_output_image){
    if(SAVE_IMAGES){
        // Convert output data to OpenCV matrix
        auto new_image_path = "images/opencl_grayscale_" + std::filesystem::path(image_path).filename().string();
        cv::imwrite(new_image_path, opencl_output_image);
    }
}

void WriteResultsToCSV(const std::string& filename, std::vector<std::tuple<std::string, double, double, double>>& results){
    std::ofstream file(filename);
    file << "Image, CPU_Time_ms, OpenCL_Time_ms, Error_MAE\n";
    for (const auto& [image, cpu_time, opencl_time, mae] : results) {
        file << image << ", " << cpu_time << ", " << opencl_time << ", " << mae << "\n";
    }
    file.close();
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
    cl_ulong opencl_event_start, opencl_event_end;

    // Initialise results vector
    std::vector<std::tuple<std::string, double, double, double>> comparison_results;

    // Initialise OpenCL platforms and devices
    InitOpenCL(&context, &command_queue, &program, &kernel);

    // Load the images
    auto image_paths = file_handler.LoadImages(TEST_DIRECTORY);
    
    // Iterate through the images
    for(const auto& image_path: image_paths){
        // Initialise image variables
        cl_int width, height;
        
        // Initialise comparison variables
        cv::Mat cpu_output_image;
        double opencl_execution_time = {};
        double cpu_execution_time = {};

        // Perform OpenCL and get output data
        auto output_data = PerformOpenCL(image_path, &context, &command_queue,
            &kernel, &opencl_execution_time, &opencl_event_start, &opencl_event_end, width, height);
        
        cv::Mat opencl_output_image(height, width, CV_8UC1, output_data.data());
        SaveImages(image_path, opencl_output_image);
        std::cout << "OpenCL Grayscale conversion complete" << std::endl;

        // Perform OpenCL vs CPU comparison
        if(PERFORM_COMP){
            cpu_output_image = PerformCPU(image_path, &cpu_execution_time);
            
            // Calculate Mean Absolute Error and push into results vector
            auto average = ComputeMAE(cpu_output_image, opencl_output_image);
            comparison_results.emplace_back(image_path, cpu_execution_time, opencl_execution_time, average);

            // Print summary
            PrintSummary(opencl_event_start, opencl_event_end, opencl_execution_time, cpu_execution_time);

            // Write results to CSV
            WriteResultsToCSV(OUTPUT_FILE, comparison_results);
        }

        if(DISPLAY_IMAGES){
            cv::imshow("OpenCL Grayscale window", opencl_output_image);
            cv::imshow("CPU Grayscale window", cpu_output_image);
            cv::waitKey(0);
        }
    }

    return 0;
}
