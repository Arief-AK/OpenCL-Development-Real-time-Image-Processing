#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include <Controller.hpp>
#include <FileHandler.hpp>

// CONSTANTS
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 0

int NUMBER_OF_ITERATIONS = 1;

bool PERFORM_COMP = true;
bool SAVE_IMAGES = false;
bool DISPLAY_IMAGES = false;
bool DISPLAY_TERMINAL_RESULTS = true;

bool LOG_EVENTS = false;

std::string TEST_DIRECTORY = "images/";
std::string OUTPUT_FILE = "results.csv";

void InitLogger(Logger& logger){
    // Set the log file
    try{
       logger.setLogFile("Grayscale.log");
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error setting log file: " << e.what() << std::endl;
    }
}

void InitOpenCL(Controller& controller, cl_context* context, cl_command_queue* command_queue, cl_program* program, cl_kernel* kernel){
    // Get OpenCL platforms
    auto platforms = controller.GetPlatforms();
    for (auto && platform : platforms){
        controller.DisplayPlatformInformation(platform);
    }

    // Inform user of chosen indexes for platform and device
    std::cout << "\nApplication will use:\nPLATFORM INDEX:\t" << PLATFORM_INDEX << "\nDEVICE INDEX:\t" << DEVICE_INDEX << "\n" << std::endl;

    // Get intended device
    auto devices = controller.GetDevices(platforms[PLATFORM_INDEX]);

    // Check device image support
    cl_bool image_support = CL_FALSE;
    clGetDeviceInfo(devices[DEVICE_INDEX], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &image_support, nullptr);
    if (!image_support) {
        std::cerr << "Device does not support images." << std::endl;
        return;
    }

    // Get OpenCL mandatory properties
    cl_int err_num = 0;
    *context = controller.CreateContext(platforms[PLATFORM_INDEX], devices);
    *command_queue = controller.CreateCommandQueue(*context, devices[DEVICE_INDEX]);
    *program = controller.CreateProgram(*context, devices[DEVICE_INDEX], "grayscale.cl");
    *kernel = controller.CreateKernel(*program, "grayscale");
}

void GetImageOpenCL(std::string image_path, std::vector<unsigned char> *input_data,
    cl_int* width, cl_int* height, Logger& logger){
    // Load the input image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if(image.empty()){
        logger.log("Failed to load image", Logger::LogLevel::ERROR);
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
    cv::Mat rgba_image(*height, *width, CV_8UC4, const_cast<unsigned char*>(_input_data.data()));

    // Assign parameters
    *input_data = _input_data;
}

void GetImageCPU(cv::Mat* input_image, std::string image_path, Logger& logger){
    // Load the input image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if(image.empty()){
        logger.log("Failed to load image", Logger::LogLevel::ERROR);
    }

    *input_image = image;
}

std::vector<uchar> PerformOpenCL(Controller& controller, std::string image_path, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
    double& avg_opencl_execution_time, double& avg_opencl_kernel_write_time, double& avg_opencl_kernel_execution_time, double& avg_opencl_kernel_read_time,
    cl_int& width, cl_int& height, Logger& logger){
    
    std::cout << "Performing OpenCL grayscaling on " << image_path << "..." << std::endl;

    // Initialise image variables
    std::vector<unsigned char> input_data;

    // Initialise profiling variables
    std::vector<cl_ulong> profiling_events;

    // Get image
    GetImageOpenCL(image_path, &input_data, &width, &height, logger);

    // Initialise output variables
    std::vector<float> output_data(width * height * 4);
    std::vector<unsigned char> grayscale_output(width * height * 4);

    // Initialise average variables
    double total_execution_time = 0.0;
    double total_write_time = 0.0, total_kernel_time = 0.0, total_read_time = 0.0;

    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        // Start profiling execution time
        auto opencl_execution_time_start = std::chrono::high_resolution_clock::now();

        // Perform Grayscaling methods
        // controller.PerformCLBufferGrayscaling();
        
        // controller.PerformCLImageGrayscaling(image_path, context, command_queue, kernel,
        //     &profiling_events, &input_data, &output_data,
        //     width, height, logger);

        // Initialise error variable
        cl_int err_num;
        
        // Initialise profiling variables
        cl_event write_event;
        cl_event kernel_event;
        cl_event read_event;
        cl_ulong write_event_start, write_event_end, kernel_event_start, kernel_event_end, read_event_start, read_event_end;

        // Define cl_image variables and format
        cl_image_format input_format;
        input_format.image_channel_order = CL_RGBA;     // RGB
        input_format.image_channel_data_type = CL_UNORM_INT8;

        cl_image_format output_format;
        output_format.image_channel_order = CL_R;       // Single channel (grayscale)
        output_format.image_channel_data_type = CL_FLOAT;

        // Initialise the global work size for kernel execution
        size_t global_work_size[2] = {width, height};

        // Create memory objects
        cl_mem input_image = clCreateImage2D(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &input_format, width, height, 0, input_data.data(), &err_num);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to create cl_image input_image mem object", Logger::LogLevel::ERROR);
        }

        cl_mem output_image = clCreateImage2D(*context, CL_MEM_WRITE_ONLY, &output_format, width, height, 0, nullptr, &err_num);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to create cl_image output_image mem object", Logger::LogLevel::ERROR);
        }

        // Initialise input image
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {width, height, 1};

        err_num = clEnqueueWriteImage(*command_queue, input_image, CL_TRUE, origin, region, 0, 0, input_data.data(), 0, nullptr, &write_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to write cl_image", Logger::LogLevel::ERROR);
        }

        // Set kernel arguments
        err_num = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &input_image);
        err_num |= clSetKernelArg(*kernel, 1, sizeof(cl_mem), &output_image);
        err_num |= clSetKernelArg(*kernel, 2, sizeof(int), &width);
        err_num |= clSetKernelArg(*kernel, 3, sizeof(int), &height);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to set kernel arguments", Logger::LogLevel::ERROR);
        }

        clWaitForEvents(1, &write_event);
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write_event_start, nullptr);
        clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &write_event_end, NULL);
        profiling_events.push_back(write_event_start);
        profiling_events.push_back(write_event_end);

        // Perform kernel
        err_num = clEnqueueNDRangeKernel(*command_queue, *kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, &kernel_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed when executing kernel", Logger::LogLevel::ERROR);
        }

        clWaitForEvents(1, &kernel_event);
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_event_start, NULL);
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_event_end, NULL);
        profiling_events.push_back(kernel_event_start);
        profiling_events.push_back(kernel_event_end);

        // Read back image data
        err_num = clEnqueueReadImage(*command_queue, output_image, CL_TRUE, origin, region, 0, 0, output_data.data(), 0, nullptr, &read_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to read back image data from kernel", Logger::LogLevel::ERROR);
        }

        clWaitForEvents(1, &read_event);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_event_start, NULL);
        clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_event_end, NULL);
        profiling_events.push_back(read_event_start);
        profiling_events.push_back(read_event_end);

        // End profiling execution time
        auto opencl_execution_time_end = std::chrono::high_resolution_clock::now();

        // Calculate total execution time(s)
        total_execution_time += std::chrono::duration<double, std::milli>(opencl_execution_time_end - opencl_execution_time_start).count();
        total_write_time += (profiling_events[1] - profiling_events[0]) * 1e-6;
        total_kernel_time += (profiling_events[3] - profiling_events[2]) * 1e-6;
        total_read_time += (profiling_events[5] - profiling_events[4]) * 1e-6;

        if(LOG_EVENTS){
            // Convert timings into string
            std::ostringstream str_write_start, str_write_end,
                str_kernel_start, str_kernel_end,
                str_read_start, str_read_end;

            str_write_start << profiling_events[0];
            str_write_end << profiling_events[1];
            logger.log("Write event start: " + str_write_start.str(), Logger::LogLevel::INFO);
            logger.log("Write event end: " + str_write_end.str(), Logger::LogLevel::INFO);

            str_kernel_start << profiling_events[2];
            str_kernel_end << profiling_events[3];
            logger.log("Kernel event start: " + str_kernel_start.str(), Logger::LogLevel::INFO);
            logger.log("Kernel event end: " + str_kernel_end.str(), Logger::LogLevel::INFO);

            str_read_start << profiling_events[4];
            str_read_end << profiling_events[5];
            logger.log("Read event start: " + str_read_start.str(), Logger::LogLevel::INFO);
            logger.log("Read event end: " + str_read_end.str(), Logger::LogLevel::INFO);
        }
    }

    logger.log("OpenCL Grayscale conversion complete", Logger::LogLevel::INFO);

    // Calculate averages
    avg_opencl_execution_time = total_execution_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_write_time = total_write_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_execution_time = total_kernel_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_read_time = total_read_time / NUMBER_OF_ITERATIONS;

    for (size_t i = 0; i < (width * height * 4); i++) {
        grayscale_output[i] = static_cast<unsigned char>(output_data[i] * 255.0f); // Extract and scale grayscale
    }

    return grayscale_output;
}

cv::Mat PerformCPU(std::string image_path, double& avg_cpu_execution_time, Logger& logger){
    std::cout << "Performing CPU grayscaling on " << image_path << "..." << std::endl;
    
    // Initialise variables
    cv::Mat input_image;
    cv::Mat output_image;

    // Get image using OpenCV
    GetImageCPU(&input_image, image_path, logger);

    // Initialise average variables
    double total_execution_time = 0.0;

    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        // Convert the image to grayscale and perform execution time profiling
        auto start = std::chrono::high_resolution_clock::now();
        cv::cvtColor(input_image, output_image, cv::COLOR_RGBA2GRAY);
        auto end = std::chrono::high_resolution_clock::now();

        total_execution_time += std::chrono::duration<double, std::milli>(end - start).count();
    }

    logger.log("CPU Grayscale conversion complete", Logger::LogLevel::INFO);

    // Calculate average
    avg_cpu_execution_time = total_execution_time / NUMBER_OF_ITERATIONS;

    return output_image;
}

void PrintEndToEndExecutionTime(std::string method, double total_execution_time_ms, Logger& logger){
    logger.log("-------------------- START OF " + method + " EXECUTION TIME (end-to-end) DETAILS --------------------", Logger::LogLevel::INFO);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << "Total execution time (end-to-end): " << total_execution_time_ms << " ms";
    logger.log(oss.str(), Logger::LogLevel::INFO);

    logger.log("-------------------- END OF " + method + " EXECUTION TIME (end-to-end) DETAILS --------------------", Logger::LogLevel::INFO);
}

void PrintRawKernelExecutionTime(double& opencl_kernel_execution_time, Logger& logger){
    logger.log("-------------------- START OF KERNEL EXEUCTION DETAILS --------------------", Logger::LogLevel::INFO);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(10) << "Kernel execution time: " << opencl_kernel_execution_time << " ms";
    logger.log(oss.str(), Logger::LogLevel::INFO);

    logger.log("-------------------- END OF KERNEL EXEUCTION DETAILS --------------------", Logger::LogLevel::INFO);
}

void PrintSummary(double& opencl_kernel_execution_time, double& opencl_execution_time,
    double& cpu_execution_time, Logger& logger){
    if(DISPLAY_TERMINAL_RESULTS)
        std::cout << "\n **************************************** START OF OpenCL SUMMARY **************************************** " << std::endl;
    
    PrintEndToEndExecutionTime("OpenCL", opencl_execution_time, logger);
    PrintRawKernelExecutionTime(opencl_kernel_execution_time, logger);
    
    if(DISPLAY_TERMINAL_RESULTS){
        std::cout << " **************************************** END OF OpenCL SUMMARY **************************************** " << std::endl;
        std::cout << "\n **************************************** START OF CPU SUMMARY **************************************** " << std::endl;
    }

    PrintEndToEndExecutionTime("CPU", cpu_execution_time, logger);

    if(DISPLAY_TERMINAL_RESULTS)
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

void WriteResultsToCSV(const std::string& filename, std::vector<std::tuple<std::string, std::string, std::string, int, double, double, double, double>>& results){
    std::ofstream file(filename);
    file << "Timestamp, Image, Resolution, Num_Iterations, avg_CPU_Time_ms, avg_OpenCL_Time_ms, avg_OpenCL_kernel_ms, Error_MAE\n";
    for (const auto& [timestamp, image, resolution, num_iterations, avg_cpu_time, avg_opencl_time, avg_opencl_kernel_time, mae] : results) {
        file << timestamp << ", " << image << ", " << resolution << ", " << num_iterations << ", " << avg_cpu_time << ", " << avg_opencl_time << ", " << avg_opencl_kernel_time << ", " << mae << "\n";
    }
    file.close();
}

int main(int, char**){
    std::cout << "Hello, from Grayscale application!\n";

    // Initialise (singleton) Logger
    Logger& logger = Logger::getInstance();
    InitLogger(logger);
    logger.setTerminalDisplay(DISPLAY_TERMINAL_RESULTS);
    logger.log("Initialised logger", Logger::LogLevel::INFO);

    // Initialise controllers
    Controller controller;
    FileHandler file_handler;

    // Initialise OpenCL variables
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err_num = 0;

    // Initialise results vector
    std::vector<std::tuple<std::string, std::string, std::string, int, double, double, double, double>> comparison_results;

    // Initialise OpenCL platforms and devices
    InitOpenCL(controller, &context, &command_queue, &program, &kernel);

    // Load the images
    auto image_paths = file_handler.LoadImages(TEST_DIRECTORY);
    
    // Iterate through the images
    for(const auto& image_path: image_paths){
        // Initialise image variables
        cl_int width, height;
        
        // Initialise comparison image
        cv::Mat cpu_output_image;

        // Initialise timing variables
        double avg_opencl_execution_time = {};
        double avg_opencl_kernel_write_time = {};
        double avg_opencl_kernel_execution_time = {};
        double avg_opencl_kernel_read_time = {};
        double avg_cpu_execution_time = {};

        // Perform OpenCL and get output data
        auto output_data = PerformOpenCL(controller, image_path, &context, &command_queue,&kernel,
            avg_opencl_execution_time, avg_opencl_kernel_write_time, avg_opencl_kernel_execution_time, avg_opencl_kernel_read_time,
            width, height, logger);
        
        cv::Mat opencl_output_image(height, width, CV_8UC1, const_cast<unsigned char*>(output_data.data()));
        SaveImages(image_path, opencl_output_image);

        // Perform OpenCL vs CPU comparison
        if(PERFORM_COMP){
            cpu_output_image = PerformCPU(image_path, avg_cpu_execution_time, logger);
            
            // Calculate Mean Absolute Error and push into results vector
            auto MAE = ComputeMAE(cpu_output_image, opencl_output_image);
            
            // Get timestamp
            auto timestamp = logger.getCurrentTime();
            
            // Get the resolution
            std::ostringstream str_width, str_height;
            str_width << width;
            str_height << height;
            std::string resolution = str_width.str()  + "x" + str_height.str();

            // Append to the comparison result vector
            comparison_results.emplace_back(timestamp, image_path, resolution, NUMBER_OF_ITERATIONS,
                                            avg_cpu_execution_time, avg_opencl_execution_time,
                                            avg_opencl_kernel_execution_time, MAE);

            // Print summary
            PrintSummary(avg_opencl_kernel_execution_time, avg_opencl_execution_time, avg_cpu_execution_time, logger);

            // Write results to CSV
            WriteResultsToCSV(OUTPUT_FILE, comparison_results);
        }

        if(DISPLAY_IMAGES){
            cv::imshow("OpenCL Grayscale window", opencl_output_image);
            cv::imshow("CPU Grayscale window", cpu_output_image);
            cv::waitKey(0);
        }
    }

    std::cout << "Done!" << std::endl;
    return 0;
}
