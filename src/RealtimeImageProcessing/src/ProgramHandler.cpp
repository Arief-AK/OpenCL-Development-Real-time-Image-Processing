#include "ProgramHandler.hpp"

ProgramHandler::ProgramHandler(int number_of_iterations, bool log_events, bool display_images, bool display_terminal_results,
int gaussian_kernel_size, float gaussian_sigma): NUMBER_OF_ITERATIONS{number_of_iterations}, LOG_EVENTS{log_events}, DISPLAY_IMAGES{display_images}, DISPLAY_TERMINAL_RESULTS{display_terminal_results},
GAUSSIAN_KERNEL_SIZE{gaussian_kernel_size}, GAUSSIAN_SIGMA{gaussian_sigma}
{
     METHOD = {"GRAYSCALE", "GAUSSIAN"};
}

void ProgramHandler::InitLogger(Logger &logger, Logger::LogLevel level)
{
    // Set the log file
    try{
        logger.setLogFile("RealtimeImageProcessing.log");
        logger.setLogLevel(level);
        logger.setTerminalDisplay(DISPLAY_TERMINAL_RESULTS);
        logger.log("Initialised logger", Logger::LogLevel::INFO);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error setting log file: " << e.what() << std::endl;
    }
}

void ProgramHandler::AddKernels(std::vector<std::string> kernels, std::string kernel_index)
{
    KERNELS.insert({kernel_index, kernels});
}

void ProgramHandler::SetDeviceProperties(int platform_index, int device_index)
{
    PLATFORM_INDEX = platform_index;
    DEVICE_INDEX = device_index;
}

void ProgramHandler::InitOpenCL(Controller &controller, cl_context *context, cl_command_queue *command_queue, cl_program *program, cl_kernel *kernel, std::string method, Logger& logger)
{
    // Get OpenCL platforms
    auto platforms = controller.GetPlatforms();

    // Initialise variables
    auto image_processing_method = 0;
    std::string kernel_file;
    std::string kernel_name;

    // Get intended device
    auto devices = controller.GetDevices(platforms[PLATFORM_INDEX]);

    if(DISPLAY_TERMINAL_RESULTS){
        for (auto && platform : platforms){
            controller.DisplayPlatformInformation(platform);
        }

        std::ostringstream oss;

        // Inform user of chosen indexes for platform and device
        oss << "\nApplication will use:\nPLATFORM INDEX:\t" << PLATFORM_INDEX << "\nDEVICE INDEX:\t" << DEVICE_INDEX << "\n" << std::endl;
        logger.log(oss.str(), Logger::LogLevel::INFO);
        oss.str("");

        char device_name[256];
        clGetDeviceInfo(devices[DEVICE_INDEX], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
        oss << "Device name: " << device_name << std::endl;
        logger.log(oss.str(), Logger::LogLevel::INFO);
        oss.str("");
    }

    // Find method
    if(method == "GRAYSCALE"){
        kernel_name = "grayscale";
    } else if(method == "EDGE"){
        kernel_name = "sobel_edge_detection";
    } else if(method == "GAUSSIAN"){
        kernel_name = "gaussian_blur";
    } else{
        std::cerr << "Unrecognised method" << std::endl;
        exit(1);
    }

    // Check device image support
    cl_bool image_support = CL_FALSE;
    clGetDeviceInfo(devices[DEVICE_INDEX], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &image_support, nullptr);

    switch (image_support)
    {
    case CL_TRUE:
        kernel_file = KERNELS[method][0];
        std::cout << "Device supports images." << std::endl;
        break;

    case CL_FALSE:
        kernel_file = KERNELS[method][1];
        std::cout << "Device does not support images. Using buffers instead of image2D structures." << std::endl;
        break;
    
    default:
        break;
    }
    controller.SetImageSupport(image_support);

    // Get OpenCL mandatory properties
    cl_int err_num = 0;
    *context = controller.CreateContext(platforms[PLATFORM_INDEX], devices);
    *command_queue = controller.CreateCommandQueue(*context, devices[DEVICE_INDEX]);
    *program = controller.CreateProgram(*context, devices[DEVICE_INDEX], kernel_file.c_str());
    *kernel = controller.CreateKernel(*program, kernel_name.c_str());
}

void ProgramHandler::GetImageOpenCL(std::string image_path, std::vector<unsigned char> *input_data, cl_int *width, cl_int *height, Logger &logger)
{
    // Load the input image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if(image.empty()){
        logger.log("Failed to load image", Logger::LogLevel::ERROR);
    }

    // Display image (if necessary)
    if(DISPLAY_IMAGES){
        cv::imshow("Reference Image Window", image);
    }

    // Convert to RGBA and get image dimensions
    cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
    *width = image.cols;
    *height = image.rows;

    // Flatten image into uchar array
    std::vector<unsigned char> _input_data(image.data, image.data + image.total() * 4);

    // Assign parameters
    *input_data = _input_data;
}

void ProgramHandler::GetMatrix(const cv::Mat &input_frame, std::vector<unsigned char> *input_data, cl_int* width, cl_int* height, Logger &logger)
{
    std::vector<unsigned char> _input_data(input_frame.data, input_frame.data + input_frame.total() * 4); // RGBA data
    *input_data = _input_data;
}

std::vector<unsigned char> ProgramHandler::PerformOpenCL(Controller &controller, std::string image_path, cl_context *context, cl_command_queue *command_queue, cl_kernel *kernel,
    double &avg_opencl_execution_time, double &avg_opencl_kernel_write_time, double &avg_opencl_kernel_execution_time, double &avg_opencl_kernel_read_time,
    double &avg_opencl_kernel_operation, cl_int &width, cl_int &height, Logger &logger, std::string method)
{
    auto image_processing_method = 0;
    std::ostringstream oss;

    // Initialise image variables
    std::vector<unsigned char> input_data;

    // Initialise profiling variables
    std::vector<cl_ulong> profiling_events;

    // Get image
    GetImageOpenCL(image_path, &input_data, &width, &height, logger);

    // Initialise output variables
    std::vector<float> grayscale_output_data;
    
    // Initialise function output variable
    std::vector<unsigned char> function_output;

    // Initialise average variables
    double total_execution_time = 0.0;
    double total_write_time = 0.0, total_kernel_time = 0.0, total_read_time = 0.0, total_operation_time = 0.0;

    // Find method
    if(method == "EDGE"){
        image_processing_method = 1;
    } else if(method == "GAUSSIAN"){
        image_processing_method = 2;
    }

    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        // Start profiling execution time
        auto opencl_execution_time_start = std::chrono::high_resolution_clock::now();

        // Switch the image processing method
        switch (image_processing_method)
        {
        case 0:
            function_output = std::vector<unsigned char>(width * height * 4);
            controller.PerformCLImageGrayscaling(context, command_queue, kernel,
                &profiling_events, &input_data, &function_output,
                width, height, logger);
            break;
        
        case 1:
            function_output = std::vector<unsigned char>(width * height);
            controller.PerformCLImageEdgeDetection(context, command_queue, kernel,
                &profiling_events, &input_data, &function_output,
                width, height, logger);
            break;
        
        case 2:
            function_output = std::vector<unsigned char>(width * height * 4);
            controller.PerformCLGaussianBlur(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA, context, command_queue, kernel,
                &profiling_events, &input_data, &function_output,
                width, height, logger);
            break;

        default:
            logger.log("Unrecognised image processing method", Logger::LogLevel::ERROR);
            break;
        }
        
        // Log the method
        oss << "Performing OpenCL " << method << " on " << image_path << "...";
        logger.log(oss.str(), Logger::LogLevel::INFO);

        // End profiling execution time
        auto opencl_execution_time_end = std::chrono::high_resolution_clock::now();

        // Calculate total execution time(s)
        total_execution_time += std::chrono::duration<double, std::milli>(opencl_execution_time_end - opencl_execution_time_start).count();
        total_write_time += (profiling_events[1] - profiling_events[0]) * 1e-6;
        total_kernel_time += (profiling_events[3] - profiling_events[2]) * 1e-6;
        total_read_time += (profiling_events[5] - profiling_events[4]) * 1e-6;
        total_operation_time  = total_write_time + total_kernel_time + total_read_time;

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

    logger.log("OpenCL " + method + " conversion complete", Logger::LogLevel::INFO);

    // Calculate averages
    avg_opencl_execution_time = total_execution_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_write_time = total_write_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_execution_time = total_kernel_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_read_time = total_read_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_operation = total_operation_time / NUMBER_OF_ITERATIONS;

    return function_output;
}

std::vector<unsigned char> ProgramHandler::PerformOpenCL(Controller &controller, const cv::Mat &input_frame, cl_context *context, cl_command_queue *command_queue, cl_kernel *kernel,
    cl_int &width, cl_int &height, Logger &logger, std::string method)
{
    auto image_processing_method = 0;

    // Assert frame dimensions
    assert(input_frame.cols == width && input_frame.rows == height);

    // Initialise profiling variables
    std::vector<cl_ulong> profiling_events;

    // Initialise function output variable
    std::vector<unsigned char> function_output;

    // RGBA input data
    std::vector<unsigned char> input_data(input_frame.data, input_frame.data + input_frame.total() * 4);

    // Initialise output variables
    std::vector<float> output_data;

    // Find method
    if(method == "EDGE"){
        image_processing_method = 1;
    } else if(method == "GAUSSIAN"){
        image_processing_method = 2;
    }
    // Start profiling execution time
    auto opencl_execution_time_start = std::chrono::high_resolution_clock::now();

    // Switch the image processing method
    switch (image_processing_method)
    {
        case 0:
            function_output = std::vector<unsigned char>(width * height * 4);
            logger.log("Performing OpenCL Grayscaling...", Logger::LogLevel::INFO);
            controller.PerformCLImageGrayscaling(context, command_queue, kernel,
                &profiling_events, &input_data, &function_output,
                width, height, logger);
            logger.log("OpenCL Grayscale conversion complete", Logger::LogLevel::INFO);
            break;
        
        case 1:
            function_output = std::vector<unsigned char>(width * height);
            logger.log("Performing OpenCL Edge Detection...", Logger::LogLevel::INFO);
            controller.PerformCLImageEdgeDetection(context, command_queue, kernel,
                &profiling_events, &input_data, &function_output,
                width, height, logger);
            logger.log("OpenCL Edge Detection complete", Logger::LogLevel::INFO);
            break;
        
        case 2:
            function_output = std::vector<unsigned char>(width * height * 4);
            logger.log("Performing OpenCL Gaussian Blur...", Logger::LogLevel::INFO);
            controller.PerformCLGaussianBlur(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA, context, command_queue, kernel,
                &profiling_events, &input_data, &function_output,
                width, height, logger);
            logger.log("OpenCL Gaussian Blur complete", Logger::LogLevel::INFO);
            break;

        default:
            logger.log("Unrecognised image processing method", Logger::LogLevel::ERROR);
            break;
    }

    // End profiling execution time
    auto opencl_execution_time_end = std::chrono::high_resolution_clock::now();
    auto total_execution_time = std::chrono::duration<double, std::milli>(opencl_execution_time_end - opencl_execution_time_start).count();
    logger.log("OpenCL " +  method + " execution time: " + std::to_string(total_execution_time) + " ms", Logger::LogLevel::INFO);

    return function_output;
}
