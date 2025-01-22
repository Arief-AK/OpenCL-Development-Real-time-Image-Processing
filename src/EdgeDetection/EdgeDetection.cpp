#include <chrono>
#include <iomanip>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

#include <Controller.hpp>
#include <FileHandler.hpp>

// CONSTANTS
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 1

int NUMBER_OF_ITERATIONS = 100;

bool PERFORM_COMP = true;
bool SAVE_IMAGES = false;
bool DISPLAY_IMAGES = false;
bool DISPLAY_TERMINAL_RESULTS = true;

bool LOG_EVENTS = false;
bool BYPASS_IMAGE_SUPPORT = true;

std::string IMAGES_DIRECTORY = "images/";
std::string OUTPUT_FILE = "results.csv";
std::string KERNEL_NAME = "edge_images.cl";

void InitLogger(Logger& logger){
    // Set the log file
    try{
       logger.setLogFile("EdgeDetection.log");
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
    // Initialise variable
    std::string kernel_name = KERNEL_NAME;

    // Inform user of chosen indexes for platform and device
    std::cout << "\nApplication will use:\nPLATFORM INDEX:\t" << PLATFORM_INDEX << "\nDEVICE INDEX:\t" << DEVICE_INDEX << "\n" << std::endl;

    // Get intended device
    auto devices = controller.GetDevices(platforms[PLATFORM_INDEX]);

    char device_name[256];
    clGetDeviceInfo(devices[DEVICE_INDEX], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    std::cout << "Device name: " << device_name << std::endl;

    size_t maxWorkGroupSize;
    clGetDeviceInfo(devices[DEVICE_INDEX], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
    printf("Max Work Group Size: %zu\n", maxWorkGroupSize);

    // Check device image support
    cl_bool image_support = CL_FALSE;
    switch (BYPASS_IMAGE_SUPPORT)
    {
    case false:
        clGetDeviceInfo(devices[DEVICE_INDEX], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &image_support, nullptr);
        if (image_support == CL_FALSE) {
            kernel_name = "edge_base.cl";
            std::cout << "Device does not support images. Using buffers instead of image2D structures." << std::endl;
        }else{
            std::cout << "Device supports images." << std::endl;
        }
        break;

    case true:
        kernel_name = "edge_base.cl";
        std::cout << "Bypass image support is True. Using buffers instead of image2D structures." << std::endl;
        break;
    
    default:
        break;
    }
    controller.SetImageSupport(image_support);

    // Get OpenCL mandatory properties
    cl_int err_num = 0;
    *context = controller.CreateContext(platforms[PLATFORM_INDEX], devices);
    *command_queue = controller.CreateCommandQueue(*context, devices[DEVICE_INDEX]);
    *program = controller.CreateProgram(*context, devices[DEVICE_INDEX], kernel_name.c_str());
    *kernel = controller.CreateKernel(*program, "sobel_edge_detection");
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

std::vector<uchar> PerformOpenCL(Controller& controller, std::string image_path, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
    double& avg_opencl_execution_time, double& avg_opencl_kernel_write_time, double& avg_opencl_kernel_execution_time, double& avg_opencl_kernel_read_time,
    double& avg_opencl_kernel_operation, cl_int& width, cl_int& height, Logger& logger){
    
    std::ostringstream oss;
    oss << "Performing OpenCL Sobel edge-detection on " << image_path << "...";
    logger.log(oss.str(), Logger::LogLevel::INFO);

    // Initialise image variables
    std::vector<unsigned char> input_data;

    // Initialise profiling variables
    std::vector<cl_ulong> profiling_events;

    // Get image
    GetImageOpenCL(image_path, &input_data, &width, &height, logger);

    // Initialise output variables
    std::vector<unsigned char> output_data(width * height);

    // Initialise average variables
    double total_execution_time = 0.0;
    double total_write_time = 0.0, total_kernel_time = 0.0, total_read_time = 0.0, total_operation_time = 0.0;

    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        // Start profiling execution time
        auto opencl_execution_time_start = std::chrono::high_resolution_clock::now();

        // Perform edge detection
        controller.PerformCLImageEdgeDetection(image_path, context, command_queue, kernel,
        &profiling_events, &input_data, &output_data,
        width, height, logger);

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

    logger.log("OpenCL Edge detection conversion complete", Logger::LogLevel::INFO);

    // Calculate averages
    avg_opencl_execution_time = total_execution_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_write_time = total_write_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_execution_time = total_kernel_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_read_time = total_read_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_operation = total_operation_time / NUMBER_OF_ITERATIONS;

    return output_data;
}

cv::Mat PerformCPU(std::string image_path, double& avg_cpu_execution_time, Logger& logger){
    std::ostringstream oss;
    oss << "Performing CPU Sobel edge detection on " << image_path << "...";
    logger.log(oss.str(), Logger::LogLevel::INFO);
    
    // Initialise variables
    cv::Mat input_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if(input_image.empty()){
        if(input_image.channels() != 1)
            logger.log("Image is not in Grayscale!", Logger::LogLevel::ERROR);
        logger.log("Failed to read image", Logger::LogLevel::ERROR);
    }

    // Initialise output variable
    cv::Mat output_image;

    // Initialise average variables
    double total_execution_time = 0.0;

    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        // Loop through each pixel
        auto start = std::chrono::high_resolution_clock::now();
        
        cv::Mat sobel_x = (cv::Mat_<float>(3,3) << -1, 0, 1,
                                                   -2, 0, 2,
                                                   -1, 0, 1);

        cv::Mat sobel_y = (cv::Mat_<float>(3,3) << -1, -2, -1,
                                                   0, 0, 0,
                                                   1, 2, 1);

        // Gradient images
        cv::Mat gradient_x, gradient_y;

        // Apply Sobel kernels to compute gradients
        cv::filter2D(input_image, gradient_x, CV_32F, sobel_x); // X-gradient
        cv::filter2D(input_image, gradient_y, CV_32F, sobel_y); // Y-gradient

        // Compute the gradient magnitude
        cv::Mat magnitude;
        cv::magnitude(gradient_x, gradient_y, magnitude);

        // Normalise the magnitude to RGB [0 - 255] and convert into 8-bit
        // cv::normalize(magnitude, output_image, 0, 255, cv::NORM_MINMAX);
        magnitude.convertTo(output_image, CV_8UC1);

        auto end = std::chrono::high_resolution_clock::now();
        total_execution_time += std::chrono::duration<double, std::milli>(end - start).count();
    }

    logger.log("CPU Sobel edge-detection complete", Logger::LogLevel::INFO);

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

void PrintRawKernelExecutionTime(double& opencl_kernel_execution_time, double& opencl_kernel_write_time, double& opencl_kernel_read_time, double& opencl_kernel_operation_time, Logger& logger){
    logger.log("-------------------- START OF KERNEL EXEUCTION DETAILS --------------------", Logger::LogLevel::INFO);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(5) << "Kernel write time: " << opencl_kernel_write_time << " ms";
    logger.log(oss.str(), Logger::LogLevel::INFO);
    oss.str("");

    oss << std::fixed << std::setprecision(5) << "Kernel execution time: " << opencl_kernel_execution_time << " ms";
    logger.log(oss.str(), Logger::LogLevel::INFO);
    oss.str("");

    oss << std::fixed << std::setprecision(5) << "Kernel read time: " << opencl_kernel_read_time << " ms";
    logger.log(oss.str(), Logger::LogLevel::INFO);
    oss.str("");

    oss << std::fixed << std::setprecision(5) << "Kernel complete operation time: " << opencl_kernel_operation_time << " ms";
    logger.log(oss.str(), Logger::LogLevel::INFO);

    logger.log("-------------------- END OF KERNEL EXEUCTION DETAILS --------------------", Logger::LogLevel::INFO);
}

void PrintSummary(double& opencl_kernel_execution_time, double& opencl_kernel_write_time, double& opencl_kernel_read_time, double& opencl_execution_time, double& opencl_kernel_operation_time,
    double& cpu_execution_time, Logger& logger){
    if(DISPLAY_TERMINAL_RESULTS)
        std::cout << "\n **************************************** START OF OpenCL SUMMARY **************************************** " << std::endl;
    
    PrintEndToEndExecutionTime("OpenCL", opencl_execution_time, logger);
    PrintRawKernelExecutionTime(opencl_kernel_execution_time, opencl_kernel_write_time, opencl_kernel_read_time, opencl_kernel_operation_time, logger);
    
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

void WriteResultsToCSV(const std::string& filename, std::vector<std::tuple<std::string, std::string, std::string, int, double, double, double, double, double, double, double>>& results){
    std::ofstream file(filename);
    file << "Timestamp, Image, Resolution, Num_Iterations, avg_CPU_Time_ms, avg_OpenCL_Time_ms, avg_OpenCL_kernel_ms, avg_OpenCL_kernel_write_ms, avg_OpenCL_kernel_read_ms, avg_OpenCL_kernel_operation_ms, Error_MAE\n";
    for (const auto& [timestamp, image, resolution, num_iterations, avg_cpu_time, avg_opencl_time, avg_opencl_kernel_time, avg_opencl_kernel_write_time, avg_opencl_kernel_read_time, avg_opencl_kernel_operation_time, mae] : results) {
        file << timestamp << ", " << image << ", " << resolution << ", " << num_iterations << ", " << avg_cpu_time << ", " << avg_opencl_time << ", " << avg_opencl_kernel_time << ", "
        << avg_opencl_kernel_write_time << ", " << avg_opencl_kernel_read_time << ", " << avg_opencl_kernel_operation_time << ", " << mae << "\n";
    }
    file.close();
}

int main()
{
    std::cout << "Hello from EdgeDetection" << std::endl;
    cv::ocl::setUseOpenCL(false);

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
    std::vector<std::tuple<std::string, std::string, std::string, int, double, double, double, double, double, double, double>> comparison_results;

    // Initialise OpenCL platforms and devices
    InitOpenCL(controller, &context, &command_queue, &program, &kernel);

    // Load the images
    auto image_paths = file_handler.LoadImages(IMAGES_DIRECTORY);

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
        double avg_opencl_kernel_operation_time = {};
        double avg_cpu_execution_time = {};

        // Perform OpenCL and get output data
        auto output_data = PerformOpenCL(controller, image_path, &context, &command_queue,&kernel,
            avg_opencl_execution_time, avg_opencl_kernel_write_time, avg_opencl_kernel_execution_time, avg_opencl_kernel_read_time, avg_opencl_kernel_operation_time,
            width, height, logger);
        
        // Initialise output image
        cv::Mat opencl_output_image;
        auto image_support = controller.GetImageSupport();

        // Initialize output image based on image support
        opencl_output_image = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char*>(output_data.data()));
        
        // Save images
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
                                            avg_opencl_kernel_execution_time, 
                                            avg_opencl_kernel_write_time, avg_opencl_kernel_read_time,
                                            avg_opencl_kernel_operation_time,
                                            MAE);

            // Print summary
            PrintSummary(avg_opencl_kernel_execution_time, avg_opencl_kernel_write_time, avg_opencl_kernel_read_time, avg_opencl_execution_time, avg_opencl_kernel_operation_time,
                         avg_cpu_execution_time, logger);

            // Write results to CSV
            WriteResultsToCSV(OUTPUT_FILE, comparison_results);
        }

        if(DISPLAY_IMAGES){
            cv::imshow("OpenCL Sobel Edge Detection window", opencl_output_image);
            cv::imshow("CPU Grayscale window", cpu_output_image);
            cv::waitKey(0);
        }
    }

    std::cout << "Done!" << std::endl;
    return 0;
}