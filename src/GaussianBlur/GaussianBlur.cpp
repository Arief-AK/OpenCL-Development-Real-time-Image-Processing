#include <chrono>
#include <iomanip>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

#include <Controller.hpp>
#include <FileHandler.hpp>

// CONSTANTS
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 0

int NUMBER_OF_ITERATIONS = 5;

int GAUSSIAN_KERNEL_SIZE = 5;
float GAUSSIAN_SIGMA = 1.5f;

bool PERFORM_COMP = true;
bool SAVE_IMAGES = false;
bool DISPLAY_IMAGES = false;
bool DISPLAY_TERMINAL_RESULTS = true;

bool LOG_EVENTS = false;

std::string TEST_DIRECTORY = "images/";
std::string KERNEL_FILE = "gaussian_blur.cl";
std::string KERNEL_NAME = "gaussian_blur";
std::string OUTPUT_FILE = "results.csv";

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
    *program = controller.CreateProgram(*context, devices[DEVICE_INDEX], KERNEL_FILE.c_str());
    *kernel = controller.CreateKernel(*program, KERNEL_NAME.c_str());
}

void GetImageOpenCL(std::string image_path, std::vector<unsigned char>& input_data,
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

    // Convert image to RGBA format
    cv::Mat rgba_image;
    cv::cvtColor(image, rgba_image, cv::COLOR_BGR2RGBA);

    // Get image dimensions
    *width = image.cols;
    *height = image.rows;

    // Flatten image into uchar array
    input_data.assign(rgba_image.data, rgba_image.data + rgba_image.total() * rgba_image.channels());
}

std::vector<uchar> PerformOpenCL(Controller& controller, std::string image_path, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
    double& avg_opencl_execution_time, double& avg_opencl_kernel_write_time, double& avg_opencl_kernel_execution_time, double& avg_opencl_kernel_read_time,
    double& avg_opencl_kernel_operation, cl_int& width, cl_int& height, Logger& logger){
    
    std::ostringstream oss;
    oss << "Performing OpenCL Gaussian Blur on " << image_path << "...";
    logger.log(oss.str(), Logger::LogLevel::INFO);

    // Initialise image variables
    std::vector<unsigned char> input_data;

    // Initialise profiling variables
    std::vector<cl_ulong> profiling_events;

    // Get image
    GetImageOpenCL(image_path, input_data, &width, &height, logger);

    // Initialise output variables
    std::vector<unsigned char> output_data(width * height * 4); // RGBA format

    // Initialise average variables
    double total_execution_time = 0.0;
    double total_write_time = 0.0, total_kernel_time = 0.0, total_read_time = 0.0, total_operation_time = 0.0;

    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        // Start profiling execution time
        auto opencl_execution_time_start = std::chrono::high_resolution_clock::now();

        // TODO: Perform OpenCL Gaussian Blur
        controller.PerformCLGaussianBlur(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA, image_path, context, command_queue, kernel,
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

cv::Mat PerformCPU(Controller& controller, std::string image_path, double& avg_cpu_execution_time, Logger& logger){
    std::ostringstream oss;
    oss << "Performing CPU Gaussian Blur on " << image_path << "...";
    logger.log(oss.str(), Logger::LogLevel::INFO);
    
    // Initialise variables
    cv::Mat input_image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if(input_image.empty()){
        logger.log("Failed to read image", Logger::LogLevel::ERROR);
    }

    // Convert image to RGBA format
    cv::Mat rgba_image;
    cv::cvtColor(input_image, rgba_image, cv::COLOR_BGR2RGBA);
    if(rgba_image.channels() != 4)
        logger.log("Image is not in RGBA!", Logger::LogLevel::ERROR);

    // Get image dimensions
    auto width = rgba_image.cols;
    auto height = rgba_image.rows;

    // Initialise output image
    cv::Mat output_image(height, width, CV_8UC4);

    // Disable OpenCL in OpenCV to ensure CPU-only execution
    cv::ocl::setUseOpenCL(false);
    std::ostringstream oss1;
    oss1 << "OpenCL Enabled: " << cv::ocl::useOpenCL();
    logger.log(oss1.str(), Logger::LogLevel::INFO);

    // Initialise average variables
    double total_execution_time = 0.0;

    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        // Loop through each pixel
        auto start = std::chrono::high_resolution_clock::now();

        // Apply Gaussian Blur
        cv::GaussianBlur(rgba_image, output_image, cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), (double)GAUSSIAN_SIGMA);

        auto end = std::chrono::high_resolution_clock::now();
        total_execution_time += std::chrono::duration<double, std::milli>(end - start).count();
    }

    logger.log("CPU Gaussian Blur complete", Logger::LogLevel::INFO);

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

int main() {
    std::cout << "Hello from GaussianBlur!" << std::endl;

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
        double avg_opencl_kernel_operation_time = {};
        double avg_cpu_execution_time = {};

        // Perform OpenCL and get output data
        auto output_data = PerformOpenCL(controller, image_path, &context, &command_queue,&kernel,
            avg_opencl_execution_time, avg_opencl_kernel_write_time, avg_opencl_kernel_execution_time, avg_opencl_kernel_read_time, avg_opencl_kernel_operation_time,
            width, height, logger);
        
        cv::Mat opencl_output_image(height, width, CV_8UC4, const_cast<unsigned char*>(output_data.data()));
        cv::cvtColor(opencl_output_image, opencl_output_image, cv::COLOR_RGBA2BGR); // Convert to BGR for OpenCV
        SaveImages(image_path, opencl_output_image);

        // Perform OpenCL vs CPU comparison
        if(PERFORM_COMP){
            cpu_output_image = PerformCPU(controller, image_path, avg_cpu_execution_time, logger);
            cv::cvtColor(cpu_output_image, cpu_output_image, cv::COLOR_RGBA2BGR); // Convert to BGR for OpenCV

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
            cv::imshow("OpenCL Gaussian Blur window", opencl_output_image);
            cv::imshow("CPU Gaussian Blur window", cpu_output_image);
            cv::waitKey(0);
        }
    }

    std::cout << "Done!" << std::endl;
    return 0;
}