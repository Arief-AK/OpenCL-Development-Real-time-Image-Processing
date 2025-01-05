#include <chrono>
#include <iomanip>
#include <opencv2/core/ocl.hpp>
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
std::string KERNEL_NAME = "grayscale_images.cl";

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
    clGetDeviceInfo(devices[DEVICE_INDEX], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &image_support, nullptr);
    if (image_support == CL_FALSE) {
        kernel_name = "grayscale_base.cl";
        std::cout << "Device does not support images. Using buffers instead of image2D structures." << std::endl;
    }else{
        std::cout << "Device supports images." << std::endl;
    }
    controller.SetImageSupport(image_support);

    // Get OpenCL mandatory properties
    cl_int err_num = 0;
    *context = controller.CreateContext(platforms[PLATFORM_INDEX], devices);
    *command_queue = controller.CreateCommandQueue(*context, devices[DEVICE_INDEX]);
    *program = controller.CreateProgram(*context, devices[DEVICE_INDEX], kernel_name.c_str());
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

std::vector<uchar> PerformOpenCL(Controller& controller, std::string image_path, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
    double& avg_opencl_execution_time, double& avg_opencl_kernel_write_time, double& avg_opencl_kernel_execution_time, double& avg_opencl_kernel_read_time,
    double& avg_opencl_kernel_operation, cl_int& width, cl_int& height, Logger& logger, std::string method){
    
    auto image_processing_method = 0;
    std::ostringstream oss;

    // Initialise image variables
    std::vector<unsigned char> input_data;

    // Initialise profiling variables
    std::vector<cl_ulong> profiling_events;

    // Get image
    GetImageOpenCL(image_path, &input_data, &width, &height, logger);

    // Initialise output variables
    std::vector<float> output_data(width * height * 4);
    
    // Initialise output image (grayscale)
    std::vector<unsigned char> grayscale_output(width * height * 4);

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

        // Swicth the image processing method
        switch (image_processing_method)
        {
        case 0:
            controller.PerformCLImageGrayscaling(image_path, context, command_queue, kernel,
                &profiling_events, &input_data, &output_data,
                width, height, logger);
            break;
        
        case 1:
            break;
        
        case 2:
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

    logger.log("OpenCL Grayscale conversion complete", Logger::LogLevel::INFO);

    // Calculate averages
    avg_opencl_execution_time = total_execution_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_write_time = total_write_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_execution_time = total_kernel_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_read_time = total_read_time / NUMBER_OF_ITERATIONS;
    avg_opencl_kernel_operation = total_operation_time / NUMBER_OF_ITERATIONS;

    for (size_t i = 0; i < (width * height * 4); i++) {
        grayscale_output[i] = static_cast<unsigned char>(output_data[i] * 255.0f); // Extract and scale grayscale
    }

    return grayscale_output;
}

cv::Mat PerformCPU(std::string image_path, double& avg_cpu_execution_time, Logger& logger){
    
    std::ostringstream oss;
    oss << "Performing CPU grayscaling on " << image_path << "...";
    logger.log(oss.str(), Logger::LogLevel::INFO);
    
    // Initialise variables
    cv::Mat input_image = cv::imread(image_path);
    if(input_image.empty() || input_image.channels() != 3){
        logger.log("Failed to read image", Logger::LogLevel::ERROR);
    }

    // Initialise output image
    auto output_image = cv::Mat(input_image.rows, input_image.cols, CV_8UC1);

    // Initialise average variables
    double total_execution_time = 0.0;

    for(int i = 0; i < NUMBER_OF_ITERATIONS; i++){
        // Loop through each pixel
        auto start = std::chrono::high_resolution_clock::now();

        // Process each pixel directly
        for (int row = 0; row < input_image.rows; row++) {
            // Pointers to the start of the row in input and output images
            const uchar* input_row = input_image.ptr<uchar>(row);
            uchar* output_row = output_image.ptr<uchar>(row);

            for (int col = 0; col < input_image.cols; col++) {
                // Calculate the grayscale value using BGR components
                int b = input_row[col * 3];
                int g = input_row[col * 3 + 1];
                int r = input_row[col * 3 + 2];

                uchar gray_value = static_cast<uchar>(0.299 * r + 0.587 * g + 0.114 * b);

                // Assign the grayscale value to the output image
                output_row[col] = gray_value;
            }
        }
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

double ComputeMAE(const cv::Mat& reference, const cv::Mat& result, Logger& logger){
    if (result.size() != reference.size()) {
        logger.log("Images have different sizes", Logger::LogLevel::ERROR);
        return -1; // Return error code for size mismatch
    }

    cv::Mat refConverted, resConverted;

    // If channel counts differ, normalize them to grayscale for comparison
    if (result.channels() != reference.channels()) {
        logger.log("Reference image has " + std::to_string(reference.channels()) + 
                   " channels, while result image has " + std::to_string(result.channels()) + " channels", 
                   Logger::LogLevel::INFO);

        // Convert both images to grayscale for comparison
        if (reference.channels() == 4) {
            cv::cvtColor(reference, refConverted, cv::COLOR_BGR2GRAY);
            logger.log("Converted reference image to grayscale", Logger::LogLevel::INFO);
        } else {
            refConverted = reference; // Already grayscale
        }

        if (result.channels() == 4) {
            cv::cvtColor(result, resConverted, cv::COLOR_RGBA2GRAY);
            logger.log("Converted result image to grayscale", Logger::LogLevel::INFO);
        } else {
            resConverted = result; // Already grayscale
        }
    } else {
        // If channels are the same, no conversion is needed
        refConverted = reference;
        resConverted = result;
    }

    // Compute the absolute difference
    cv::Mat difference;
    cv::absdiff(refConverted, resConverted, difference);

    // Compute and return the mean absolute error (MAE)
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

std::vector<uchar> ProcessFrameOpenCL(Controller& controller, const cv::Mat& input_frame, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel, cl_int width, cl_int height) {
    // Process frame using OpenCL grayscale
    auto width = static_cast<cl_int>(input_frame.cols);
    auto height = static_cast<cl_int>(input_frame.rows);
    
    std::vector<unsigned char> input_data(input_frame.data, input_frame.data + input_frame.total() * 4); // RGBA data
    std::vector<float> output_data(width * height * 4); // Output buffer for grayscale

    // Process the frame using OpenCL
    //PerformOpenCL

    // Convert output to uchar for display
    std::vector<uchar> grayscale_output(width * height);
    for (size_t i = 0; i < width * height; i++) {
        grayscale_output[i] = static_cast<unsigned char>(output_data[i] * 255.0f);
    }

    return grayscale_output;
}

int main() {
    std::cout << "Hello from RealtimeImage Module!" << std::endl;
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

    // Initialise OpenCL platforms and devices
    InitOpenCL(controller, &context, &command_queue, &program, &kernel);

    // Camera pipeline
    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }

    // Get FPS
    auto fps = cap.get(cv::CAP_PROP_FPS);
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert frame to RGBA
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);

        
        //std::vector<uchar> grayscale_output = PerformOpenCL

        // Convert grayscale output to cv::Mat for display
        cv::Mat grayscale_image(height, width, CV_8UC1, grayscale_output.data());

        // Display the grayscaled frame
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(grayscale_image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255), 2);

        cv::imshow("Grayscale Camera Feed", grayscale_image);
        if (cv::waitKey(1) == 27) break; // Exit on 'Esc' key
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
