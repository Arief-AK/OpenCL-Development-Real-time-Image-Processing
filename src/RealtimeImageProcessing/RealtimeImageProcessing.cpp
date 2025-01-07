#include <chrono>
#include <iomanip>
#include <opencv2/core/ocl.hpp>

#include <ProgramHandler.hpp>
#include <FileHandler.hpp>
#include <Comparator.hpp>

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

void PerformOnImages(ProgramHandler& program_handler, Logger& logger){
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
    program_handler.SetKernelProperties(KERNEL_NAME, PLATFORM_INDEX, DEVICE_INDEX);
    program_handler.InitOpenCL(controller, &context, &command_queue, &program, &kernel);

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
        auto output_data = program_handler.PerformOpenCL(controller, image_path, &context, &command_queue,&kernel,
            avg_opencl_execution_time, avg_opencl_kernel_write_time, avg_opencl_kernel_execution_time, avg_opencl_kernel_read_time, avg_opencl_kernel_operation_time,
            width, height, logger, "GRAYSCALE");

        logger.PrintEndToEndExecutionTime("OpenCL", avg_opencl_execution_time);
        logger.PrintRawKernelExecutionTime(avg_opencl_kernel_execution_time, avg_opencl_kernel_write_time, avg_opencl_kernel_read_time, avg_opencl_kernel_operation_time);

        // Initialise output image
        cv::Mat opencl_output_image;
        auto image_support = controller.GetImageSupport();

        // Initialize output image based on image support
        if (image_support == CL_TRUE) {
            opencl_output_image = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char*>(output_data.data()));
        } else {
            opencl_output_image = cv::Mat(height, width, CV_8UC4, const_cast<unsigned char*>(output_data.data()));
        }

        if(DISPLAY_IMAGES){
            cv::imshow("OpenCL Grayscale window", opencl_output_image);
            cv::waitKey(0);
        }
    }

}

void PerformOnCamera(ProgramHandler& program_handler, Logger& logger){
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
    program_handler.SetKernelProperties(KERNEL_NAME, PLATFORM_INDEX, DEVICE_INDEX);
    program_handler.InitOpenCL(controller, &context, &command_queue, &program, &kernel);

    // Camera pipeline
    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1024, height=768, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        exit(1);
    }
    
    // Get FPS
    auto fps = cap.get(cv::CAP_PROP_FPS);

    // Timer variables
    auto last_toggle_time = std::chrono::high_resolution_clock::now();
    bool is_grayscale = true;
    
    // Initialise frame
    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert frame to RGBA
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGBA);
        auto width = frame.cols;
        auto height = frame.rows;
        cv::resize(frame, frame, cv::Size(width, height));

        // Check timer and toggle mode every 10 seconds
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_toggle_time).count();
        if (elapsed_time >= 10) {
            is_grayscale = !is_grayscale; // Toggle mode
            last_toggle_time = current_time; // Reset timer
        }

        if(is_grayscale){
            std::vector<uchar> grayscale_output = program_handler.PerformOpenCL(controller, frame, &context, &command_queue,&kernel,
            width, height, logger, "GRAYSCALE");

            // Initialise output image
            cv::Mat grayscale_image;
            auto image_support = controller.GetImageSupport();

            // Initialize output image based on image support
            if (image_support == CL_TRUE) {
                grayscale_image = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char*>(grayscale_output.data()));
            } else {
                grayscale_image = cv::Mat(height, width, CV_8UC4, const_cast<unsigned char*>(grayscale_output.data()));
            }

            // Display the grayscaled frame
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps)) + " [Grayscale]";
            cv::putText(grayscale_image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255), 2);

            cv::imshow("Camera Feed", grayscale_image);
        } else {
            // Display original frame
            cv::Mat original_frame;
            cv::cvtColor(frame, original_frame, cv::COLOR_RGBA2BGR);

            // Overlay FPS
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps)) + " [Normal]";
            cv::putText(original_frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            // Display normal frame
            cv::imshow("Camera Feed", original_frame);
        }

        if (cv::waitKey(1) == 27) break; // Exit on 'Esc' key
    }

    // Cleanup
    cap.release();
    cv::destroyAllWindows();
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}

int main() {
    std::cout << "Hello from RealtimeImage Module!" << std::endl;
    cv::ocl::setUseOpenCL(false);

    // Initialise (singleton) Logger
    Logger& logger = Logger::getInstance();
    auto program_handler = ProgramHandler(NUMBER_OF_ITERATIONS, LOG_EVENTS, DISPLAY_IMAGES, DISPLAY_TERMINAL_RESULTS);
    program_handler.InitLogger(logger, Logger::LogLevel::INFO);

    //PerformOnCamera(program_handler, logger);
    PerformOnImages(program_handler, logger);
    return 0;
}
