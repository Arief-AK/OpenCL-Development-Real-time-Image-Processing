#include <chrono>
#include <iomanip>
#include <opencv2/core/ocl.hpp>

#include <ProgramHandler.hpp>
#include <FileHandler.hpp>
#include <Comparator.hpp>

// CONSTANTS
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 1

int NUMBER_OF_ITERATIONS = 1;

bool PERFORM_COMP = true;
bool SAVE_IMAGES = false;
bool DISPLAY_IMAGES = false;
bool DISPLAY_TERMINAL_RESULTS = true;

bool LOG_EVENTS = false;

std::string IMAGES_DIRECTORY = "images/";
std::vector<std::string> METHOD = {"GRAYSCALE", "EDGE" , "GAUSSIAN"};

std::vector<std::string> GRAYSCALE_KERNELS = {"grayscale_images.cl", "grayscale_base.cl"};
std::vector<std::string> GAUSSIAN_KERNELS = {"gaussian_images.cl", "gaussian_base.cl"};
std::vector<std::string> EDGE_KERNELS = {"edge_images.cl", "edge_base.cl"};

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

    // Initialise output vector
    std::vector<unsigned char> output_data;

    // Load the images
    auto image_paths = file_handler.LoadImages(IMAGES_DIRECTORY);
    
    auto image_processing_method = 0;
    for (const auto& image_method : METHOD){
        // Initalise for each method
        program_handler.InitOpenCL(controller, &context, &command_queue, &program, &kernel, image_method);

        // Iterate through the images
        for(const auto& image_path: image_paths){
            // Initialise image variables
            cl_int width, height;

            // Find method
            if(image_method == "EDGE"){
                image_processing_method = 1;
            } else if(image_method == "GAUSSIAN"){
                image_processing_method = 2;
            }

            // Initialise timing variables
            double avg_opencl_execution_time = {};
            double avg_opencl_kernel_write_time = {};
            double avg_opencl_kernel_execution_time = {};
            double avg_opencl_kernel_read_time = {};
            double avg_opencl_kernel_operation_time = {};
            double avg_cpu_execution_time = {};

            // Perform OpenCL and get output data
            switch (image_processing_method)
            {
            case 0:
                output_data = program_handler.PerformOpenCL(controller, image_path, &context, &command_queue,&kernel,
                avg_opencl_execution_time, avg_opencl_kernel_write_time, avg_opencl_kernel_execution_time, avg_opencl_kernel_read_time, avg_opencl_kernel_operation_time,
                width, height, logger, "GRAYSCALE");
                break;

            case 1:
                output_data = program_handler.PerformOpenCL(controller, image_path, &context, &command_queue,&kernel,
                avg_opencl_execution_time, avg_opencl_kernel_write_time, avg_opencl_kernel_execution_time, avg_opencl_kernel_read_time, avg_opencl_kernel_operation_time,
                width, height, logger, "EDGE");
                break;

            case 2:
                output_data = program_handler.PerformOpenCL(controller, image_path, &context, &command_queue,&kernel,
                avg_opencl_execution_time, avg_opencl_kernel_write_time, avg_opencl_kernel_execution_time, avg_opencl_kernel_read_time, avg_opencl_kernel_operation_time,
                width, height, logger, "GAUSSIAN");
                break;
            
            default:
                break;
            }

            logger.PrintEndToEndExecutionTime("OpenCL", avg_opencl_execution_time);
            logger.PrintRawKernelExecutionTime(avg_opencl_kernel_execution_time, avg_opencl_kernel_write_time, avg_opencl_kernel_read_time, avg_opencl_kernel_operation_time);

            // Initialise output image
            cv::Mat opencl_output_image;
            auto image_support = controller.GetImageSupport();

            // Initialize output image based on image support
            switch (image_processing_method)
            {
            case 0:
                if (image_support == CL_TRUE) {
                    opencl_output_image = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char*>(output_data.data()));
                } else {
                    opencl_output_image = cv::Mat(height, width, CV_8UC4, const_cast<unsigned char*>(output_data.data()));
                }   
                break;

            case 1:
                opencl_output_image = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char*>(output_data.data()));
                break;

            case 2:
                opencl_output_image = cv::Mat(height, width, CV_8UC4, const_cast<unsigned char*>(output_data.data()));
                cv::cvtColor(opencl_output_image, opencl_output_image, cv::COLOR_RGBA2BGR);
                break;
            
            default:
                break;
            }

            if(DISPLAY_IMAGES){
                cv::imshow("OpenCL " + image_method + " window", opencl_output_image);
                cv::waitKey(0);
            }
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
    auto image_processing_method = 0;
    
    // Initialise frame and output variables
    cv::Mat frame;
    cv::Mat output_image;
    std::vector<uchar> output;

    std::string fps_text;

    bool kernel_initialised = false;

    // Initialise output image
    auto image_support = controller.GetImageSupport();

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
            image_processing_method += 1;
            if(image_processing_method == 4){
                image_processing_method = 0;
                kernel_initialised = false;
            }
            last_toggle_time = current_time; // Reset timer
        }

        switch (image_processing_method)
        {
        case 0:
            // Perform Gaussian
            if(!kernel_initialised){
                program_handler.InitOpenCL(controller, &context, &command_queue, &program, &kernel, "GAUSSIAN");
                kernel_initialised = true;
            }
            
            output = program_handler.PerformOpenCL(controller, frame, &context, &command_queue, &kernel,
            width, height, logger, "GAUSSIAN");

            fps_text = "FPS: " + std::to_string(static_cast<int>(fps)) + " [Gaussian]";
            output_image = cv::Mat(height, width, CV_8UC4, const_cast<unsigned char*>(output.data()));
            cv::cvtColor(output_image, output_image, cv::COLOR_RGBA2BGR);
            break;

        case 1:
            // Normal
            cv::cvtColor(frame, output_image, cv::COLOR_RGBA2BGR);
            fps_text = "FPS: " + std::to_string(static_cast<int>(fps)) + " [Normal]";
            kernel_initialised = false;
            break;

        case 2:
            // Perform grayscaling
            if(!kernel_initialised){
                program_handler.InitOpenCL(controller, &context, &command_queue, &program, &kernel, "GRAYSCALE");
                kernel_initialised = true;
            }
            
            output = program_handler.PerformOpenCL(controller, frame, &context, &command_queue,&kernel,
            width, height, logger, "GRAYSCALE");

            fps_text = "FPS: " + std::to_string(static_cast<int>(fps)) + " [Grayscale]";

            // Initialize output image based on image support
            switch (image_support)
            {
            case CL_TRUE:
                output_image = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char*>(output.data()));
                break;

            case CL_FALSE:
                output_image = cv::Mat(height, width, CV_8UC4, const_cast<unsigned char*>(output.data()));
                break;
            
            default:
                break;
            }
            break;

        case 3:
            // Perform Edge-detection
            if(!kernel_initialised){
                program_handler.InitOpenCL(controller, &context, &command_queue, &program, &kernel, "GRAYSCALE");
                kernel_initialised = true;
            }

            output = program_handler.PerformOpenCL(controller, frame, &context, &command_queue,&kernel,
            width, height, logger, "EDGE");

            fps_text = "FPS: " + std::to_string(static_cast<int>(fps)) + " [EDGE]";
            output_image = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char*>(output.data()));
            break;
        
        default:
            break;
        }

        // Display the output frame
        cv::putText(output_image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255), 2);
        cv::imshow("Camera Feed", output_image);

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

    // Initialise OpenCL platforms and devices
    program_handler.SetDeviceProperties(PLATFORM_INDEX, DEVICE_INDEX);
    program_handler.AddKernels(GRAYSCALE_KERNELS, "GRAYSCALE");
    program_handler.AddKernels(EDGE_KERNELS, "EDGE");
    program_handler.AddKernels(GAUSSIAN_KERNELS, "GAUSSIAN");

    PerformOnCamera(program_handler, logger);
    //PerformOnImages(program_handler, logger);
    return 0;
}
