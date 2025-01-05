#include "ProgramHandler.hpp"

ProgramHandler::ProgramHandler(int number_of_iterations, bool log_events, bool display_images): NUMBER_OF_ITERATIONS{number_of_iterations}, LOG_EVENTS{log_events}, DISPLAY_IMAGES{display_images} {}

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
            controller.PerformCLImageGrayscaling(context, command_queue, kernel,
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

std::vector<unsigned char> ProgramHandler::PerformOpenCL(Controller &controller, const cv::Mat &input_frame, cl_context *context, cl_command_queue *command_queue, cl_kernel *kernel,
    double &avg_opencl_execution_time, double &avg_opencl_kernel_write_time, double &avg_opencl_kernel_execution_time, double &avg_opencl_kernel_read_time,
    double &avg_opencl_kernel_operation, cl_int &width, cl_int &height, Logger &logger, std::string method)
{
    auto image_processing_method = 0;
    std::ostringstream oss;

    // Initialise image variables
    std::vector<unsigned char> input_data;

    // Initialise profiling variables
    std::vector<cl_ulong> profiling_events;

    // Get matrix
    GetMatrix(input_frame, &input_data, &width, &height, logger);

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
            logger.log("Performing OpenCL Grayscaling...", Logger::LogLevel::INFO);
            controller.PerformCLImageGrayscaling(context, command_queue, kernel,
                &profiling_events, &input_data, &output_data,
                width, height, logger);
            break;
        
        case 1:
            logger.log("Performing OpenCL Edge Detection...", Logger::LogLevel::INFO);
            break;
        
        case 2:
            logger.log("Performing OpenCL Gaussian Blur...", Logger::LogLevel::INFO);
            break;

        default:
            logger.log("Unrecognised image processing method", Logger::LogLevel::ERROR);
            break;
        }

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
