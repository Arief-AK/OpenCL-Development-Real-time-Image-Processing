#include "Comparator.hpp"

Comparator::Comparator(int num_methods, int num_iterations): m_num_methods{num_methods}, NUMBER_OF_ITERATIONS{num_iterations} {}

cv::Mat Comparator::PerformCPU_Grayscaling(std::string image_path, double &avg_cpu_execution_time, Logger &logger)
{
    cv::ocl::setUseOpenCL(false);
    
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

double Comparator::ComputeMAE(const cv::Mat &reference, const cv::Mat &result, Logger &logger)
{
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
