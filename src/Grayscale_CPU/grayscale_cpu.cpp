#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

void GetImage(cv::Mat* input_image){
    // Load the input image using OpenCV
    std::string imagePath = "galaxy.jpg";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if(image.empty()){
        std::cerr << "Failed to load image" << std::endl;
    }
    cv::imshow("Display Window", image);
    cv::waitKey(0);

    *input_image = image;
}

int main()
{
    std::cout << "Hello from grayscale_CPU" << std::endl;
    
    // Initialise variables
    cv::Mat input_image;
    cv::Mat output_image;

    // Get image using OpenCV
    GetImage(&input_image);

    // Convert the image to grayscale and perform execution time profiling
    auto start = std::chrono::high_resolution_clock::now();
    cv::cvtColor(input_image, output_image, cv::COLOR_RGBA2GRAY);
    auto end = std::chrono::high_resolution_clock::now();

    // Output the image
    cv::imwrite("grayscale_cpu.jpg", output_image);
    cv::imshow("Grayscale Window", output_image);
    cv::waitKey(0);

    // Print results
    auto execution_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "\n-------------------- START OF KERNEL EXEUCTION DETAILS --------------------" << std::endl;
    std::cout << "Grayscale conversion execution time: " << execution_time << " ms" << std::endl;
    std::cout << "-------------------- END OF KERNEL EXEUCTION DETAILS --------------------" << std::endl;
    
    return 0;
}