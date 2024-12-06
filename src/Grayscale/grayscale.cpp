#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <Controller.hpp>

void GetImage(std::vector<unsigned char> *input_data, std::vector<unsigned char> *output_data){
    // Load the input image using OpenCV
    std::string imagePath = "galaxy.jpg";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if(image.empty()){
        std::cerr << "Failed to load image" << std::endl;
    }
    cv::imshow("Display Window", image);
    cv::waitKey(0);

    // Convert to RGBA and get image dimensions
    cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
    auto width = image.cols;
    auto height = image.rows;

    // Flatten image into uchar array
    std::vector<unsigned char> _input_data(image.data, image.data + image.total() * 4);
    std::vector<unsigned char> _output_data(width * height);

    // Assign parameters
    *input_data = _input_data;
    *output_data = _output_data;
}

int main(int, char**){
    std::cout << "Hello, from OpenCL-Dev-Realtime-Image-Processing!\n";
    
    // Initialise variables
    std::vector<unsigned char> input_data;
    std::vector<unsigned char> output_data;

    // Get image
    GetImage(&input_data, &output_data);

    // Initialise OpenCL variables
    Controller controller;

    // Get OpenCL platforms
    auto platforms = controller.GetPlatforms();
    for (auto && platform : platforms){
        controller.DisplayPlatformInformation(platform);
    }

    return 0;
}
