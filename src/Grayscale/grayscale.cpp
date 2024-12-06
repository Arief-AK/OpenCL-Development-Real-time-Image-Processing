#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int, char**){
    std::cout << "Hello, from OpenCL-Dev-Realtime-Image-Processing!\n";
    
    std::string imagePath = "galaxy.jpg";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::imshow("Display Window", image);
    cv::waitKey(0);
    
    return 0;
}
