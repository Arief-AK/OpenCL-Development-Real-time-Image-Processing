#ifndef COMPARATOR_H
#define COMPARATOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include <Logger.hpp>

class Comparator
{
public:
    Comparator(int num_methods, int num_iterations);

    cv::Mat PerformCPU_Grayscaling(std::string image_path, double& avg_cpu_execution_time, Logger& logger);

private:
    int m_num_methods;
    int NUMBER_OF_ITERATIONS;

    double ComputeMAE(const cv::Mat& reference, const cv::Mat& result, Logger& logger);
};

#endif // COMPARATOR_H