#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <cmath>
#include <corecrt_math_defines.h>

#include <InfoPlatform.hpp>
#include <Logger.hpp>

class Controller
{
public:
    Controller();

    void CheckError(cl_int err, const char* name);
    
    std::vector<cl_platform_id> GetPlatforms();
    std::vector<cl_device_id> GetDevices(cl_platform_id platform);

    cl_context CreateContext(cl_platform_id platform, std::vector<cl_device_id> devices);
    cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device);
    cl_program CreateProgram(cl_context context, cl_device_id device, const char* filename);
    cl_kernel CreateKernel(cl_program program, const char* kernel_name);

    void DisplayPlatformInformation(cl_platform_id platform);
    void Cleanup(cl_context context = 0, cl_command_queue commandQueue = 0, cl_program program = 0, cl_kernel kernel = 0, cl_sampler sampler = 0, cl_mem* mem_objects = 0, int num_mem_objects = 0);

    void PerformCLImageGrayscaling(std::string image_path, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
        std::vector<cl_ulong>* profiling_events, std::vector<unsigned char>* input_data, std::vector<float>* output_data,
        cl_int& width, cl_int& height, Logger& logger);

    void PerformCLImageEdgeDetection(std::string image_path, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
        std::vector<cl_ulong>* profiling_events, std::vector<unsigned char>* input_data, std::vector<unsigned char>* output_data,
        cl_int& width, cl_int& height, Logger& logger);

    void PerformCLGaussianBlur(int& kernel_size, float& kernel_sigma, std::string image_path, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
        std::vector<cl_ulong>* profiling_events, std::vector<unsigned char>* input_data, std::vector<unsigned char>* output_data,
        cl_int& width, cl_int& height, Logger& logger);

private:
    cl_uint num_platforms, num_devices;

    std::vector<float> GenerateGaussianKernel(int kernel_size, float sigma);
};

#endif // CONTROLLER_H