#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <utility>

#define _USE_MATH_DEFINES
#include <cmath>

#include <InfoPlatform.hpp>
#include <Logger.hpp>

class Controller
{
public:
    Controller();

    void CheckError(cl_int err, const char* name);
    
    std::vector<cl_platform_id> GetPlatforms();
    std::vector<cl_device_id> GetDevices(cl_platform_id platform);

    cl_bool GetImageSupport();
    void SetImageSupport(cl_bool image_support);

    cl_context CreateContext(cl_platform_id platform, std::vector<cl_device_id> devices);
    cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device);
    cl_program CreateProgram(cl_context context, cl_device_id device, const char* filename);
    cl_kernel CreateKernel(cl_program program, const char* kernel_name);

    void DisplayPlatformInformation(cl_platform_id platform);
    void Cleanup(cl_context context = 0, cl_command_queue commandQueue = 0, cl_program program = 0, cl_kernel kernel = 0, cl_sampler sampler = 0, cl_mem* mem_objects = 0, int num_mem_objects = 0);

    void PerformCLImageGrayscaling(cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
    std::vector<cl_ulong>* profiling_events, std::vector<unsigned char>* input_data, std::vector<unsigned char>* output_data,
    cl_int& width, cl_int& height, Logger& logger);

    void PerformCLImageEdgeDetection(cl_context *context, cl_command_queue *command_queue, cl_kernel *kernel,
    std::vector<cl_ulong> *profiling_events, std::vector<unsigned char> *input_data, std::vector<unsigned char> *output_data,
    cl_int &width, cl_int &height, Logger &logger);

    void PerformCLGaussianBlur(int& kernel_size, float& kernel_sigma, cl_context *context, cl_command_queue *command_queue, cl_kernel *kernel,
    std::vector<cl_ulong> *profiling_events, std::vector<unsigned char> *input_data, std::vector<unsigned char> *output_data,
    cl_int &width, cl_int &height, Logger &logger);

private:
    cl_uint num_platforms, num_devices;
    cl_bool m_image_support;

    void _profileEvent(cl_event& event, std::vector<cl_ulong> *profiling_events);
    std::vector<unsigned char> ConvertToUChar(const std::vector<float>& input_data);

    std::pair<cl_mem, cl_mem> _initGrayscaleBuffers(cl_context* context, cl_command_queue* command_queue, std::vector<unsigned char>* input_data, cl_int width, cl_int height, cl_event* write_event, Logger& logger);
    std::pair<cl_mem, cl_mem> _initGrayscaleImage2D(cl_context* context, cl_command_queue* command_queue, std::vector<unsigned char>* input_data, cl_int width, cl_int height, cl_event* write_event, Logger& logger);

    std::pair<cl_mem, cl_mem> _initEdgeDetectionBuffers(cl_context* context, cl_command_queue* command_queue, std::vector<unsigned char>* input_data, cl_int width, cl_int height, cl_event* write_event, Logger& logger);
    std::pair<cl_mem, cl_mem> _initEdgeDetectionImage2D(cl_context* context, cl_command_queue* command_queue, std::vector<unsigned char>* input_data, cl_int width, cl_int height, cl_event* write_event, Logger& logger);

    std::pair<cl_mem, cl_mem> _initGaussianBlurBuffers(cl_context* context, cl_command_queue* command_queue, std::vector<unsigned char>* input_data, cl_int width, cl_int height, cl_event* write_event, Logger& logger);
    std::pair<cl_mem, cl_mem> _initGaussianBlurImage2D(cl_context* context, cl_command_queue* command_queue, std::vector<unsigned char>* input_data, cl_int width, cl_int height, cl_event* write_event, Logger& logger);

    std::vector<float> _GenerateGaussianKernelBuffers(int kernel_size, float sigma);
    std::vector<float> _GenerateGaussianKernelImage2D(int kernel_size, float sigma);
    std::vector<float> _GenerateGausianKernel(int kernel_size, float sigma);
};

#endif // CONTROLLER_H