#include <Controller.hpp>
#include <Logger.hpp>

#include <opencv2/opencv.hpp>

class ProgramHandler
{
public:
    ProgramHandler(int number_of_iterations, bool log_events, bool display_images, bool display_terminal_results, int gaussian_kernel_size = 17, float gaussian_sigma = 6.0f);

    void InitLogger(Logger& logger, Logger::LogLevel level, bool save_to_file);
    void InitOpenCL(Controller& controller, cl_context* context, cl_command_queue* command_queue, cl_program* program, cl_kernel* kernel, std::string method, Logger& logger);

    void AddKernels(std::vector<std::string> kernels, std::string kernel_index);
    void SetDeviceProperties(int platform_index, int device_index);

    std::vector<unsigned char> PerformOpenCL(Controller& controller, std::string image_path, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
    double& avg_opencl_execution_time, double& avg_opencl_kernel_write_time, double& avg_opencl_kernel_execution_time, double& avg_opencl_kernel_read_time,
    double& avg_opencl_kernel_operation, cl_int& width, cl_int& height, Logger& logger, std::string method);

    std::vector<unsigned char> PerformOpenCL(Controller& controller, const cv::Mat& input_frame, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
    cl_int& width, cl_int& height, Logger& logger, std::string method);

private:
    bool LOG_EVENTS;
    bool DISPLAY_IMAGES;
    bool DISPLAY_TERMINAL_RESULTS;

    int NUMBER_OF_ITERATIONS;
    int PLATFORM_INDEX;
    int DEVICE_INDEX;
    
    int GAUSSIAN_KERNEL_SIZE;
    float GAUSSIAN_SIGMA;

    std::vector<std::string> METHOD;
    std::map<std::string, std::vector<std::string>> KERNELS;

    void GetImageOpenCL(std::string image_path, std::vector<unsigned char> *input_data,
    cl_int* width, cl_int* height, Logger& logger);

    void GetMatrix(const cv::Mat& input_frame, std::vector<unsigned char> *input_data,
    cl_int* width, cl_int* height, Logger& logger);
};
