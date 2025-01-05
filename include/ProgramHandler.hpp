#include <Controller.hpp>
#include <Logger.hpp>

#include <opencv2/opencv.hpp>

class ProgramHandler
{
public:
    ProgramHandler(int number_of_iterations, bool log_events, bool display_images);

    std::vector<unsigned char> PerformOpenCL(Controller& controller, std::string image_path, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
    double& avg_opencl_execution_time, double& avg_opencl_kernel_write_time, double& avg_opencl_kernel_execution_time, double& avg_opencl_kernel_read_time,
    double& avg_opencl_kernel_operation, cl_int& width, cl_int& height, Logger& logger, std::string method);

    std::vector<unsigned char> PerformOpenCL(Controller& controller, const cv::Mat& input_frame, cl_context* context, cl_command_queue* command_queue, cl_kernel* kernel,
    double& avg_opencl_execution_time, double& avg_opencl_kernel_write_time, double& avg_opencl_kernel_execution_time, double& avg_opencl_kernel_read_time,
    double& avg_opencl_kernel_operation, cl_int& width, cl_int& height, Logger& logger, std::string method);

private:
    bool LOG_EVENTS;
    bool DISPLAY_IMAGES;

    int NUMBER_OF_ITERATIONS;

    void GetImageOpenCL(std::string image_path, std::vector<unsigned char> *input_data,
    cl_int* width, cl_int* height, Logger& logger);

    void GetMatrix(const cv::Mat& input_frame, std::vector<unsigned char> *input_data,
    cl_int* width, cl_int* height, Logger& logger);
};
