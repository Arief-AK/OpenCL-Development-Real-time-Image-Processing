#include <chrono>
#include <iomanip>
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv.hpp>

#include <Controller.hpp>
#include <FileHandler.hpp>

// CONSTANTS
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 0

int NUMBER_OF_ITERATIONS = 100;

bool PERFORM_COMP = true;
bool SAVE_IMAGES = false;
bool DISPLAY_IMAGES = false;
bool DISPLAY_TERMINAL_RESULTS = true;

bool LOG_EVENTS = false;

std::string TEST_DIRECTORY = "images/";
std::string OUTPUT_FILE = "results.csv";

void InitLogger(Logger& logger){
    // Set the log file
    try{
       logger.setLogFile("Grayscale.log");
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error setting log file: " << e.what() << std::endl;
    }
}

void InitOpenCL(Controller& controller, cl_context* context, cl_command_queue* command_queue, cl_program* program, cl_kernel* kernel){
    // Get OpenCL platforms
    auto platforms = controller.GetPlatforms();
    for (auto && platform : platforms){
        controller.DisplayPlatformInformation(platform);
    }

    // Inform user of chosen indexes for platform and device
    std::cout << "\nApplication will use:\nPLATFORM INDEX:\t" << PLATFORM_INDEX << "\nDEVICE INDEX:\t" << DEVICE_INDEX << "\n" << std::endl;

    // Get intended device
    auto devices = controller.GetDevices(platforms[PLATFORM_INDEX]);

    // Check device image support
    cl_bool image_support = CL_FALSE;
    clGetDeviceInfo(devices[DEVICE_INDEX], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &image_support, nullptr);
    if (!image_support) {
        std::cerr << "Device does not support images." << std::endl;
        return;
    }

    // Get OpenCL mandatory properties
    cl_int err_num = 0;
    *context = controller.CreateContext(platforms[PLATFORM_INDEX], devices);
    *command_queue = controller.CreateCommandQueue(*context, devices[DEVICE_INDEX]);
    *program = controller.CreateProgram(*context, devices[DEVICE_INDEX], "grayscale.cl");
    *kernel = controller.CreateKernel(*program, "grayscale");
}

int main()
{
    std::cout << "Hello from EdgeDetection" << std::endl;
    return 0;
}