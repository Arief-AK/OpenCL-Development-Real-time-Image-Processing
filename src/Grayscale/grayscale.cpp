#include <opencv2/opencv.hpp>
#include <iomanip>

#include <Controller.hpp>

// CONSTANTS
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 0

bool TIME_KERNEL = true;

void GetImage(std::vector<unsigned char> *input_data, std::vector<unsigned char> *output_data, cl_int* width, cl_int* height){
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
    *width = image.cols;
    *height = image.rows;

    // Flatten image into uchar array
    std::vector<unsigned char> _input_data(image.data, image.data + image.total() * 4);
    std::vector<unsigned char> _output_data(*width * *height);

    // Assign parameters
    *input_data = _input_data;
    *output_data = _output_data;
}

int main(int, char**){
    std::cout << "Hello, from OpenCL-Dev-Realtime-Image-Processing!\n";
    
    // Initialise variables
    std::vector<unsigned char> input_data;
    std::vector<unsigned char> output_data;
    cl_int width, height;

    // Get image
    GetImage(&input_data, &output_data, &width, &height);

    // Initialise OpenCL variables
    Controller controller;

    // Get OpenCL platforms
    auto platforms = controller.GetPlatforms();
    for (auto && platform : platforms){
        controller.DisplayPlatformInformation(platform);
    }

    // Inform user of chosen indexes for platform and device
    std::cout << "\nApplication will use:\nPLATFORM INDEX:\t" << PLATFORM_INDEX << "\nDEVICE INDEX:\t" << DEVICE_INDEX << "\n" << std::endl;

    // Get intended device
    auto devices = controller.GetDevices(platforms[PLATFORM_INDEX]);

    // Get OpenCL mandatory properties
    cl_int err_num = 0;
    auto context = controller.CreateContext(platforms[PLATFORM_INDEX], devices);
    auto command_queue = controller.CreateCommandQueue(context, devices[DEVICE_INDEX]);
    auto program = controller.CreateProgram(context, devices[DEVICE_INDEX], "grayscale.cl");
    auto kernel = controller.CreateKernel(program, "grayscale");

    // Create buffers
    auto input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * input_data.size(), input_data.data(), &err_num);
    if(err_num != CL_SUCCESS){
        std::cerr << "Failed to create input buffer" << std::endl;
    }
    auto output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * output_data.size(), nullptr, &err_num);
    if(err_num != CL_SUCCESS){
        std::cerr << "Failed to create output buffer" << std::endl;
    }

    // Assign the kernel arguments
    err_num = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err_num |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err_num |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    err_num |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    std::cout << "Successfully assigned kernel arguments" << std::endl;

    // Initialise the global work size and execute kernel
    size_t global_work_size[] = {static_cast<size_t>(width), static_cast<size_t>(height)};

    // Create an event
    cl_event event;

    err_num = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, &event);
    if(err_num != CL_SUCCESS){
        std::cerr << "Failed when executing kernel" << std::endl;
    }

    // Wait for the event to complete
    clWaitForEvents(1, &event);

    // Get the timing
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

    // Get the duration
    if(TIME_KERNEL){
        double time_ms = (end - start) * 1e-6;
        std::cout << "\n-------------------- START OF KERNEL EXEUCTION DETAILS --------------------" << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "Kernel execution time: " << time_ms << " ms" << std::endl;
        std::cout << "-------------------- END OF KERNEL EXEUCTION DETAILS --------------------" << std::endl;
        std::cout << std::endl;
    }

    // Read the buffer
    err_num  = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, sizeof(unsigned char) * output_data.size(), output_data.data(), 0, nullptr, nullptr);
    if(err_num != CL_SUCCESS){
        std::cerr << "Failed to read buffer" << std::endl;
    }

    // Convert output to OpenCV matrix
    cv::Mat output_image(height, width, CV_8UC1, output_data.data());
    cv::imwrite("grayscale_galaxy.jpg", output_image);
    std::cout << "Grayscale conversion complete. Displaying Grayscale window." << std::endl;

    // Display grayscale window
    cv::Mat grayscale_image = cv::imread("grayscale_galaxy.jpg", cv::IMREAD_COLOR);
    cv::imshow("Grayscale window", grayscale_image);
    cv::waitKey(0);
    return 0;
}
