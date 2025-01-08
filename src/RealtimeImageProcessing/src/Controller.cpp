#include "Controller.hpp"

Controller::Controller() : num_platforms{0}, num_devices{0}, m_image_support{false} {}

void Controller::CheckError(cl_int err, const char *name)
{
    if(err != CL_SUCCESS){
        std::cerr << "Error: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::vector<cl_platform_id> Controller::GetPlatforms()
{
    cl_int err_num;
    cl_platform_id* platform_IDs;
    std::vector<cl_platform_id> m_platforms;

    // Determine platforms
    err_num = clGetPlatformIDs(0, NULL, &num_platforms);
    CheckError((err_num != CL_SUCCESS) ? err_num : (num_platforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");

    // Allocate platforms
    platform_IDs = (cl_platform_id*)alloca(sizeof(cl_platform_id) * num_platforms);
    std::cout << "Number of platforms: " << num_platforms << std::endl;

    // Retrieve platforms
    err_num = clGetPlatformIDs(num_platforms, platform_IDs, NULL);
    CheckError((err_num != CL_SUCCESS) ? err_num : (num_platforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");
    
    for(cl_uint i = 0; i < num_platforms; i++){
        m_platforms.push_back(platform_IDs[i]);
    }

    std::cout << "Found " << m_platforms.size() << " platforms" << std::endl;
    return m_platforms;
}

std::vector<cl_device_id> Controller::GetDevices(cl_platform_id platform)
{
    cl_int err_num;
    cl_device_id* devices;
    std::vector<cl_device_id> m_devices = {};

    // Determine devices
    err_num = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if(err_num != CL_SUCCESS && err_num != CL_DEVICE_NOT_FOUND){
        CheckError(err_num, "clGetDeviceIDs");
    }

    // Allocate devices
    devices = (cl_device_id*)alloca(sizeof(cl_device_id) * num_devices);

    // Retrieve devices
    err_num = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    CheckError(err_num, "clGetDeviceIDs");

    for(cl_uint i = 0; i < num_devices; i++){
        m_devices.push_back(devices[i]);
    }

    std::cout << "Found " << m_devices.size() << " devices" << std::endl;
    return m_devices;
}

void Controller::_profileEvent(cl_event &event, std::vector<cl_ulong> *profiling_events)
{
    cl_ulong event_start, event_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &event_start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &event_end, NULL);
    profiling_events->push_back(event_start);
    profiling_events->push_back(event_end);
}

std::vector<unsigned char> Controller::ConvertToUChar(const std::vector<float> &input_data)
{
    std::vector<unsigned char> output_data(input_data.size());

    for (size_t i = 0; i < input_data.size(); i++){
        output_data[i] = static_cast<unsigned char>(input_data[i] * 255.0f);
    }
    
    return output_data;
}

cl_bool Controller::GetImageSupport()
{
    return m_image_support;
}

void Controller::SetImageSupport(cl_bool image_support)
{
    m_image_support = image_support;
}

cl_context Controller::CreateContext(cl_platform_id platform, std::vector<cl_device_id> devices)
{
    cl_int err_num;
    cl_context context;
    cl_device_id* m_devices = new cl_device_id[devices.size()];
    cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};

    // Copy contents of devices vector to m_devices array
    std::memcpy(m_devices, devices.data(), devices.size() * sizeof(cl_device_id));

    // Create context
    context = clCreateContext(context_properties, num_devices, m_devices, NULL, NULL, &err_num);
    CheckError(err_num, "clCreateContext");

    std::cout << "Successfully created a context" << std::endl;
    return context;
}

cl_command_queue Controller::CreateCommandQueue(cl_context context, cl_device_id device)
{
    cl_command_queue command_queue;
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;

    // Create a command queue
    command_queue = clCreateCommandQueue(context, device, properties, NULL);
    if(command_queue == NULL){
        std::cerr << "Failed to create CommandQueue" << std::endl;
        return NULL;
    }

    std::cout << "Successfully created CommandQueue" << std::endl;
    return command_queue;
}

cl_program Controller::CreateProgram(cl_context context, cl_device_id device, const char *filename)
{
    cl_int err_num;
    cl_program program;    

    // Open the kernel file
    std::ifstream kernelFile(filename, std::ios::in);
    if(!kernelFile.is_open()){
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        if (errno == ENOENT) {
            std::cerr << "Error: File does not exist." << std::endl;
        } else if (errno == EACCES) {
            std::cerr << "Error: Permission denied." << std::endl;
        } else {
            std::cerr << "Error: " << strerror(errno) << std::endl;
        }
        return NULL;
    }

     // Read the kernel file
    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    // Convert the buffer from the output stream to a standard string
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();

    // Create a program
    program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, NULL);
    if(program == NULL){
        std::cerr << "Failed to create program objects from source" << std::endl;
        return NULL;
    }

    // Build the program    
    err_num = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err_num != CL_SUCCESS){
        // Determine the reason for failure
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        std::cerr << "Error in kernel:" << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    std::cout << "Successfully created a program" << std::endl;
    return program;
}

cl_kernel Controller::CreateKernel(cl_program program, const char *kernel_name)
{
    cl_int err_num;
    cl_kernel kernel;

    kernel = clCreateKernel(program, kernel_name, &err_num);
    CheckError(err_num, "clCreateKernel");

    std::cout << "Successfully created the " << kernel_name << " kernel" << std::endl;
    return kernel;
}

void Controller::DisplayPlatformInformation(cl_platform_id platform)
{
    InfoPlatform platform_handler(platform);
    platform_handler.Display();
}

void Controller::Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_sampler sampler, cl_mem *mem_objects, int num_mem_objects)
{
    std::cout << "Performing cleanup" << std::endl;

    // Free all memory objects
    if(num_mem_objects > 0){
        for (int i = 0; i < num_mem_objects; i++){
            if (mem_objects[i] != 0)
                clReleaseMemObject(mem_objects[i]);
        }
    }

    // Free the command queues
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    // Free the kernels
    if (kernel != 0)
        clReleaseKernel(kernel);

    // Free the program objects
    if (program != 0)
        clReleaseProgram(program);

    // Free the context
    if (context != 0)
        clReleaseContext(context);

    // Free the sampler
    if (sampler != 0)
        clReleaseSampler(sampler);

    std::cout << "Succesfully cleaned environment" << std::endl;
}

std::pair<cl_mem, cl_mem> Controller::_initGrayscaleBuffers(cl_context *context, cl_command_queue *command_queue, std::vector<unsigned char> *input_data, cl_int width, cl_int height, cl_event *write_event, Logger& logger)
{
    // Initialise error variable
    cl_int err_num;

    // Define buffers
    cl_mem input_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4 * sizeof(unsigned char), input_data->data(), &err_num);
    cl_mem output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, width * height * sizeof(float) * 4, nullptr, &err_num);

    return std::make_pair(input_buffer, output_buffer);
}

std::pair<cl_mem, cl_mem> Controller::_initGrayscaleImage2D(cl_context *context, cl_command_queue *command_queue, std::vector<unsigned char> *input_data, cl_int width, cl_int height, cl_event *write_event, Logger& logger)
{
    // Initialise error variable
    cl_int err_num;

    // Define cl_image variables and format
    cl_image_format input_format;
    input_format.image_channel_order = CL_RGBA;     // RGB
    input_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_format output_format;
    output_format.image_channel_order = CL_R;       // Single channel (grayscale)
    output_format.image_channel_data_type = CL_FLOAT;

    // Create memory objects
    cl_mem input_image = clCreateImage2D(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &input_format, width, height, 0, input_data->data(), &err_num);
    if(err_num != CL_SUCCESS){
        logger.log("Failed to create cl_image input_image mem object", Logger::LogLevel::ERROR);
    }

    cl_mem output_image = clCreateImage2D(*context, CL_MEM_WRITE_ONLY, &output_format, width, height, 0, nullptr, &err_num);
    if(err_num != CL_SUCCESS){
        logger.log("Failed to create cl_image output_image mem object", Logger::LogLevel::ERROR);
    }

    return std::make_pair(input_image, output_image);
}

std::pair<cl_mem, cl_mem> Controller::_initEdgeDetectionBuffers(cl_context *context, cl_command_queue *command_queue, std::vector<unsigned char> *input_data, cl_int width, cl_int height, cl_event *write_event, Logger &logger)
{
    // Initialise error variable
    cl_int err_num;

    cl_mem input_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * 4 * width * height, input_data->data(), &err_num);
    cl_mem output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(float) * width * height, nullptr, &err_num);

    return std::make_pair(input_buffer, output_buffer);
}

std::pair<cl_mem, cl_mem> Controller::_initEdgeDetectionImage2D(cl_context *context, cl_command_queue *command_queue, std::vector<unsigned char> *input_data, cl_int width, cl_int height, cl_event *write_event, Logger &logger)
{
    // Initialise error variable
    cl_int err_num;

    // Define cl_image variables and format
    cl_image_format input_format;
    input_format.image_channel_order = CL_RGBA;     // RGB
    input_format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_format output_format;
    output_format.image_channel_order = CL_R;       // Single channel (grayscale)
    output_format.image_channel_data_type = CL_FLOAT;

    // Create memory objects
    cl_mem input_image = clCreateImage2D(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &input_format,
        width, height, 0, input_data->data(), &err_num);
    if(err_num != CL_SUCCESS){
        logger.log("Failed to create cl_image input_image mem object", Logger::LogLevel::ERROR);
    }

    cl_mem output_image = clCreateImage2D(*context, CL_MEM_WRITE_ONLY, &output_format,
        width, height, 0, nullptr, &err_num);
    if(err_num != CL_SUCCESS){
        logger.log("Failed to create cl_image output_image mem object", Logger::LogLevel::ERROR);
    }    

    return std::make_pair(input_image, output_image);
}

std::pair<cl_mem, cl_mem> Controller::_initGaussianBlurBuffers(cl_context *context, cl_command_queue *command_queue, std::vector<unsigned char> *input_data, cl_int width, cl_int height, cl_event *write_event, Logger &logger)
{
    // Initialise error variable
    cl_int err_num;

    cl_mem input_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * 4 * width * height, input_data->data(), &err_num);
    cl_mem output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * 4 * width * height, nullptr, &err_num);

    return std::make_pair(input_buffer, output_buffer);
}

std::pair<cl_mem, cl_mem> Controller::_initGaussianBlurImage2D(cl_context *context, cl_command_queue *command_queue, std::vector<unsigned char> *input_data, cl_int width, cl_int height, cl_event *write_event, Logger &logger)
{
    // Initialise error variable
    cl_int err_num;

    // Define cl_image variables and format
    cl_image_format image_format;
    image_format.image_channel_order = CL_RGBA;                // RGBA
    image_format.image_channel_data_type = CL_UNORM_INT8;

    // Create memory objects
    cl_mem input_image = clCreateImage2D(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &image_format,
        width, height, 0, input_data->data(), &err_num);
    if(err_num != CL_SUCCESS){
        logger.log("Failed to create cl_image input_image mem object", Logger::LogLevel::ERROR);
    }

    cl_mem output_image = clCreateImage2D(*context, CL_MEM_WRITE_ONLY, &image_format,
        width, height, 0, nullptr, &err_num);
    if(err_num != CL_SUCCESS){
        logger.log("Failed to create cl_image output_image mem object", Logger::LogLevel::ERROR);
    }
    
    return std::make_pair(input_image, output_image);
}

std::vector<float> Controller::_GenerateGaussianKernelBuffers(int kernel_size, float sigma)
{
    std::vector<float> kernel(kernel_size * kernel_size);
    int half_size = kernel_size / 2;
    float sum = 0.0f;

    for (int y = -half_size; y <= half_size; y++) {
        for (int x = -half_size; x <= half_size; x++) {
            float value = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[(y + half_size) * kernel_size + (x + half_size)] = value;
            sum += value;
        }
    }

    // Normalize the kernel
    for (float& value : kernel) {
        value /= sum;
    }

    return kernel;
}

std::vector<float> Controller::_GenerateGaussianKernelImage2D(int kernel_size, float sigma)
{
    // Initialise kernel vector
    std::vector<float> kernel(kernel_size * kernel_size);

    // Initialise kernel variables
    int half_kernel_size = kernel_size / 2;
    float sum = 0.0;

    // Generate kernel
    for (int x = -half_kernel_size; x < half_kernel_size; x++){
        for (int y = -half_kernel_size; y < half_kernel_size; y++){
            // Calculate kernel value
            float value = exp(-((x * x + y * y) / (2 * sigma * sigma))) / (2 * M_PI * sigma * sigma);

            // Store kernel value
            kernel[(x + half_kernel_size) * kernel_size + (y + half_kernel_size)] = exp(-((x * x + y * y) / (2 * sigma * sigma))) / (2 * M_PI * sigma * sigma);
            
            // Calculate sum
            sum += kernel[(x + half_kernel_size) * kernel_size + (y + half_kernel_size)];
        }
    }

    // Normalize the kernel
    for (float& value : kernel) {
        value /= sum;
    }

    return kernel;
}

std::vector<float> Controller::_GenerateGausianKernel(int kernel_size, float sigma)
{
    // Initialise variable
    std::vector<float> gaussian_kernel;

    switch (m_image_support)
    {
    case CL_TRUE:
        gaussian_kernel = _GenerateGaussianKernelImage2D(kernel_size, sigma);
        break;

    case CL_FALSE:
        gaussian_kernel = _GenerateGaussianKernelBuffers(kernel_size, sigma);
        break;
    
    default:
        std::cerr << "Failed to create Gaussian Kernel" << std::endl;
        exit(1);
        break;
    }
    
    return gaussian_kernel;
}

void Controller::PerformCLImageGrayscaling(cl_context *context, cl_command_queue *command_queue, cl_kernel *kernel,
    std::vector<cl_ulong> *profiling_events, std::vector<unsigned char> *input_data, std::vector<unsigned char> *output_data,
    cl_int &width, cl_int &height, Logger &logger)
{
    // Initialise error variable
    cl_int err_num;
    
    // Initialise profiling variables
    cl_event write_event;
    cl_event kernel_event;
    cl_event read_event;

    // Initialise grayscale variables
    std::vector<float> float_output_data(output_data->size());
    std::pair<cl_mem, cl_mem> buffers;

    // Initialise the global work size for kernel execution
    size_t global_work_size[2] = {width, height};

    // Initialise input image
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    switch (m_image_support){
    case CL_TRUE:
        buffers = _initGrayscaleImage2D(context, nullptr, input_data, width, height, &write_event, logger);
        err_num = clEnqueueWriteImage(*command_queue, buffers.first, CL_FALSE, origin, region, 0, 0, input_data->data(), 0, nullptr, &write_event);
        break;
    
    case CL_FALSE:
        buffers = _initGrayscaleBuffers(context, nullptr, input_data, width, height, &write_event, logger);
        err_num = clEnqueueWriteBuffer(*command_queue, buffers.first, CL_FALSE, 0, width * height * 4 * sizeof(unsigned char), input_data->data(), 0, nullptr, &write_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to write cl_mem (buffer) to kernel", Logger::LogLevel::ERROR);
        }
        break;

    default:
        break;
    }

    clWaitForEvents(1, &write_event);
    _profileEvent(write_event, profiling_events);

    // Set kernel arguments
    err_num = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &buffers.first);
    err_num |= clSetKernelArg(*kernel, 1, sizeof(cl_mem), &buffers.second);
    err_num |= clSetKernelArg(*kernel, 2, sizeof(int), &width);
    err_num |= clSetKernelArg(*kernel, 3, sizeof(int), &height);
    if(err_num != CL_SUCCESS){
        logger.log("Failed to set kernel arguments", Logger::LogLevel::ERROR);
    }

    // Perform kernel
    err_num = clEnqueueNDRangeKernel(*command_queue, *kernel, 2, nullptr, global_work_size, nullptr, 1, &write_event, &kernel_event);
    if(err_num != CL_SUCCESS){
        logger.log("Failed when executing kernel", Logger::LogLevel::ERROR);
    }

    clWaitForEvents(1, &kernel_event);
    _profileEvent(kernel_event, profiling_events);

    switch (m_image_support){
    case CL_TRUE:
        err_num = clEnqueueReadImage(*command_queue, buffers.second, CL_FALSE, origin, region, 0, 0, float_output_data.data(), 1, &kernel_event, &read_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed when executing kernel", Logger::LogLevel::ERROR);
        }
        break;
    
    case CL_FALSE:
        err_num = clEnqueueReadBuffer(*command_queue, buffers.second, CL_FALSE, 0, width * height * sizeof(float) * 4, float_output_data.data(), 1, &kernel_event, &read_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed when executing kernel", Logger::LogLevel::ERROR);
        }
        break;

    default:
        break;
    }

    *output_data = ConvertToUChar(float_output_data);

    clWaitForEvents(1, &read_event);
    _profileEvent(read_event, profiling_events);

    clReleaseMemObject(buffers.first);
    clReleaseMemObject(buffers.second);
    clReleaseEvent(kernel_event);

}

void Controller::PerformCLImageEdgeDetection(cl_context *context, cl_command_queue *command_queue, cl_kernel *kernel,
    std::vector<cl_ulong> *profiling_events, std::vector<unsigned char> *input_data, std::vector<unsigned char> *output_data,
    cl_int &width, cl_int &height, Logger &logger)
{
    // Initialise error variable
    cl_int err_num;
    
    // Initialise profiling variables
    cl_event write_event;
    cl_event kernel_event;
    cl_event read_event;

    // Initialise edge-detection variables
    std::vector<float> float_output_data(output_data->size());
    std::pair<cl_mem, cl_mem> buffers;

    // Initialise the global work size for kernel execution
    size_t global_work_size[2] = {width, height};

    // Initialise input image
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    switch (m_image_support)
    {
    case CL_TRUE:
        buffers = _initEdgeDetectionImage2D(context, command_queue, input_data, width, height, &write_event, logger);
        err_num = clEnqueueWriteImage(*command_queue, buffers.first, CL_FALSE, origin, region, 0, 0, input_data->data(), 0, nullptr, &write_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to write cl_image", Logger::LogLevel::ERROR);
        }
        break;
    
    case CL_FALSE:
        buffers = _initEdgeDetectionBuffers(context, command_queue, input_data, width, height, &write_event, logger);
        err_num = clEnqueueWriteBuffer(*command_queue, buffers.first, CL_FALSE, 0, width * height * 4 * sizeof(unsigned char), input_data->data(), 0, nullptr, &write_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to write cl_mem (buffer) to kernel", Logger::LogLevel::ERROR);
        }
        break;

    default:
        break;
    }

    clWaitForEvents(1, &write_event);
    _profileEvent(write_event, profiling_events);

    // Set kernel arguments
    err_num = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &buffers.first);
    err_num |= clSetKernelArg(*kernel, 1, sizeof(cl_mem), &buffers.second);
    err_num |= clSetKernelArg(*kernel, 2, sizeof(int), &width);
    err_num |= clSetKernelArg(*kernel, 3, sizeof(int), &height);
    if(err_num != CL_SUCCESS){
        logger.log("Failed to set kernel arguments", Logger::LogLevel::ERROR);
    }

    // Perform kernel
    err_num = clEnqueueNDRangeKernel(*command_queue, *kernel, 2, nullptr, global_work_size, nullptr, 1, &write_event, &kernel_event);
    if(err_num != CL_SUCCESS){
        logger.log("Failed when executing kernel", Logger::LogLevel::ERROR);
    }

    clWaitForEvents(1, &kernel_event);
    _profileEvent(kernel_event, profiling_events);

    // Read back image data
    switch (m_image_support)
    {
    case CL_TRUE:
        err_num = clEnqueueReadImage(*command_queue, buffers.second, CL_FALSE, origin, region, 0, 0, float_output_data.data(), 1, &kernel_event, &read_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to read back image data from kernel", Logger::LogLevel::ERROR);
        }
        break;

    case CL_FALSE:
        clEnqueueReadBuffer(*command_queue, buffers.second, CL_TRUE, 0, sizeof(float) * width * height, float_output_data.data(), 1, &kernel_event, &read_event);
        break;
    
    default:
        break;
    }

    *output_data = ConvertToUChar(float_output_data);

    clWaitForEvents(1, &read_event);
    _profileEvent(read_event, profiling_events);

    clReleaseMemObject(buffers.first);
    clReleaseMemObject(buffers.second);
    clReleaseEvent(kernel_event);
}

void Controller::PerformCLGaussianBlur(int& kernel_size, float& kernel_sigma, cl_context *context, cl_command_queue *command_queue, cl_kernel *kernel,
    std::vector<cl_ulong> *profiling_events, std::vector<unsigned char> *input_data, std::vector<unsigned char> *output_data,
    cl_int &width, cl_int &height, Logger &logger)
{
    // Initialise error variable
    cl_int err_num;
    
    // Initialise profiling variables
    cl_event write_event;
    cl_event kernel_event;
    cl_event read_event;

    std::pair<cl_mem, cl_mem> buffers;

    // Initialise the global work size for kernel execution
    size_t global_work_size[2] = {width, height};

    // Initialise input image
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    switch (m_image_support)
    {
    case CL_TRUE:
        buffers = _initGaussianBlurImage2D(context, command_queue, input_data, width, height, &write_event, logger);
        err_num = clEnqueueWriteImage(*command_queue, buffers.first, CL_FALSE, origin, region, 0, 0, input_data->data(), 0, nullptr, &write_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to write cl_image", Logger::LogLevel::ERROR);
        }
        break;

    case CL_FALSE:
        buffers = _initGaussianBlurBuffers(context, command_queue, input_data, width, height, &write_event, logger);
        err_num = clEnqueueWriteBuffer(*command_queue, buffers.first, CL_FALSE, 0, width * height * 4 * sizeof(unsigned char), input_data->data(), 0, nullptr, &write_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to write cl_mem (buffer) to kernel", Logger::LogLevel::ERROR);
        }
        break;
    
    default:
        break;
    }

    // Create Gaussian Kernel
    std::vector<float> gaussian_kernel;
    switch (m_image_support)
    {
    case CL_TRUE:
        gaussian_kernel = _GenerateGaussianKernelImage2D(kernel_size, kernel_sigma);
        break;
    
    case CL_FALSE:
        gaussian_kernel = _GenerateGaussianKernelBuffers(kernel_size, kernel_sigma);
        break;

    default:
        break;
    }

    cl_mem kernel_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gaussian_kernel.size() * sizeof(float),
        gaussian_kernel.data(), &err_num);
    if(err_num != CL_SUCCESS){
        logger.log("Failed to create cl_mem gaussian_kernel_buffer mem object", Logger::LogLevel::ERROR);
    }

    clWaitForEvents(1, &write_event);
    _profileEvent(write_event, profiling_events);

    // Set kernel arguments
    switch (m_image_support)
    {
    case CL_TRUE:
        err_num = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &buffers.first);
        err_num |= clSetKernelArg(*kernel, 1, sizeof(cl_mem), &buffers.second);
        err_num |= clSetKernelArg(*kernel, 2, sizeof(cl_mem), &kernel_buffer);
        err_num |= clSetKernelArg(*kernel, 3, sizeof(int), &kernel_size);
        err_num |= clSetKernelArg(*kernel, 4, sizeof(int), &width);
        err_num |= clSetKernelArg(*kernel, 5, sizeof(int), &height);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to set kernel arguments", Logger::LogLevel::ERROR);
        }   
        break;

    case CL_FALSE:
        clSetKernelArg(*kernel, 0, sizeof(cl_mem), &buffers.first);
        clSetKernelArg(*kernel, 1, sizeof(cl_mem), &buffers.second);
        clSetKernelArg(*kernel, 2, sizeof(cl_mem), &kernel_buffer);
        clSetKernelArg(*kernel, 3, sizeof(int), &kernel_size);
        clSetKernelArg(*kernel, 4, sizeof(int), &width);
        clSetKernelArg(*kernel, 5, sizeof(int), &height);
        break;
    
    default:
        break;
    }

    // Perform kernel
    err_num = clEnqueueNDRangeKernel(*command_queue, *kernel, 2, nullptr, global_work_size, nullptr, 1, &write_event, &kernel_event);
    if(err_num != CL_SUCCESS){
        logger.log("Failed when executing kernel", Logger::LogLevel::ERROR);
    }

    clWaitForEvents(1, &kernel_event);
    _profileEvent(kernel_event, profiling_events);

    // Read back image data
    switch (m_image_support)
    {
    case CL_TRUE:
        err_num = clEnqueueReadImage(*command_queue, buffers.second, CL_FALSE, origin, region, 0, 0, output_data->data(), 1, &kernel_event, &read_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to read back image data from kernel", Logger::LogLevel::ERROR);
        }   
        break;
    
    case CL_FALSE:
        clEnqueueReadBuffer(*command_queue, buffers.second, CL_TRUE, 0, sizeof(unsigned char) * 4 * width * height, output_data->data(), 1, &kernel_event, &read_event);
        break;

    default:
        break;
    }

    clWaitForEvents(1, &read_event);
    _profileEvent(read_event, profiling_events);

    clReleaseMemObject(buffers.first);
    clReleaseMemObject(buffers.second);
    clReleaseEvent(kernel_event);
}