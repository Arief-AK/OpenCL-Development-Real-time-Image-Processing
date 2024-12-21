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

std::pair<cl_mem, cl_mem> Controller::_initGrayscaleBuffers(cl_context *context, cl_mem *input_image, cl_mem *output_image, cl_command_queue *command_queue, std::vector<unsigned char> *input_data, cl_int width, cl_int height, cl_event *write_event, Logger& logger)
{
    // Initialise error variable
    cl_int err_num;

    // Define buffers
    cl_mem input_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * 4 * sizeof(unsigned char), input_data->data(), &err_num);
    cl_mem output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), nullptr, &err_num);

    return std::make_pair(input_buffer, output_buffer);
}

std::pair<cl_mem, cl_mem> Controller::_initGrayscleImage2D(cl_context *context, cl_mem *input_image, cl_mem *output_image, cl_command_queue *command_queue, std::vector<unsigned char> *input_data, cl_int width, cl_int height, cl_event *write_event, Logger& logger)
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

    return std::make_pair(*input_image, *output_image);
}

void Controller::PerformCLImageGrayscaling(std::string image_path, cl_context *context, cl_command_queue *command_queue, cl_kernel *kernel,
                                           std::vector<cl_ulong> *profiling_events, std::vector<unsigned char> *input_data, std::vector<float> *output_data,
                                           cl_int &width, cl_int &height, Logger &logger)
{
    // Initialise error variable
    cl_int err_num;
    
    // Initialise profiling variables
    cl_event write_event;
    cl_event kernel_event;
    cl_event read_event;
    cl_ulong write_event_start, write_event_end, kernel_event_start, kernel_event_end, read_event_start, read_event_end;

    std::pair<cl_mem, cl_mem> buffers;

    // Initialise the global work size for kernel execution
    size_t global_work_size[2] = {width, height};

    // Initialise input image
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    switch (m_image_support){
    case true:
        buffers = _initGrayscleImage2D(context, nullptr, nullptr, nullptr, input_data, width, height, &write_event, logger);
        err_num = clEnqueueWriteImage(*command_queue, buffers.first, CL_FALSE, origin, region, 0, 0, input_data->data(), 0, nullptr, &write_event);
        break;
    
    case false:
        buffers = _initGrayscaleBuffers(context, nullptr, nullptr, nullptr, input_data, width, height, &write_event, logger);
        err_num = clEnqueueWriteBuffer(*command_queue, buffers.first, CL_FALSE, 0, width * height * 4 * sizeof(unsigned char), input_data->data(), 0, nullptr, &write_event);
        if(err_num != CL_SUCCESS){
            logger.log("Failed to write cl_mem (buffer) to kernel", Logger::LogLevel::ERROR);
        }
        break;

    default:
        break;
    }

    clWaitForEvents(1, &write_event);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &write_event_start, nullptr);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &write_event_end, NULL);
    profiling_events->push_back(write_event_start);
    profiling_events->push_back(write_event_end);

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
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernel_event_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernel_event_end, NULL);
    profiling_events->push_back(kernel_event_start);
    profiling_events->push_back(kernel_event_end);

    switch (m_image_support){
    case true:
        err_num = clEnqueueReadImage(*command_queue, buffers.second, CL_FALSE, origin, region, 0, 0, output_data->data(), 1, &kernel_event, &read_event);
        break;
    
    case false:
        err_num = clEnqueueReadBuffer(*command_queue, buffers.second, CL_FALSE, 0, width * height * sizeof(unsigned char), output_data->data(), 1, &kernel_event, &read_event);
        break;

    default:
        break;
    }

    clWaitForEvents(1, &read_event);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_event_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_event_end, NULL);
    profiling_events->push_back(read_event_start);
    profiling_events->push_back(read_event_end);
}
