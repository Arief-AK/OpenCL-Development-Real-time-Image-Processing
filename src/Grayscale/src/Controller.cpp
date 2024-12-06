#include "Controller.hpp"

Controller::Controller() : num_platforms{0}, num_devices{0} {}

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

    // Create a command queue
    command_queue = clCreateCommandQueue(context, device, 0, NULL);
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
