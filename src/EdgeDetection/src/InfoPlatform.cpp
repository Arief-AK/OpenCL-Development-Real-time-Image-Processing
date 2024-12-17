#include "InfoPlatform.hpp"

InfoPlatform::InfoPlatform(cl_platform_id id)
{
    m_profile = retrievePlatformInfo(id, CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE");
    m_name = retrievePlatformInfo(id, CL_PLATFORM_NAME, "CL_PLATFORM_NAME");
    m_version = retrievePlatformInfo(id, CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
    m_vendor = retrievePlatformInfo(id, CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
}

void InfoPlatform::DisplaySinglePlatformInfo(cl_platform_id id, cl_platform_info name, std::string str)
{
    // Retrieve and display information
    std::string info = retrievePlatformInfo(id, name, str);
    std::cout << "\t" << str << "\t" << info << std::endl;
}

void InfoPlatform::Display()
{
    std::cout << "\nPLATFORM PROPERTIES:" << std::endl;
    std::cout << "\t" << "CL_PLATFORM_PROFILE" << "\t" << m_profile << std::endl;
    std::cout << "\t" << "CL_PLATFORM_NAME" << "\t" << m_name << std::endl;
    std::cout << "\t" << "CL_PLATFORM_VERSION" << "\t" << m_version << std::endl;
    std::cout << "\t" << "CL_PLATFORM_VENDOR" << "\t" << m_vendor << std::endl;
}

std::string InfoPlatform::GetPlatformInfo(cl_platform_info name)
{
    // Initalise placeholder variable
    std::string info = {};

    switch (name)
    {
    case CL_PLATFORM_PROFILE:
        info = m_profile;
        break;
    
    case CL_PLATFORM_NAME:
        info = m_name;
        break;

    case CL_PLATFORM_VERSION:
        info = m_version;
        break;

    case CL_PLATFORM_VENDOR:
        info = m_vendor;
        break;

    default:
        std::cerr << "Unrecognised platform info" << std::endl;
        info = nullptr;
    }

    return info;
}

std::string InfoPlatform::retrievePlatformInfo(cl_platform_id id, cl_platform_info name, std::string str)
{
    // Initialise variables
    cl_int err_num = {};
    size_t param_value_size;

    // Determine the number of platforms with the `name` parameter
    err_num = clGetPlatformInfo(id, name, 0, NULL, &param_value_size);
    if(err_num != CL_SUCCESS){
        std::cerr << "Failed to find OpenCL platform information with parameter:\t" << str << std::endl;
        exit(-1);
    }

    // Allocate memory for the platform information
    char* info = (char*)alloca(sizeof(char) * param_value_size);

    // Retrieve the platform information
    err_num = clGetPlatformInfo(id, name, param_value_size, info, NULL);
    if(err_num != CL_SUCCESS){
        std::cerr << "Failed to find OpenCL platform information with parameter:\t" << str << std::endl;
        exit(-1);
    }

    // Set the member variable
    setPlatformInfo(name, info);

    // Return the info
    return info;
}

void InfoPlatform::setPlatformInfo(cl_platform_info name, std::string info)
{
    switch (name)
    {
    case CL_PLATFORM_PROFILE:
        m_profile = info;
        break;
    
    case CL_PLATFORM_NAME:
        m_name = info;
        break;

    case CL_PLATFORM_VERSION:
        m_version = info;
        break;

    case CL_PLATFORM_VENDOR:
        m_vendor = info;
        break;

    default:
        std::cerr << "Unrecognised platform info" << std::endl;
        exit(-1);
    }
}
