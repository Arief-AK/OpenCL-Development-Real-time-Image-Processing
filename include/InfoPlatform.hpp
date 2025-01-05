#ifndef INFOPLATFORM_H
#define INFOPLATFORM_H

#include <CL/cl.h>
#include <iostream>

class InfoPlatform
{
public:
    InfoPlatform(cl_platform_id id);
    void DisplaySinglePlatformInfo(cl_platform_id id, cl_platform_info name, std::string str);
    void Display();

    std::string GetPlatformInfo(cl_platform_info name);

private:
    std::string retrievePlatformInfo(cl_platform_id id, cl_platform_info name, std::string str);
    void setPlatformInfo(cl_platform_info name, std::string info);

    std::string m_profile;
    std::string m_name;
    std::string m_version;
    std::string m_vendor;
};

#endif // INFOPLATFORM_H