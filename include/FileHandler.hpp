#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

class FileHandler
{
public:
    FileHandler();

    std::vector<std::string> LoadImages(const std::string& directory);
    
    void SaveImages(std::string image_path, cv::Mat& opencl_output_image);
    void WriteResultsToCSV(const std::string& filename, std::vector<std::tuple<std::string, std::string, std::string, int, double, double, double, double, double, double, double>>& results); 

private:
    bool SAVE_IMAGES;

    std::string m_directory_name;
    std::vector<std::string> m_image_paths;
};
