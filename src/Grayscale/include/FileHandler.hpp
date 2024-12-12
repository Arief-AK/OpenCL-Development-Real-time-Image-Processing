#include <vector>
#include <sstream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

class FileHandler
{
public:
    FileHandler();

    std::vector<std::string> LoadImages(const std::string& directory);

private:
    std::string m_directory_name;
    std::vector<std::string> m_image_paths;
};
