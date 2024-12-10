#include "FileHandler.hpp"

FileHandler::FileHandler():m_directory_name{"images"}, m_image_paths{}{}

std::vector<std::string> FileHandler::LoadImages(const std::string &directory){    
    // Iterate through the directory
    for (const auto& entry : fs::directory_iterator(directory)){
        // Check file-type
        if(entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
            m_image_paths.push_back(entry.path().string());
    }
    
    return m_image_paths;
}
