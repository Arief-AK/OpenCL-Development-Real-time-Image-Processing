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

void FileHandler::SaveImages(std::string image_path, cv::Mat &opencl_output_image)
{
    if(SAVE_IMAGES){
        // Convert output data to OpenCV matrix
        auto new_image_path = "images/opencl_grayscale_" + std::filesystem::path(image_path).filename().string();
        cv::imwrite(new_image_path, opencl_output_image);
    }
}

void FileHandler::WriteResultsToCSV(const std::string &filename, std::vector<std::tuple<std::string, std::string, std::string, int, double, double, double, double, double, double, double>> &results)
{
    std::ofstream file(filename);
    file << "Timestamp, Image, Resolution, Num_Iterations, avg_CPU_Time_ms, avg_OpenCL_Time_ms, avg_OpenCL_kernel_ms, avg_OpenCL_kernel_write_ms, avg_OpenCL_kernel_read_ms, avg_OpenCL_kernel_operation_ms, Error_MAE\n";
    for (const auto& [timestamp, image, resolution, num_iterations, avg_cpu_time, avg_opencl_time, avg_opencl_kernel_time, avg_opencl_kernel_write_time, avg_opencl_kernel_read_time, avg_opencl_kernel_operation_time, mae] : results) {
        file << timestamp << ", " << image << ", " << resolution << ", " << num_iterations << ", " << avg_cpu_time << ", " << avg_opencl_time << ", " << avg_opencl_kernel_time << ", "
        << avg_opencl_kernel_write_time << ", " << avg_opencl_kernel_read_time << ", " << avg_opencl_kernel_operation_time << ", " << mae << "\n";
    }
    file.close();
}
