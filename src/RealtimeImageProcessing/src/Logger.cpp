#include "Logger.hpp"

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() = default;

Logger::~Logger() {
    if (m_log_file.is_open()) {
        m_log_file.close();
    }
}

std::string Logger::getCurrentTime()
{
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);
    std::ostringstream oss;
    oss << (1900 + localTime->tm_year) << "-"
        << (1 + localTime->tm_mon) << "-"
        << localTime->tm_mday << " "
        << localTime->tm_hour << ":"
        << localTime->tm_min << ":"
        << localTime->tm_sec;
    return oss.str();
}

std::string Logger::_printLogLevel(LogLevel level)
{
    switch (level) {
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

void Logger::setLogFile(const std::string &file_name)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if(m_log_file.is_open()){
        m_log_file.close();
    }

    m_log_file.open(file_name, std::ios::out | std::ios::app);
    if(!m_log_file){
        throw std::runtime_error("Failed to open log file: " + file_name);
    }
}

void Logger::setTerminalDisplay(bool print_on_terminal)
{
    m_print_terminal = print_on_terminal;
}

void Logger::log(const std::string &message, LogLevel level)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string timestamp = getCurrentTime();
    std::string level_str = _printLogLevel(level);
    std::string log_message = "[" + timestamp + "]" + "[" + level_str + "] " + message;

    if(m_set_level == level){
        // Print to terminal
        if(m_print_terminal)
            std::cout << log_message << std::endl;
    }

    if(m_log_file.is_open()){
        m_log_file << log_message << std::endl;
    }
}

void Logger::setLogLevel(LogLevel level)
{
    m_set_level = level;
}

void Logger::PrintEndToEndExecutionTime(std::string method, double total_execution_time_ms)
{
    log("-------------------- START OF " + method + " EXECUTION TIME (end-to-end) DETAILS --------------------", Logger::LogLevel::INFO);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << "Total execution time (end-to-end): " << total_execution_time_ms << " ms";
    log(oss.str(), Logger::LogLevel::INFO);

    log("-------------------- END OF " + method + " EXECUTION TIME (end-to-end) DETAILS --------------------", Logger::LogLevel::INFO);
}

void Logger::PrintRawKernelExecutionTime(double &opencl_kernel_execution_time, double &opencl_kernel_write_time, double &opencl_kernel_read_time, double &opencl_kernel_operation_time)
{
    log("-------------------- START OF KERNEL EXEUCTION DETAILS --------------------", Logger::LogLevel::INFO);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(5) << "Kernel write time: " << opencl_kernel_write_time << " ms";
    log(oss.str(), Logger::LogLevel::INFO);
    oss.str("");

    oss << std::fixed << std::setprecision(5) << "Kernel execution time: " << opencl_kernel_execution_time << " ms";
    log(oss.str(), Logger::LogLevel::INFO);
    oss.str("");

    oss << std::fixed << std::setprecision(5) << "Kernel read time: " << opencl_kernel_read_time << " ms";
    log(oss.str(), Logger::LogLevel::INFO);
    oss.str("");

    oss << std::fixed << std::setprecision(5) << "Kernel complete operation time: " << opencl_kernel_operation_time << " ms";
    log(oss.str(), Logger::LogLevel::INFO);

    log("-------------------- END OF KERNEL EXEUCTION DETAILS --------------------", Logger::LogLevel::INFO);
}

void Logger::PrintSummary(double &opencl_kernel_execution_time, double &opencl_kernel_write_time, double &opencl_kernel_read_time, double &opencl_execution_time, double &opencl_kernel_operation_time, double &cpu_execution_time)
{
    if(m_print_terminal)
        std::cout << "\n **************************************** START OF OpenCL SUMMARY **************************************** " << std::endl;
    
    PrintEndToEndExecutionTime("OpenCL", opencl_execution_time);
    PrintRawKernelExecutionTime(opencl_kernel_execution_time, opencl_kernel_write_time, opencl_kernel_read_time, opencl_kernel_operation_time);
    
    if(m_print_terminal){
        std::cout << " **************************************** END OF OpenCL SUMMARY **************************************** " << std::endl;
        std::cout << "\n **************************************** START OF CPU SUMMARY **************************************** " << std::endl;
    }

    PrintEndToEndExecutionTime("CPU", cpu_execution_time);

    if(m_print_terminal)
        std::cout << "\n **************************************** END OF CPU SUMMARY **************************************** " << std::endl;
}
