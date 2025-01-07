#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <ctime>

class Logger
{
public:
    enum class LogLevel {
        INFO,
        WARNING,
        ERROR
    };

    // Disable copy and assingment operators
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string getCurrentTime();

    static Logger& getInstance();
    void setLogLevel(LogLevel level);
    void setLogFile(const std::string& file_name);
    void setTerminalDisplay(bool print_on_terminal);
    void log(const std::string& message, LogLevel level);

    void PrintEndToEndExecutionTime(std::string method, double total_execution_time_ms, Logger& logger);
    void PrintRawKernelExecutionTime(double& opencl_kernel_execution_time, double& opencl_kernel_write_time, double& opencl_kernel_read_time, double& opencl_kernel_operation_time, Logger& logger);
    void PrintSummary(double& opencl_kernel_execution_time, double& opencl_kernel_write_time, double& opencl_kernel_read_time, double& opencl_execution_time, double& opencl_kernel_operation_time,
    double& cpu_execution_time, Logger& logger);

private:
    std::ofstream m_log_file;
    std::mutex m_mutex;
    bool m_print_terminal;
    LogLevel m_set_level;

    Logger();
    ~Logger();

    std::string _printLogLevel(LogLevel level);
};

#endif  // LOGGER_HPP