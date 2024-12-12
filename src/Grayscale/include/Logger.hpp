#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <fstream>
#include <sstream>
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

    static Logger& getInstance();
    void setLogFile(const std::string& file_name);
    void log(const std::string& message, LogLevel level);

private:
    std::ofstream m_log_file;
    std::mutex m_mutex;

    Logger();
    ~Logger();

    std::string _getCurrentTime();
    std::string _printLogLevel(LogLevel level);
};

#endif  // LOGGER_HPP