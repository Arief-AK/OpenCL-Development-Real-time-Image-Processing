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

    std::string getCurrentTime();

    static Logger& getInstance();
    void setLogFile(const std::string& file_name);
    void setTerminalDisplay(bool print_on_terminal);
    void log(const std::string& message, LogLevel level);

private:
    std::ofstream m_log_file;
    std::mutex m_mutex;
    bool m_print_terminal;

    Logger();
    ~Logger();

    std::string _printLogLevel(LogLevel level);
};

#endif  // LOGGER_HPP