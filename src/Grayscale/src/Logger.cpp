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

void Logger::log(const std::string &message, LogLevel level)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::string timestamp = getCurrentTime();
    std::string level_str = _printLogLevel(level);
    std::string log_message = "[" + timestamp + "]" + "[" + level_str + "] " + message;

    // Print to terminal
    std::cout << log_message << std::endl;

    if(m_log_file.is_open()){
        m_log_file << log_message << std::endl;
    }
}

