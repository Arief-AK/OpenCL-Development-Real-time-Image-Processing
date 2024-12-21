#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    std::cout << "Hello from CameraModule!" << std::endl;

    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }

    // Get FPS from OpenCV
    auto fps = cap.get(cv::CAP_PROP_FPS);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Overlay FPS on the frame
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Camera Feed", frame);
        if (cv::waitKey(1) == 27) break;  // Exit on 'ESC'
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}