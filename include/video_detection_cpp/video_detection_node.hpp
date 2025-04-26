#ifndef VIDEO_DETECTION_CPP_VIDEO_DETECTION_NODE_HPP_
#define VIDEO_DETECTION_CPP_VIDEO_DETECTION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <pybind11/embed.h>  // 用于嵌入 Python
#include <memory>
#include <string>

namespace py = pybind11;

class VideoDetectionNode : public rclcpp::Node
{
public:
    VideoDetectionNode();
    ~VideoDetectionNode();

private:
    void process_video();
    void detect_and_annotate(cv::Mat &frame);

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    cv::VideoCapture video_capture_;
    std::string video_path_;
    double playback_speed_;
    cv::Size display_size_;

    py::object model_;  // 使用 pybind11 的 YOLO 模型对象
    std::string model_path_;
};

#endif  // VIDEO_DETECTION_CPP_VIDEO_DETECTION_NODE_HPP_