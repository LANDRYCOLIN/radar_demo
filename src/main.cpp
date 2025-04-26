#include "video_detection_cpp/video_detection_node.hpp"
#include <rclcpp/rclcpp.hpp>
#include <signal.h>
#include <pybind11/embed.h>

namespace py = pybind11;
std::shared_ptr<VideoDetectionNode> node_ptr = nullptr;

void signal_handler(int signum)
{
    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signal_handler);
    rclcpp::init(argc, argv);
    node_ptr = std::make_shared<VideoDetectionNode>();
    rclcpp::spin(node_ptr);
    node_ptr = nullptr;
    if (Py_IsInitialized()) {
        py::finalize_interpreter();
    }
    
    rclcpp::shutdown();
    return 0;
}