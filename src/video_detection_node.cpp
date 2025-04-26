#include "video_detection_cpp/video_detection_node.hpp"
#include <pybind11/embed.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <fstream>
namespace py = pybind11;
VideoDetectionNode::VideoDetectionNode()
    : Node("video_detection_node")
{
    // 确保只初始化一次
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }
    
    std::string package_share_directory = ament_index_cpp::get_package_share_directory("video_detection_cpp");
    model_path_ = package_share_directory + "/models/best.pt";
    RCLCPP_INFO(this->get_logger(), "Model path: %s", model_path_.c_str());

    this->declare_parameter<std::string>("video_path", "input_video.mp4");
    this->declare_parameter<std::string>("model_path", "best.pt");
    this->declare_parameter<double>("playback_speed", 1.0);
    this->declare_parameter<std::vector<long>>("display_size", {800, 600});

    this->get_parameter("video_path", video_path_);
    // this->get_parameter("model_path", model_path_);
    this->get_parameter("playback_speed", playback_speed_);
    std::vector<long> display_size_vec;
    this->get_parameter("display_size", display_size_vec);
    display_size_ = cv::Size(static_cast<int>(display_size_vec[0]), static_cast<int>(display_size_vec[1]));

    try {
        py::module ultralytics = py::module::import("ultralytics");
        py::object YOLO = ultralytics.attr("YOLO");
        model_ = YOLO(model_path_);
        RCLCPP_INFO(this->get_logger(), "YOLO model loaded successfully.");
    } catch (const py::error_already_set &e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load YOLO model: %s", e.what());
        rclcpp::shutdown();
        return;
    }

    // 初始化视频捕获
    RCLCPP_INFO(this->get_logger(), "Trying to open video file: %s", video_path_.c_str());
    
    // 尝试多种可能的路径
    std::vector<std::string> possible_paths = {
        "input_video.mp4",  // 相对路径
        "/home/radar/ros2_ws/input_video.mp4",  // 绝对路径 1
        "/home/radar/ros2_ws/src/video_detection_cpp/input_video.mp4",  // 绝对路径 2
        "/home/radar/input_video.mp4"  // 绝对路径 3
    };

    bool video_opened = false;
    for (const auto& path : possible_paths) {
        RCLCPP_INFO(this->get_logger(), "Checking if file exists at: %s", path.c_str());
        
        // 检查文件是否存在
        std::ifstream file_check(path);
        if (!file_check.good()) {
            RCLCPP_WARN(this->get_logger(), "File does not exist at: %s", path.c_str());
            continue;
        }
        
        RCLCPP_INFO(this->get_logger(), "File exists! Trying to open with OpenCV: %s", path.c_str());
        
        // 尝试使用 OpenCV 打开
        video_capture_.release(); // 确保先释放之前的实例
        video_capture_ = cv::VideoCapture();
        
        // 打开文件但设置超时，避免无限等待
        video_capture_.open(path, cv::CAP_ANY);
        
        // 检查是否成功打开
        if (!video_capture_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV cannot open file despite it exists: %s", path.c_str());
            continue;
        }
        
        // 尝试读取第一帧以确认视频确实可用
        cv::Mat test_frame;
        if (!video_capture_.read(test_frame)) {
            RCLCPP_ERROR(this->get_logger(), "File opened but cannot read first frame: %s", path.c_str());
            video_capture_.release();
            continue;
        }
        
        RCLCPP_INFO(this->get_logger(), 
                    "Video opened successfully at %s! Resolution: %dx%d", 
                    path.c_str(), test_frame.cols, test_frame.rows);
        video_opened = true;
        break;
    }

    if (!video_opened) {
        RCLCPP_FATAL(this->get_logger(), "Failed to open video file from any path. Please check if the file exists and has correct format.");
        rclcpp::shutdown();
        return;
    }

    // 创建图像发布器
    image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("detection_image", 10);

    // 创建定时器
    auto timer_callback = [this]() { this->process_video(); };
    this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000 / (30 * playback_speed_))), timer_callback);
}

VideoDetectionNode::~VideoDetectionNode()
{
    // 先释放与 Python 相关的资源
    {
        py::gil_scoped_acquire acquire;
        model_ = py::none();  // 明确释放 YOLO 模型
    }
    
    // 释放 OpenCV 资源
    video_capture_.release();
    
    // 注意：不要在析构函数中调用 finalize_interpreter，因为可能有多个对象使用 Python
    // py::finalize_interpreter() 应该只在主程序结束时调用一次
}

void VideoDetectionNode::process_video()
{
    RCLCPP_INFO(this->get_logger(), "Beginning to process video frame");
    
    cv::Mat frame;
    if (!video_capture_.read(frame)) {
        RCLCPP_INFO(this->get_logger(), "End of video stream or failed to read frame.");
        rclcpp::shutdown();
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Successfully read frame, size: %dx%d", frame.cols, frame.rows);
    
    try {
        // 检测并标注
        RCLCPP_INFO(this->get_logger(), "Beginning detection and annotation");
        detect_and_annotate(frame);
        RCLCPP_INFO(this->get_logger(), "Detection completed successfully");

        // 调整显示大小
        cv::resize(frame, frame, display_size_);
        
        // 发布图像
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        image_publisher_->publish(*msg);
        RCLCPP_INFO(this->get_logger(), "Published annotated image");
        
        // WSL环境可能无法显示窗口，先注释掉
        // cv::imshow("Detection", frame);
        // if (cv::waitKey(1) == 'q') {
        //     RCLCPP_INFO(this->get_logger(), "User requested shutdown.");
        //     rclcpp::shutdown();
        // }
    }
    catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in process_video: %s", e.what());
    }
    catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in process_video");
    }
}

void VideoDetectionNode::detect_and_annotate(cv::Mat &frame)
{
    try {
        RCLCPP_INFO(this->get_logger(), "Acquiring Python GIL");
        // 获取 GIL
        py::gil_scoped_acquire acquire;
        
        RCLCPP_INFO(this->get_logger(), "Converting frame to RGB for YOLO");
        // 将 OpenCV 图像转换为 Python 对象
        py::module cv2 = py::module::import("cv2");
        py::object frame_py = cv2.attr("cvtColor")(frame, cv2.attr("COLOR_BGR2RGB"));

        RCLCPP_INFO(this->get_logger(), "Running YOLO prediction");
        // 使用 YOLO 模型进行预测
        py::object results = model_.attr("predict")(frame_py);
        RCLCPP_INFO(this->get_logger(), "YOLO prediction completed");

        // 遍历检测结果
        int detection_count = 0;
        for (auto result : results) {
            detection_count++;
            // ...处理结果...
        }
        
        RCLCPP_INFO(this->get_logger(), "Processed %d detections", detection_count);
        // GIL 会在代码块结束时自动释放
    } catch (const py::error_already_set &e) {
        RCLCPP_ERROR(this->get_logger(), "Error during YOLO inference: %s", e.what());
    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in detect_and_annotate: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in detect_and_annotate");
    }
}