//
// Created by wyf on 24-8-2.
//

//#ifndef CAMERA_INFER_H
//#define CAMERA_INFER_H
#include "rclcpp/rclcpp.hpp"
#include "utils.h"
#include "infer.h"
#include <utility>
#include "config.h"
#include <sensor_msgs/msg/image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/point.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <boost/thread/thread.hpp>
#include "tensorrt_yolo_msg/msg/results.hpp"
#include "tensorrt_yolo_msg/msg/infer_result.hpp"
#include "tensorrt_yolo_msg/msg/key_point.hpp"
#include "byte_tracker.h" //bytetrack
#include <chrono>


class CameraInfer:public YoloDetector, public BYTEtracker{
public:
    CameraInfer(rclcpp::Node::SharedPtr& node);
    void draw_image(cv::Mat& img);
    void draw_image(cv::Mat& img, cv::Mat& depth_img);
    void bytetrack();
    void img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void process_images();

protected:
    
    std::string rgbImageTopic_;
    std::string depthImageTopic_;

    int frame_count_;
    sensor_msgs::msg::Image::SharedPtr rgb_image_;
    sensor_msgs::msg::Image::SharedPtr depth_image_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;;
    std::chrono::high_resolution_clock::time_point start_time_;
    image_transport::ImageTransport it_;
    tensorrt_yolo_msg::msg::Results results_msg_;
    std::vector<strack> output_stracks_;
    std::mutex mutex_;
    const uint32_t time_threshold_ = 1000000; // 时间戳对齐阈值，单位为纳秒
    // 追踪开关，默认为false
    bool track_;
    bool depth_;
    bool pose_;
};
//#endif //CAMERA_INFER_H
