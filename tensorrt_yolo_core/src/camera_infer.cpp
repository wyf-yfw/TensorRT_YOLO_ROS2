//
// Created by wyf on 24-8-1.
//
#include "camera_infer.h"
#include "rclcpp/rclcpp.hpp"
#include "utils.h"
#include "infer.h"
#include <utility>
#include <functional>
#include "config.h"
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/point.h>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include "byte_tracker.h" //bytetrack
#include <chrono>


CameraInfer::CameraInfer(rclcpp::Node::SharedPtr& node):
it_(node),
frame_count_(0),
start_time_(std::chrono::high_resolution_clock::now()),
YoloDetector(node)
{
    // 模式选择
    node->declare_parameter<bool>("track", false);
    node->declare_parameter<bool>("depth", false);
    node->declare_parameter<bool>("pose", false);
    node->get_parameter("track", track_);
    node->get_parameter("depth", depth_);
    node->get_parameter("pose", pose_);

   
    //订阅rgb图像话题
    node->declare_parameter<std::string>("rgbImageTopic", "/camera/camera/color/image_raw");
    node->get_parameter("rgbImageTopic", rgbImageTopic_);

    if(!depth_) {
        rgb_subscription_ = node->create_subscription<sensor_msgs::msg::Image>(
        rgbImageTopic_,  
        10, 
        std::bind(&CameraInfer::img_callback, this, std::placeholders::_1));
    }else{
        //订阅depth图像话题
        node->declare_parameter<std::string>("depthImageTopic", "/camera/camera/depth/image_rect_raw");
        node->get_parameter("depthImageTopic", depthImageTopic_);
        
        rgb_subscription_ = node->create_subscription<sensor_msgs::msg::Image>(
            rgbImageTopic_,
            10,
            std::bind(&CameraInfer::rgb_callback, this, std::placeholders::_1)
        );

        depth_subscription_ = node->create_subscription<sensor_msgs::msg::Image>(
            depthImageTopic_,
            10,
            std::bind(&CameraInfer::depth_callback, this, std::placeholders::_1)
        );
       
    }

}
void CameraInfer::draw_image(cv::Mat& img){
    if(track_){
        draw_tracking_id(img, output_stracks_, results_msg_);
    }
    if(pose_){
        draw_pose_points(img, results_msg_);
    }
    // 绘制图像
    draw_fps(img, frame_count_, start_time_);
    draw_detection_box(img,results_msg_);
    calculate_draw_center_point(img, results_msg_);
}
void CameraInfer::draw_image(cv::Mat& img, cv::Mat& depth_img){

    if(track_){
        draw_tracking_id(img, output_stracks_, results_msg_);
    }
    if(pose_){
        draw_pose_points(img, results_msg_);
    }
    draw_fps(img, frame_count_, start_time_);
    draw_detection_box(img,results_msg_);
    calculate_draw_center_point(img,depth_img, results_msg_);
}
void CameraInfer::bytetrack()
{
    // 需要跟踪的目标类型
    std::vector<detect_result> objects;
    // 用于存储结果的转换
    std::vector<detect_result> results;
    // result格式转换
    yolo_detece2detect_result(results_msg_, results);
    // 判断需要跟踪的目标类型
    for (detect_result dr : results) {
        for (int tc: track_classes) {
            if (dr.classId == tc)
            {
                objects.push_back(dr);
            }
        }
    }
    // 目标跟踪
     output_stracks_ = update(objects);
    // 清除用于原始的results_msg_
    results_msg_.results.clear();
}
void CameraInfer::img_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
// 使用 cv_bridge 转换 ROS 图像消息到 OpenCV 图像
    cv::Mat img = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
    // 推理
    if (img.empty()) return;
    if(pose_){
        results_msg_ = inference(img, true);
    }else{
            RCLCPP_INFO(node_->get_logger(), "4444444444444444444444444444444444444444");

        results_msg_ = inference(img);
    }
    if(track_){
        bytetrack();
    }
    draw_image(img);

    std::shared_ptr<rclcpp::Publisher<tensorrt_yolo_msg::msg::Results>> publisher = node_->create_publisher<tensorrt_yolo_msg::msg::Results>("infer_results", 10);
    publisher->publish(results_msg_);
    cv::imshow("imgshow", img);
    cv::waitKey(30);
}
void CameraInfer::rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
        rgb_image_ = msg;
        process_images();
}
void CameraInfer::depth_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
   std::lock_guard<std::mutex> lock(mutex_);
        rgb_image_ = msg;
        process_images();
}
void CameraInfer::process_images()
{
     if (rgb_image_ && depth_image_)
    {
        // 将时间戳转换为 std::chrono::nanoseconds
        auto rgb_time = std::chrono::nanoseconds(rgb_image_->header.stamp.nanosec);
        auto depth_time = std::chrono::nanoseconds(depth_image_->header.stamp.nanosec);

        // 计算时间差的绝对值
        auto time_diff = std::abs(rgb_time.count() - depth_time.count());
        if (time_diff < time_threshold_)
        {
            // 转换 ROS 图像消息到 OpenCV 格式
            cv_bridge::CvImagePtr cv_rgb_image;
            cv_bridge::CvImagePtr cv_depth_image;

            try
            {
                cv_rgb_image = cv_bridge::toCvCopy(rgb_image_, sensor_msgs::image_encodings::BGR8);
                cv_depth_image = cv_bridge::toCvCopy(depth_image_, sensor_msgs::image_encodings::TYPE_16UC1);
            }
            catch (cv_bridge::Exception& e)
            {
                RCLCPP_ERROR(node_->get_logger(), "CV Bridge error: %s", e.what());
                return;
            }

            // 处理对齐后的图像
            cv::Mat img = cv_rgb_image->image;
            cv::Mat depth_cv_image = cv_depth_image->image;

            // 在此处对齐后的图像进行处理   
            if(pose_){
                results_msg_ = inference(img, true);
            }else{
                results_msg_ = inference(img);
            }
            if(track_){
                bytetrack();
            }
            draw_image(img, depth_cv_image);
            std::shared_ptr<rclcpp::Publisher<tensorrt_yolo_msg::msg::Results>> publisher = node_->create_publisher<tensorrt_yolo_msg::msg::Results>("infer_results", 10);
            publisher->publish(results_msg_);
            cv::imshow("imgshow", img);
            cv::waitKey(30);
        }
    }
}   


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("camera_infert_node");
    CameraInfer c(node);
    rclcpp::spin(node);
    rclcpp::shutdown;
}