//
// Created by wyf on 24-8-1.
//

#ifndef IMAGE_INFER_H
#define IMAGE_INFER_H

#include "utils.h"
#include "infer.h"
#include <unistd.h>
#include "tensorrt_yolo_msg/msg/results.hpp"
#include "tensorrt_yolo_msg/msg/infer_result.hpp"
#include "tensorrt_yolo_msg/msg/key_point.hpp"

class ImageInfer : public YoloDetector{
public:
    ImageInfer(rclcpp::Node::SharedPtr& node);
    void SaveResult(const cv::Mat& img, int num);
    int run();
    void draw_image(cv::Mat& img, tensorrt_yolo_msg::msg::InferResult inferResult);
private:
    const char* imageDir_;
    std::string imagePath_;
    std::vector<std::string> file_names_;
    bool save_;
};
#endif //IMAGE_INFER_H
