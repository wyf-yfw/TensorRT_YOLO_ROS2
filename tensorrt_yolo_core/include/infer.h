#ifndef INFER_H
#define INFER_H

#include <opencv2/opencv.hpp>
#include "public.h"
#include "config.h"
#include "types.h"
#include "rclcpp/rclcpp.hpp"
#include "tensorrt_yolo_msg/msg/results.hpp"
#include "tensorrt_yolo_msg/msg/infer_result.hpp"
#include "tensorrt_yolo_msg/msg/key_point.hpp"

using namespace nvinfer1;



class YoloDetector
{
public:
    YoloDetector(rclcpp::Node::SharedPtr& node);
    ~YoloDetector();
    tensorrt_yolo_msg::msg::Results inference(cv::Mat& img);
    tensorrt_yolo_msg::msg::Results inference(cv::Mat& img, bool pose);
    rclcpp::Node::SharedPtr&       node_;  

private:
    void get_engine();
    void deserialize_engine();
    void serialize_engine();

private:

    Logger              gLogger;
    std::string         trtFile_;
    std::string         onnxFile_;

    int                 numClass_;
    float               nmsThresh_;
    float               confThresh_;
    int                 numKpt_;
    int                 kptDims_;
    int                 numBoxElement_;
    ICudaEngine *       engine;
    IRuntime *          runtime;
    IExecutionContext * context;

    cudaStream_t        stream;

    float *             outputData;
    std::vector<void *> vBufferD;
    float *             transposeDevice;
    float *             decodeDevice;

    int                 OUTPUT_CANDIDATES;  // 8400: 80 * 80 + 40 * 40 + 20 * 20
    cv::Mat*            img_;
};

#endif  // INFER_H
