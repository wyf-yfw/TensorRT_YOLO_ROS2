# ROS2中通过 TensorRT部署YOLOv8 目标检测模型

## 更新日志

### v4.0 -2024.8.27

- 完成ros2 Tensorrt 10下的兼容
- 分离msg和主函数部分

### v3.1 -2024.8.26

- 完成对TensorRT 10的兼容

### v3.0 - 2024.8.25

- 大量config参数从config.cpp文件转移到launch文件当中,方便参数调整
- 合并多个publish为一
- 增加pose检测

### v2.1 - 2024.8.22

- 合并d_camera_infer_node和camera_infer_node，统一使用camera_infer_node
- 增加depth变量

### v2.0 -  2024.8.19

- 加入bytetrack算法
- 增加d_camera_infer_node和camera_infer_node的track功能

### v1.1 - 2024.8.14

- 删除用于表示检测结果的type.h文件
- 增加infer_result.msg和results.msg文件用于表示和publish检测结果
- 实现检测结果在ros上的publish
- 增加d435i_yolo.launch文件
- 增加track变量
- 完善config文件
- 完善readme

### v1.0 - 2024.8.3

- 实现基本的相机和照片的目标检测功能

## 实现效果

### [目标检测](https://www.bilibili.com/video/BV1Niv9erEuw/?spm_id_from=333.788.recommend_more_video.0&vd_source=bb696fabd15eaa2a7c74687a5ff42a1b)


### [目标追踪](https://www.bilibili.com/video/BV1NmpSeeE2o/?spm_id_from=333.788.recommend_more_video.1&vd_source=bb696fabd15eaa2a7c74687a5ff42a1b)


### [pose检测](https://www.bilibili.com/video/BV1fzWCeBEPx/?vd_source=bb696fabd15eaa2a7c74687a5ff42a1b)

## 环境

#### ~~支持TensorRT 8~~

#### [支持TensorRT 10](https://github.com/wyf-yfw/TensorRT_YOLO_ROS2/releases/tag/v12.2.10)

#### [ROS1仓库](https://github.com/wyf-yfw/TensorRT_YOLO_ROS)

注意：

- 运行时可能报错 段错误(核心已转储)，这是你自己的opencv版本和ros默认的版本造成冲突导致的，删除自己的版本使用ros默认的opencv即可解决报错
- ~~目前仅支持TensorRT 8，使用10会报错~~

## 文件结构

── tensorrt_yolo_core

│   ├── CMakeLists.txt

│   ├── images

│   │   ├── bus.jpg

│   │   ├── dog.jpg

│   │   ├── eagle.jpg

│   │   ├── field.jpg

│   │   ├── giraffe.jpg

│   │   ├── herd_of_horses.jpg

│   │   ├── person.jpg

│   │   ├── room.jpg

│   │   ├── street.jpg

│   │   └── zidane.jpg

│   ├── include

│   │   ├── bytekalman_filter.h

│   │   ├── byte_tracker.h

│   │   ├── calibrator.h

│   │   ├── camera_infer.h

│   │   ├── config.h

│   │   ├── dataType.h

│   │   ├── image_infer.h

│   │   ├── infer.h

│   │   ├── lapjv.h

│   │   ├── postprocess.h

│   │   ├── preprocess.h

│   │   ├── public.h

│   │   ├── strack.h

│   │   ├── TensorRT_YOLO_ROS2

│   │   ├── types.h

│   │   └── utils.h

│   ├── launch

│   │   ├── d435i_yolo.launch.py

│   │   └── __pycache__

│   │       ├── d435i_yolo.launch.cpython-310.pyc

│   │       └── image_infer.launch.cpython-310.pyc

│   ├── onnx_model

│   │   ├── yolov8s.onnx

│   │   └── yolov8s-pose.plan

│   ├── package.xml

│   └── src

│       ├── bytekalman_filter.cpp

│       ├── byte_tracker.cpp

│       ├── calibrator.cpp

│       ├── camera_infer.cpp

│       ├── config.cpp

│       ├── image_infer.cpp

│       ├── infer.cpp

│       ├── lapjv.cpp

│       ├── postprocess.cu

│       ├── preprocess.cu

│       └── strack.cpp

└── tensorrt_yolo_msg

​    ├── CMakeLists.txt

​    ├── include

​    │   └── tensorrt_yolo_msg

​    ├── msg

​    │   ├── InferResult.msg

​    │   ├── KeyPoint.msg

​    │   └── Results.msg

​    ├── package.xml

​    └── src

## 导出ONNX模型

1. 安装 `YOLOv8`

```bash
pip install ultralytics
```

- 建议同时从 `GitHub` 上 clone 或下载一份 `YOLOv8` 源码到本地；
- 在本地 `YOLOv8`一级 `ultralytics` 目录下，新建 `weights` 目录，并且放入`.pt`模型

2. 安装onnx相关库

```bash
pip install onnx==1.12.0
pip install onnxsim==0.4.33
```

3. 导出onnx模型

- 可以在一级 `ultralytics` 目录下，新建 `export_onnx.py` 文件
- 向文件中写入如下内容：

```python
from ultralytics import YOLO

model = YOLO("./weights/Your.pt", task="detect")
path = model.export(format="onnx", simplify=True, device=0, opset=12, dynamic=False, imgsz=640)
```

- 运行 `python export_onnx.py` 后，会在 `weights` 目录下生成 `.onnx`

## 安装编译

1. 将仓库clone到自己的ros工作空间中；

   ```bash
   cd catkin_ws/src
   git clone https://github.com/wyf-yfw/TensorRT_YOLO_ROS2.git
   ```

2. 如果是自己数据集上训练得到的模型，记得更改 `src/config.cpp` 中的相关配置，所有的配置信息全部都包含在`config.cpp`中；

3. 确认 `CMakeLists.txt` 文件中 `cuda` 和 `tensorrt` 库的路径，与自己环境要对应，一般情况下是不需修改的；

4. 将已导出的 `onnx` 模型拷贝到 `onnx_model` 目录下

5. 编译工作空间

## 运行节点

目前共有两个节点，分别是image_infer_node、camera_infer_node

先运行自己的相机节点，然后运行相应的推理节点

d435i相机可以直接运行launch文件

```bash
ros2 launch tensorrt_yolo_core d435i_yolo.launch.py
```

在launch文件中调整自己的参数，需要移植可以直接将下面这部分复制到自己的launch文件当中

```python
def generate_launch_description():
    
    return LaunchDescription(declare_configurable_parameters(configurable_parameters) + [
        OpaqueFunction(function=launch_setup, kwargs = {'params' : 		set_configurable_parameters(configurable_parameters)}),
        
        DeclareLaunchArgument(
            'plan_file',
            default_value='/home/wyf/colcon_ws/src/tensorrt_yolo_ros2/tensorrt_yolo_core/onnx_model/yolov8s-pose.plan',
            description='Path to the .plan file'
        ),
        DeclareLaunchArgument(
            'onnx_file',
      	 default_value='/home/wyf/colcon_ws/src/tensorrt_yolo_ros2/tensorrt_yolo_core/onnx_model/yolov8s.onnx',
            description='Path to the .onnx file'
        ),
        DeclareLaunchArgument(
            'nms_thresh',
            default_value='0.7',
            description='Non-Maximum Suppression threshold'
        ),
        DeclareLaunchArgument(
            'conf_thresh',
            default_value='0.7',
            description='Confidence threshold'
        ),
        DeclareLaunchArgument(
            'num_class',
            default_value='1',
            description='Number of classes for object detection'
        ),
        DeclareLaunchArgument(
            'num_kpt',
            default_value='17',
            description='Number of keypoints for pose detection'
        ),
        DeclareLaunchArgument(
            'kpt_dims',
            default_value='3',
            description='Dimensions of keypoints for pose detection'
        ),
        DeclareLaunchArgument(
            'track',
            default_value='true',
            description='Whether to enable tracking'
        ),
        DeclareLaunchArgument(
            'depth',
            default_value='false',
            description='Whether to enable depth camera'
        ),
        DeclareLaunchArgument(
            'pose',
            default_value='true',
            description='Whether to enable pose detection'
        ),
        DeclareLaunchArgument(
            'rgb_image_topic',
            default_value='/camera/camera/color/image_raw',
            description='RGB image topic'
        ),
        DeclareLaunchArgument(
            'depth_image_topic',
            default_value='/camera/camera/depth/image_rect_raw',
            description='Depth image topic'
        ),
```

## 节点订阅数据

```
const std::string rgbImageTopic = "/camera/color/image_raw"; //ros中image的topic
const std::string depthImageTopic = "/camera/depth/image_rect_raw"; //ros中depth image的topic
```

## 节点发布数据

节点对外publisher有一个，为infer_results，发布的内容均为一个列表，列表中的元素结构是

```cpp
float32[4] bbox
float32 conf
int32 classId
float32[3] coordinate //d_camera_infer_node发布三维空间坐标，camera_infer_node发布二维坐标，z=0
int32 Id // 关闭追踪模式默认为0，开启目标追踪为当前追踪的id值
float32 kpts // 存储未缩放到原始图像上的关键点数据
KeyPoint[] kpts // 存储经过缩放处理后的关键点数据，为了将关键点坐标映射回原始图像中的坐标系

```



### TODO

- ~~完成目标追踪~~
- ~~完成pose检测~~
- ~~完成ros1 Tensorrt 10兼容~~
- ~~完成ros2 Tensorrt 10兼容~~
- 完成ros2 Tensorrt 8兼容
