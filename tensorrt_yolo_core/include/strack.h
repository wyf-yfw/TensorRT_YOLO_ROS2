#pragma once

#include <opencv2/opencv.hpp>
#include "bytekalman_filter.h"
#include "tensorrt_yolo_msg/msg/results.hpp"
#include "tensorrt_yolo_msg/msg/infer_result.hpp"
#include "tensorrt_yolo_msg/msg/key_point.hpp"
enum TrackState { New = 0, Tracked, Lost, Removed };

class strack
{
public:
	strack(std::vector<float> tlwh_, float score, int class_id,std::vector<float> kpts, std::vector<tensorrt_yolo_msg::msg::KeyPoint> vKpts);
	~strack();

	 std::vector<float> static tlbr_to_tlwh( std::vector<float> &tlbr);
	void static multi_predict(std::vector<strack*> &stracks, byte_kalman::ByteKalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	 std::vector<float> tlwh_to_xyah( std::vector<float> tlwh_tmp);
	 std::vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();

	void activate(byte_kalman::ByteKalmanFilter &kalman_filter, int frame_id);
	void re_activate(strack &new_track, int frame_id, bool new_id = false);
	void update(strack &new_track, int frame_id);

public:
	bool is_activated;
	int track_id;
	int state;

	 std::vector<float> _tlwh;
	 std::vector<float> tlwh;
	 std::vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;
    int classId_;
    KAL_MEAN mean;
	KAL_COVA covariance;
	float score;
    std::vector<float> kpts_;
    std::vector<tensorrt_yolo_msg::msg::KeyPoint> vKpts_;
private:
	byte_kalman::ByteKalmanFilter kalman_filter;
};