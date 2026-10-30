#pragma once
#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include <array>
#include <cstdio>
#include <future>

#include <sophus/se3.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace ORB_SLAM3
{
    class System;
}

namespace android_slam
{

    struct Image
    {
        std::vector<uint8_t> data;
        int64_t time_stamp = 0;
        double readback_ms = 0.0;
        uint64_t duplicate_polls = 0;
        uint64_t backward_images = 0;
    };

    struct ImuPoint
    {
        float ax;
        float ay;
        float az;
        float wx;
        float wy;
        float wz;
        int64_t time_stamp;
    };

    // 坐标契约（GNSS 融合前固化）：
    //   raw SLAM: 原点、方向和单目尺度由初始化确定，尚无地理意义。
    //   现有对齐目标/显示轴序: (east, up, north)
    //   输出(PC/文件): 交换内部 y/z；只有完成地理对齐后才可解释为 ENU。
    //   GNSS 的 ENU: (east, north, up)，右手系、米；融合数学在 ENU 中完成。
    struct TrackingResult
    {
        struct Pos
        {
            float x;
            float y;
            float z;
        };

        std::array<float, 16> last_pose{};
        int64_t time_stamp = 0;   // 该帧相机时间戳(单调纳秒)，用于与 GNSS 对齐
        std::vector<Pos> trajectory;
        std::vector<Pos> map_points;
        std::string tracking_status;

        float processing_delta_time = 0.0f;
        bool pose_valid = false;
    };

    // 内部坐标 -> 输出坐标: (east, up, north) -> (east, north, up)
    inline TrackingResult::Pos toOutputPos(const TrackingResult::Pos& p)
    {
        return TrackingResult::Pos{ p.x, p.z, p.y };
    }

    // 仅重排位置分量，不执行 raw SLAM 地理对齐；不能用于旋转矩阵/姿态。
    inline Eigen::Vector3f enuToInternal(float east, float north, float up)
    {
        return Eigen::Vector3f(east, up, north);
    }

    class SlamKernel
    {
    private:
        static constexpr int64_t k_nano_second_in_one_second = 1000000000;
        static constexpr double k_nano_sec_to_sec_radio = 1.0 / (double)(k_nano_second_in_one_second);

    public:
        SlamKernel(int32_t img_width, int32_t img_height, std::string vocabulary_data, int64_t begin_time_stamp);
        SlamKernel(const SlamKernel&) = delete;
        SlamKernel& operator=(const SlamKernel&) = delete;
        ~SlamKernel();

        TrackingResult handleData(const std::vector<Image>& images, const std::vector<ImuPoint>& imus);

        void reset();

        // 锚点对齐：raw SLAM 坐标 <-> 真实世界坐标，Umeyama 相似变换
        // addSlamAnchor: 记录 raw SLAM 锚点(内部 SLAM 坐标)
        void addSlamAnchor(float x, float y, float z);
        // setWorldAnchors: 世界锚点，单位为内部世界坐标(east,up,north)；
        //   GNSS 阶段请先用 enuToInternal() 把 ENU(east,north,up) 转换过来。
        void setWorldAnchors(const std::vector<Eigen::Vector3f>& anchors);
        void clearAnchors();
        int getSlamAnchorCount() const;
        int getWorldAnchorCount() const;
        bool computeAlignment();
        void resetAlignment();
        bool isAligned() const { return m_aligned; }
        float getSimScale() const { return m_aligned ? m_sim_scale : 1.0f; }

    private:
        int32_t m_width;
        int32_t m_height;
        const int64_t m_begin_time_stamp;

        std::unique_ptr<::ORB_SLAM3::System> m_orb_slam;

        std::chrono::steady_clock::time_point m_last_time;

        std::vector<Eigen::Vector3f> m_anchor_slam;
        std::vector<Eigen::Vector3f> m_anchor_world;
        Eigen::Matrix3f m_R = Eigen::Matrix3f::Identity();
        Eigen::Vector3f m_t = Eigen::Vector3f::Zero();
        float m_sim_scale = 1.0f;
        bool m_aligned = false;

        // Diagnostic output is separate from the existing trajectory log.
        FILE* m_debug_file = nullptr;
        int64_t m_last_input_timestamp = -1;
        uint64_t m_debug_frame_index = 0;
        void writeDebugLine(const std::string& line, bool flush);
        std::future<void> m_debug_image_job;
        cv::Mat m_debug_previous_image;
        int64_t m_debug_previous_image_ts = 0;
        int m_debug_loss_snapshots = 0;
        int m_debug_weak_stage = 0;
        std::string m_debug_frames_dir;

    };

}
