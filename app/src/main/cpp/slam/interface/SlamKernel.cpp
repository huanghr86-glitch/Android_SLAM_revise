#include "SlamKernel.h"
#include <iostream>
#include <set>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include <core/System.h>
#include <utils/Settings.h>
#include <feature/ORBVocabulary.h>
#include "frame/KeyFrame.h"
#include "map/MapPoint.h"

#include <android/log.h>
#define DEBUG_INFO(...) ::__android_log_print(ANDROID_LOG_INFO, "[Android SLAM DEBUG]", __VA_ARGS__)

namespace android_slam
{

    SlamKernel::SlamKernel(int32_t img_width, int32_t img_height, std::string vocabulary_data, int64_t begin_time_stamp)
    : m_width(img_width)
    , m_height(img_height)
    , m_begin_time_stamp(begin_time_stamp)
    , m_last_time(std::chrono::steady_clock::now())
    {
        ORB_SLAM3::Settings::SettingDesc desc{};
        desc.sensor = ORB_SLAM3::System::eSensor::MONOCULAR;
        desc.cameraInfo.cameraType = ORB_SLAM3::Settings::CameraType::PinHole;
        //手机相机标定结果
        desc.cameraInfo.fx = 520.4708852845047f;
        desc.cameraInfo.fy = 523.7118931356812f;
        desc.cameraInfo.cx = 319.57018786503784f;
        desc.cameraInfo.cy = 238.20910561697508f;
        desc.distortion->k1 = -0.02640806687490354f;
        desc.distortion->k2 = 0.2033386379661157f;
        desc.distortion->p1 = 6.519121558220371e-06f;
        desc.distortion->p2 = -0.0009077232655942344f;
        /*
        desc.cameraInfo.fx = 458.654f;
        desc.cameraInfo.fy = 457.296f;
        desc.cameraInfo.cx = 367.215f;
        desc.cameraInfo.cy = 248.375f;
        desc.distortion->k1 = -0.28340811f;
        desc.distortion->k2 = 0.07395907f;
        desc.distortion->p1 = 0.00019359f;
        desc.distortion->p2 = 1.76187114e-05f;
         */
        desc.imageInfo.width = img_width;
        desc.imageInfo.height = img_height;
        desc.imageInfo.newWidth = 640;
        desc.imageInfo.newHeight = 480;
        desc.imageInfo.fps = 20;
        desc.imageInfo.bRGB = true;
        //手机imu标定结果
        desc.imuInfo.noiseGyro = 0.00037129946286789615f;
        desc.imuInfo.noiseAcc = 0.008926024139864043f;
        desc.imuInfo.gyroWalk = 4.310313880423259e-06f;
        desc.imuInfo.accWalk = 0.0002903411803301644f;
        /*
        desc.imuInfo.noiseGyro = 1.7e-4f;
        desc.imuInfo.noiseAcc = 2.0000e-3f;
        desc.imuInfo.gyroWalk = 1.9393e-05f;
        desc.imuInfo.accWalk = 3.0000e-03f;
         */
        desc.imuInfo.frequency = 200.0f;
        //手机联合标定结果
        desc.imuInfo.cvTbc = static_cast<cv::Mat>(cv::Mat_<float>(4, 4) << 0.0148655429818f, -0.999880929698f, 0.00414029679422f, -0.0216401454975f,
                0.999557249008f, 0.0149672133247f, 0.025715529948f, -0.064676986768f,
                -0.0257744366974f, 0.00375618835797f, 0.999660727178f, 0.00981073058949f,
                0.0f, 0.0f, 0.0f, 1.0f
        );
        //desc.imuInfo.cvTbc = static_cast<cv::Mat>(cv::Mat_<float>(4, 4) << +0.0148655429818f, -0.99988092969800f, +0.00414029679422f, -0.02164014549750f, +0.9995572490080f, +0.01496721332470f, +0.02571552994800f, -0.06467698676800f, -0.0257744366974f, +0.00375618835797f, +0.99966072717800f, +0.00981073058949f, 0.0f, 0.0f, 0.0f, 1.0f
        //);
        desc.imuInfo.bInsertKFsWhenLost = true;
        desc.orbInfo.nFeatures = 1000;
        desc.orbInfo.scaleFactor = 1.2f;
        desc.orbInfo.nLevels = 8;
        desc.orbInfo.initThFAST = 20;
        desc.orbInfo.minThFAST = 7;
        desc.viewerInfo.keyframeSize = 0.05f;
        desc.viewerInfo.keyframeLineWidth = 1.0f;
        desc.viewerInfo.graphLineWidth = 0.9f;
        desc.viewerInfo.pointSize = 2.0f;
        desc.viewerInfo.cameraSize = 0.08f;
        desc.viewerInfo.cameraLineWidth = 3.0f;
        desc.viewerInfo.viewPointX = 0.0f;
        desc.viewerInfo.viewPointY = -0.7f;
        desc.viewerInfo.viewPointZ = -3.5f;
        desc.viewerInfo.viewPointF = 500.0f;
        desc.viewerInfo.imageViewerScale = 1.0f;
        auto slam_settings = new ::ORB_SLAM3::Settings(desc);


        auto vocabulary = new ::ORB_SLAM3::ORBVocabulary();
        vocabulary->loadFromAndroidTextFile(std::move(vocabulary_data));


        m_orb_slam = std::make_unique<::ORB_SLAM3::System>(
        vocabulary, slam_settings, static_cast<const ORB_SLAM3::System::eSensor>(desc.sensor));


    }

    // unique_ptr needs to know how to delete the ptr, so the dtor should be impl with the definition of the ptr class.
    SlamKernel::~SlamKernel()
    {
        m_orb_slam->Shutdown();
    }

    void SlamKernel::reset()
    {
        m_orb_slam->Reset();
    }

    TrackingResult SlamKernel::handleData(const std::vector<Image>& images, const std::vector<ImuPoint>& imus)
    {
        TrackingResult res;

        if (images.empty())
        {
            res.tracking_status = "NO_IMAGES_YET";
            return res;
        }

        const Image& image = images[0];
        res.time_stamp = image.time_stamp;
        assert((image.time_stamp >= m_begin_time_stamp) && "Invalid time stamp.");

        cv::Mat cv_image(m_height, m_width, CV_8UC3);
        memcpy(cv_image.data, image.data.data(), sizeof(uint8_t) * image.data.size());


        double image_time_stamp = (double) (image.time_stamp - m_begin_time_stamp) * k_nano_sec_to_sec_radio;
        Sophus::SE3f pose = m_orb_slam->TrackMonocular(cv_image, image_time_stamp);


        Eigen::Matrix4f mat_pose = pose.matrix();
        if (m_aligned && m_sim_scale > 1e-6f)
        {
            // Tcw in SLAM world: Xc = Rcw * Xs + tcw.
            // Output world: Xo = s * R * Xs + t.
            const Eigen::Matrix3f Rcw = mat_pose.block<3, 3>(0, 0);
            const Eigen::Vector3f tcw = mat_pose.block<3, 1>(0, 3);
            const float inv_scale = 1.0f / m_sim_scale;
            const Eigen::Matrix3f Rcw_out = inv_scale * (Rcw * m_R.transpose());
            const Eigen::Vector3f tcw_out = tcw - Rcw_out * m_t;

            mat_pose.setIdentity();
            mat_pose.block<3, 3>(0, 0) = Rcw_out;
            mat_pose.block<3, 1>(0, 3) = tcw_out;
        }
        {
            res.last_pose[+0] = mat_pose(0, 0);
            res.last_pose[+1] = mat_pose(1, 0);
            res.last_pose[+2] = mat_pose(2, 0);
            res.last_pose[+3] = mat_pose(3, 0);
            res.last_pose[+4] = mat_pose(0, 1);
            res.last_pose[+5] = mat_pose(1, 1);
            res.last_pose[+6] = mat_pose(2, 1);
            res.last_pose[+7] = mat_pose(3, 1);
            res.last_pose[+8] = mat_pose(0, 2);
            res.last_pose[+9] = mat_pose(1, 2);
            res.last_pose[10] = mat_pose(2, 2);
            res.last_pose[11] = mat_pose(3, 2);
            res.last_pose[12] = mat_pose(0, 3);
            res.last_pose[13] = mat_pose(1, 3);
            res.last_pose[14] = mat_pose(2, 3);
            res.last_pose[15] = mat_pose(3, 3);
        }

        ORB_SLAM3::Map* active_map = m_orb_slam->getAtlas().GetCurrentMap();
        if (active_map)
        {
            auto transform_point = [&](const Eigen::Vector3f& p)
            {
                Eigen::Vector3f q = p;
                if (m_aligned)
                    q = m_sim_scale * (m_R * q) + m_t;
                return TrackingResult::Pos{ q.x(), q.y(), q.z() };
            };

            {
                std::vector<ORB_SLAM3::KeyFrame*> key_frames = active_map->GetAllKeyFrames();

                static auto key_frame_cmp = [](const ORB_SLAM3::KeyFrame* kf1, const ORB_SLAM3::KeyFrame* kf2)
                {
                    return kf1->mnFrameId < kf2->mnFrameId;
                };
                std::set<ORB_SLAM3::KeyFrame*, decltype(key_frame_cmp)> kf_set(key_frame_cmp);
                for (ORB_SLAM3::KeyFrame* kf: key_frames)
                {
                    if (!kf || kf->isBad())
                        continue;
                    kf_set.insert(kf);
                }

                for (ORB_SLAM3::KeyFrame* kf: kf_set)
                {
                    if (!kf || kf->isBad())
                        continue;

                    Eigen::Vector3f position = kf->GetPoseInverse().translation();
                    res.trajectory.push_back(transform_point(position));
                }

                Eigen::Vector3f last_position = pose.inverse().translation();
                res.trajectory.push_back(transform_point(last_position));
            }

            {
                std::vector<ORB_SLAM3::MapPoint*> local_mps = active_map->GetReferenceMapPoints();
                std::unordered_set<ORB_SLAM3::MapPoint*> local_mp_ust(local_mps.begin(), local_mps.end());

                for (ORB_SLAM3::MapPoint* mp: local_mps)
                {
                    if (!mp || mp->isBad())
                        continue;

                    res.map_points.push_back(transform_point(mp->GetWorldPos()));
                }

                std::vector<ORB_SLAM3::MapPoint*> all_mps = active_map->GetAllMapPoints();
                for (ORB_SLAM3::MapPoint* mp: all_mps)
                {
                    if (!mp || mp->isBad() || local_mp_ust.find(mp) != local_mp_ust.end())
                        continue;

                    res.map_points.push_back(transform_point(mp->GetWorldPos()));
                }
            }
        }

        {
            int status = m_orb_slam->getTrackingState();
            switch (status)
            {
                case -1:
                    res.tracking_status = "SYSTEM_NOT_READY";
                    break;
                case 0:
                    res.tracking_status = "NO_IMAGES_YET";
                    break;
                case 1:
                    res.tracking_status = "NOT_INITIALIZED";
                    break;
                case 2:
                    res.tracking_status = "OK";
                    break;
                case 3:
                    res.tracking_status = "RECENTLY_LOST";
                    break;
                case 4:
                    res.tracking_status = "LOST";
                    break;
                case 5:
                    res.tracking_status = "OK_KLT";
                    break;
                default:
                    break;
            }
        }

        {
            const std::chrono::steady_clock::time_point curr_time = m_last_time;
            m_last_time = std::chrono::steady_clock::now();

            res.processing_delta_time = std::chrono::duration<float>(m_last_time - curr_time).count();
        }

        return res;
    }

    void SlamKernel::addSlamAnchor(float x, float y, float z)
    {
        m_anchor_slam.push_back(Eigen::Vector3f(x, y, z));
    }

    void SlamKernel::setWorldAnchors(const std::vector<Eigen::Vector3f>& anchors)
    {
        m_anchor_world = anchors;
    }

    void SlamKernel::clearAnchors()
    {
        m_anchor_slam.clear();
        resetAlignment();
    }

    int SlamKernel::getSlamAnchorCount() const
    {
        return static_cast<int>(m_anchor_slam.size());
    }

    int SlamKernel::getWorldAnchorCount() const
    {
        return static_cast<int>(m_anchor_world.size());
    }

    void SlamKernel::resetAlignment()
    {
        m_aligned = false;
        m_sim_scale = 1.0f;
        m_R = Eigen::Matrix3f::Identity();
        m_t = Eigen::Vector3f::Zero();
    }

    bool SlamKernel::computeAlignment()
    {
        const size_t n = m_anchor_slam.size();
        if (n < 3 || n != m_anchor_world.size())
        {
            resetAlignment();
            return false;
        }

        // Umeyama: 求 world = s * R * slam + t 的相似变换
        Eigen::Vector3d cs = Eigen::Vector3d::Zero();
        Eigen::Vector3d cw = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < n; ++i)
        {
            cs += m_anchor_slam[i].cast<double>();
            cw += m_anchor_world[i].cast<double>();
        }
        cs /= static_cast<double>(n);
        cw /= static_cast<double>(n);

        // 检查锚点在水平面(x-z)内是否近似共线：共线时无法唯一确定朝向
        {
            Eigen::Matrix2d cov2 = Eigen::Matrix2d::Zero();
            for (size_t i = 0; i < n; ++i)
            {
                const Eigen::Vector3d a = m_anchor_slam[i].cast<double>() - cs;
                cov2(0, 0) += a.x() * a.x();
                cov2(0, 1) += a.x() * a.z();
                cov2(1, 0) += a.x() * a.z();
                cov2(1, 1) += a.z() * a.z();
            }
            Eigen::JacobiSVD<Eigen::Matrix2d> cov_svd(cov2);
            const Eigen::Vector2d sv = cov_svd.singularValues();
            if (sv(1) < 5e-2 * sv(0))
            {
                DEBUG_INFO("[Android Slam App Info] alignment failed: anchors nearly collinear.");
                resetAlignment();
                return false;
            }
        }

        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < n; ++i)
        {
            const Eigen::Vector3d a = m_anchor_slam[i].cast<double>() - cs;
            const Eigen::Vector3d b = m_anchor_world[i].cast<double>() - cw;
            H += a * b.transpose();
        }

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Matrix3d Rd = V * U.transpose();
        if (Rd.determinant() < 0.0)
        {
            V.col(2) *= -1.0;
            Rd = V * U.transpose();
        }

        double num = 0.0;
        double den = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const Eigen::Vector3d a = m_anchor_slam[i].cast<double>() - cs;
            const Eigen::Vector3d b = m_anchor_world[i].cast<double>() - cw;
            num += b.dot(Rd * a);
            den += a.squaredNorm();
        }
        const double s = (den > 1e-12) ? (num / den) : 1.0;

        const Eigen::Vector3d td = cw - s * Rd * cs;

        m_R = Rd.cast<float>();
        m_t = td.cast<float>();
        m_sim_scale = static_cast<float>(s);
        m_aligned = true;
        return true;
    }

}
