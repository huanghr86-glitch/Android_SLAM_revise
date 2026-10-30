#include "SlamKernel.h"
#include "utils/RecoveryPointPool.h"
#include "utils/RecoveryValidation.h"
#include <iostream>
#include <set>
#include <unordered_set>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include "utils/CornerDiagnostics.h"

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
        m_debug_file = std::fopen("/storage/emulated/0/Documents/SlamDebug.txt", "w");
        for (const char* path : {"/storage/emulated/0/Documents/SlamMapping.txt",
                                "/storage/emulated/0/Documents/SlamKeyframes.txt",
                                "/storage/emulated/0/Documents/SlamReloc.txt"})
            if (FILE* file = std::fopen(path, "w")) std::fclose(file);
        ORB_SLAM3::SetCornerFrameTime(-1);
        ORB_SLAM3::ResetCornerPointTotals();
        if (m_debug_file) std::setvbuf(m_debug_file, nullptr, _IOFBF, 16384);
        const auto session = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        m_debug_frames_dir = "/storage/emulated/0/Documents/SlamDebugFrames/" + std::to_string(session);
        std::string recovery_profile;
        std::ifstream recovery_profile_file("/storage/emulated/0/Documents/SlamRecoveryProfile.txt");
        recovery_profile_file >> recovery_profile;
        const bool profile_valid = recovery_profile.empty() || recovery_profile == "normalized_only" || recovery_profile == "normalized_short";
        const bool short_enabled = recovery_profile != "normalized_only";
        ORB_SLAM3::SetCornerShortRecoveryEnabled(short_enabled);
        std::ostringstream config;
        config << "SLAM_CONFIG version=corner_diag_v6 session=" << session
               << " sensor=MONOCULAR imu_fused=0 unique_images_only=1 visual_reloc_timeout_s=1.5"
               << " recovery_policy=expanded_projection_pool reloc_score_policy=current_query_only recent_point_window_s=2"
               << " pool_max_points=" << ORB_SLAM3::kRecoveryPoolMaxPoints
               << " pool_recent_kfs=" << ORB_SLAM3::kRecoveryPoolRecentKeyFrames
               << " pool_covis_kfs=" << ORB_SLAM3::kRecoveryPoolCovisKeyFrames
               << " pool_recent_s=" << ORB_SLAM3::kRecoveryPoolRecentSeconds
               << " pool_max_expansions=" << ORB_SLAM3::kRecoveryPoolMaxExpansions
               << " pool_candidate_limit=" << ORB_SLAM3::kRecoveryPoolCandidateLimit
               << " pnp_policy=bounded_refine_applied pnp_sample_size=6 pnp_max_samples=300"
               << " tri_match_policy=keyframe_points_below100"
               << " fallback_policy=one_wide_local_retry fallback_min_unique=" << ORB_SLAM3::kFallbackMinimumUnique
               << " fallback_min_cells=" << ORB_SLAM3::kFallbackMinimumCells
               << " fallback_max_gap_s=" << ORB_SLAM3::kFallbackMaxFrameGap
               << " fallback_max_streak=" << ORB_SLAM3::kFallbackMaxStreak
               << " quality_policy=octave_chi2 quality_chi2=" << ORB_SLAM3::kRecoveryChi2
               << " recovery_profile=" << (short_enabled ? "normalized_short" : "normalized_only")
               << " profile_valid=" << (profile_valid ? 1 : 0)
               << " short_window_s=" << ORB_SLAM3::kShortRecoveryWindow
               << " short_confirm_frames=2 short_shared_min=15 short_shared_fraction=0.5"
               << " short_opt_budget=" << ORB_SLAM3::kShortOptimizationBudget
               << " short_anchor_rotation_deg=" << ORB_SLAM3::kShortAnchorRotation
               << " short_anchor_translation_depth=" << ORB_SLAM3::kShortAnchorTranslationDepth
               << " residual_log_limit=" << ORB_SLAM3::kRecoveryPointLogLimit
               << " fallback_max_rotation_deg=" << ORB_SLAM3::kFallbackMaxRotation
               << " fallback_max_translation_depth=" << ORB_SLAM3::kFallbackMaxTranslationDepth
               << " width=" << img_width << " height=" << img_height
               << " nominal_fps=" << desc.imageInfo.fps << " features=" << desc.orbInfo.nFeatures
               << " fx=" << desc.cameraInfo.fx << " fy=" << desc.cameraInfo.fy
               << " cx=" << desc.cameraInfo.cx << " cy=" << desc.cameraInfo.cy
               << " k1=" << desc.distortion->k1 << " k2=" << desc.distortion->k2
               << " p1=" << desc.distortion->p1 << " p2=" << desc.distortion->p2
               << " frames_dir=" << m_debug_frames_dir;
        writeDebugLine(config.str(), true);
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
        ORB_SLAM3::WriteCornerPointTotals(-1, ORB_SLAM3::CornerFrameTime());
        if (m_debug_image_job.valid()) m_debug_image_job.wait();
        if (m_debug_file) std::fclose(m_debug_file);
    }

    void SlamKernel::writeDebugLine(const std::string& line, bool flush)
    {
        __android_log_print(ANDROID_LOG_INFO, "SLAM_DIAG", "%s", line.c_str());
        if (m_debug_file)
        {
            std::fprintf(m_debug_file, "%s\n", line.c_str());
            if (flush) std::fflush(m_debug_file);
        }
        else
            __android_log_print(ANDROID_LOG_ERROR, "SLAM_DIAG", "SLAM_IO debug_file_unavailable=1");
    }

    void SlamKernel::reset()
    {
        m_orb_slam->Reset();
    }

    TrackingResult SlamKernel::handleData(const std::vector<Image>& images, const std::vector<ImuPoint>& imus)
    {
        const auto kernel_start = std::chrono::steady_clock::now();
        TrackingResult res;

        if (images.empty())
        {
            res.tracking_status = "NO_IMAGES_YET";
            return res;
        }

        const Image& image = images[0];
        res.time_stamp = image.time_stamp;
        if (image.time_stamp < m_begin_time_stamp || image.time_stamp <= m_last_input_timestamp)
        {
            writeDebugLine("SLAM_INPUT_REJECT ts=" + std::to_string(image.time_stamp) +
                    " previous_ts=" + std::to_string(m_last_input_timestamp), true);
            return res;
        }
        const double frame_dt_ms = m_last_input_timestamp < 0 ? -1.0 :
                (image.time_stamp - m_last_input_timestamp) * 1e-6;
        m_last_input_timestamp = image.time_stamp;

        cv::Mat cv_image(m_height, m_width, CV_8UC3);
        memcpy(cv_image.data, image.data.data(), sizeof(uint8_t) * image.data.size());


        auto& tracker = m_orb_slam->getTracker();
        auto* previous_map = m_orb_slam->getAtlas().GetCurrentMap();
        const long map_before = previous_map ? static_cast<long>(previous_map->GetId()) : -1;
        const int state_before = tracker.mState;
        const bool previous_valid = state_before == ORB_SLAM3::Tracking::OK && tracker.mCurrentFrame.isSet();
        const Sophus::SE3f previous_pose = previous_valid ? tracker.mCurrentFrame.GetPose() : Sophus::SE3f();
        double image_time_stamp = (double) (image.time_stamp - m_begin_time_stamp) * k_nano_sec_to_sec_radio;
        const auto track_start = std::chrono::steady_clock::now();
        Sophus::SE3f pose = m_orb_slam->TrackMonocular(cv_image, image_time_stamp);
        const double track_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - track_start).count();
        const auto diag = tracker.GetFrameDiagnostics();
        const int state_after = m_orb_slam->getTrackingState();
        res.pose_valid = state_after == ORB_SLAM3::Tracking::OK && tracker.mCurrentFrame.isSet();


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

        const double kernel_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - kernel_start).count();
        const auto diagnostic_start = std::chrono::steady_clock::now();
        cv::Mat gray, small_gray, laplacian;
        cv::cvtColor(cv_image, gray, cv::COLOR_RGB2GRAY);
        cv::resize(gray, small_gray, cv::Size(320, 240));
        cv::Scalar mean, deviation;
        cv::meanStdDev(small_gray, mean, deviation);
        cv::Laplacian(small_gray, laplacian, CV_32F);
        cv::Scalar lap_mean, lap_std;
        cv::meanStdDev(laplacian, lap_mean, lap_std);
        const double dark_ratio = cv::countNonZero(small_gray < 10) / static_cast<double>(small_gray.total());
        const double bright_ratio = cv::countNonZero(small_gray > 245) / static_cast<double>(small_gray.total());
        const long map_after = active_map ? static_cast<long>(active_map->GetId()) : -1;
        double rotation_deg = -1.0, translation_slam = -1.0;
        if (previous_valid && res.pose_valid && map_before == map_after)
        {
            const Sophus::SE3f relative_pose = pose * previous_pose.inverse();
            rotation_deg = relative_pose.so3().log().norm() * (180.0 / 3.141592653589793);
            translation_slam = (pose.inverse().translation() - previous_pose.inverse().translation()).norm();
        }

        // Save at most five loss episodes. JPEG encoding and disk I/O run off
        // the tracking thread; never queue an unbounded stream of images.
        int snapshot = 0;
        int weak_snapshot = 0;
        const bool capture_loss = state_before == ORB_SLAM3::Tracking::OK && !res.pose_valid && m_debug_loss_snapshots < 5;
        const int weak_level = res.pose_valid && diag.local_inliers >= 0 ?
                (diag.local_inliers <= 50 ? 2 : (diag.local_inliers <= 80 ? 1 : 0)) : 0;
        if ((capture_loss || weak_level > m_debug_weak_stage) &&
                (!m_debug_image_job.valid() || m_debug_image_job.wait_for(std::chrono::seconds(0)) == std::future_status::ready))
        {
            if (capture_loss) snapshot = ++m_debug_loss_snapshots;
            else { weak_snapshot = weak_level; m_debug_weak_stage = weak_level; }
            cv::Mat before = m_debug_previous_image;
            cv::Mat current = cv_image.clone();
            const auto keys = tracker.mCurrentFrame.mvKeys;
            const auto labels = diag.map_match_labels;
            const std::string prefix = m_debug_frames_dir + (snapshot ? "/loss_" + std::to_string(snapshot) :
                    "/weak_" + std::to_string(weak_snapshot));
            const std::string current_suffix = snapshot ? "_lost_" : "_current_";
            const auto before_ts = m_debug_previous_image_ts;
            const auto current_ts = image.time_stamp;
            m_debug_image_job = std::async(std::launch::async,
                    [before, current, keys, labels, prefix, current_suffix, before_ts, current_ts, dir = m_debug_frames_dir]() {
                mkdir("/storage/emulated/0/Documents/SlamDebugFrames", 0770);
                mkdir(dir.c_str(), 0770);
                try
                {
                    // The SLAM input buffer is RGB; imwrite expects BGR.
                    cv::Mat bgr, overlay;
                    cv::cvtColor(current, bgr, cv::COLOR_RGB2BGR);
                    bool ok = cv::imwrite(prefix + current_suffix + std::to_string(current_ts) + ".jpg", bgr);
                    cv::drawKeypoints(bgr, keys, overlay, cv::Scalar(0, 255, 0));
                    ok = cv::imwrite(prefix + "_keypoints.jpg", overlay) && ok;
                    overlay = bgr.clone();
                    for (size_t i = 0; i < keys.size(); ++i)
                    {
                        const int label = i < labels.size() ? labels[i] : 0;
                        const cv::Scalar color = label == 1 ? cv::Scalar(0, 255, 0) :
                                (label == 2 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255));
                        cv::circle(overlay, keys[i].pt, 2, color, 1);
                    }
                    ok = cv::imwrite(prefix + "_map_matches.jpg", overlay) && ok;
                    if (!before.empty())
                    {
                        cv::cvtColor(before, bgr, cv::COLOR_RGB2BGR);
                        ok = cv::imwrite(prefix + "_before_" + std::to_string(before_ts) + ".jpg", bgr) && ok;
                    }
                    __android_log_print(ok ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, "SLAM_DIAG",
                            "SLAM_SNAPSHOT prefix=%s ok=%d", prefix.c_str(), ok ? 1 : 0);
                }
                catch (const cv::Exception& error)
                {
                    __android_log_print(ANDROID_LOG_ERROR, "SLAM_DIAG", "SLAM_SNAPSHOT error=%s", error.what());
                }
            });
        }
        if (res.pose_valid)
        {
            m_debug_previous_image = cv_image.clone();
            m_debug_previous_image_ts = image.time_stamp;
        }
        const double diagnostic_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - diagnostic_start).count();
        std::ostringstream debug;
        debug << std::fixed << std::setprecision(3)
              << "SLAM_FRAME seq=" << m_debug_frame_index++ << " ts=" << image.time_stamp
              << " slam_t=" << std::setprecision(9) << image_time_stamp << std::setprecision(3)
              << " orb_frame=" << diag.frame_id << " state_before=" << state_before
              << " fallback_attempted=" << diag.fallback_attempted << " fallback_accepted=" << diag.fallback_accepted
              << " fallback_original_inliers=" << diag.fallback_original_inliers
              << " short_pending=" << diag.short_pending << " short_confirmed=" << diag.short_confirmed
              << " short_final_ok=" << diag.short_final_ok
              << " state=" << res.tracking_status << " valid=" << res.pose_valid
              << " map_before=" << map_before << " map=" << map_after
              << " kfs=" << (active_map ? active_map->KeyFramesInMap() : 0)
              << " map_points=" << res.map_points.size() << " keypoints=" << diag.keypoints
              << " keypoint_cells=" << diag.keypoint_cells << " inlier_cells=" << diag.inlier_cells
              << " frame_dt_ms=" << frame_dt_ms << " readback_ms=" << image.readback_ms
              << " track_ms=" << track_ms << " kernel_ms=" << kernel_ms << " diagnostic_ms=" << diagnostic_ms
              << " duplicate_polls=" << image.duplicate_polls << " backward_images=" << image.backward_images
              << " motion_raw=" << diag.motion_raw << " motion_wide=" << diag.motion_wide
              << " motion_inliers=" << diag.motion_inliers << " ref_raw=" << diag.ref_raw
              << " ref_inliers=" << diag.ref_inliers << " local_points=" << diag.local_points
              << " local_visible=" << diag.local_visible << " local_added=" << diag.local_added
              << " local_matches=" << diag.local_matches << " local_outliers=" << diag.local_outliers
              << " local_inliers=" << diag.local_inliers << " local_ok=" << diag.local_ok
              << " proj_tested=" << diag.projection_tested
              << " proj_visible=" << diag.projection_rejected[0] << " proj_depth=" << diag.projection_rejected[1]
              << " proj_bounds=" << diag.projection_rejected[2] << " proj_distance=" << diag.projection_rejected[3]
              << " proj_angle=" << diag.projection_rejected[4] << " proj_other=" << diag.projection_rejected[5]
              << " recent_points=" << diag.recent_points << " recent_preassociated=" << diag.recent_preassociated
              << " recent_tested=" << diag.recent_tested << " recent_visible=" << diag.recent_visible
              << " recent_proj_depth=" << diag.recent_projection[1] << " recent_proj_bounds=" << diag.recent_projection[2]
              << " recent_proj_distance=" << diag.recent_projection[3] << " recent_proj_angle=" << diag.recent_projection[4]
              << " recent_proj_other=" << diag.recent_projection[5]
              << " recent_inliers=" << diag.recent_inliers
              << " search_attempted=" << diag.projection_match.attempted
              << " search_no_features=" << diag.projection_match.no_features
              << " search_no_free=" << diag.projection_match.no_free_features
              << " search_descriptor=" << diag.projection_match.descriptor_rejected
              << " search_ratio=" << diag.projection_match.ratio_rejected << " search_matched=" << diag.projection_match.matched
              << " ref_kf=" << diag.ref_kf << " last_kf=" << diag.last_kf
              << " ref_in_db=" << diag.ref_in_db << " last_in_db=" << diag.last_in_db
              << " db_map_kfs=" << diag.db_map_kfs << " db_newest_kf=" << diag.db_newest_kf
              << " db_newest_age_s=" << diag.db_newest_age_s
              << " reloc_candidates=" << diag.reloc_candidates << " reloc_bow_best=" << diag.reloc_bow_best
              << " reloc_good_best=" << diag.reloc_good_best << " new_kf=" << diag.new_keyframe
              << " lost_s=" << diag.lost_s << " event=" << diag.event
              << " rotation_deg=" << rotation_deg << " translation_slam=" << std::setprecision(9) << translation_slam << std::setprecision(3)
              << " gray_mean=" << mean[0] << " gray_std=" << deviation[0]
              << " lap_var=" << lap_std[0] * lap_std[0] << " dark_ratio=" << dark_ratio
              << " bright_ratio=" << bright_ratio << " snapshot=" << snapshot << " weak_snapshot=" << weak_snapshot;
        writeDebugLine(debug.str(), state_after != ORB_SLAM3::Tracking::OK ||
                state_before != state_after || map_before != map_after || m_debug_frame_index % 15 == 0);

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
