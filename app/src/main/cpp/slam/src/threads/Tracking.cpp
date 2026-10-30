/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "threads/Tracking.h"
#include <iostream>
#include <mutex>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <memory>
#include <unordered_set>

#include "core/System.h"

#include "threads/LocalMapping.h"
#include "threads/LoopClosing.h"

#include "feature/ORBmatcher.h"

#include "utils/Converter.h"
#include "utils/GeometricTools.h"
#include "utils/RecoveryPointPool.h"
#include "utils/CornerGeometryDiagnostics.h"
#include "utils/RecoveryValidation.h"

#include "solver/G2oTypes.h"
#include "solver/Optimizer.h"
#include "solver/MLPnPsolver.h"

#include "camera_models/Pinhole.h"
#include "camera_models/KannalaBrandt8.h"

using namespace std;

namespace ORB_SLAM3
{
    namespace
    {
        struct RecoveryPoolSnapshot
        {
            bool built = false;
            vector<RecoveryProjectionPoint> points;
            unordered_set<unsigned long> extra_ids;
            size_t seed_points = 0;
        };

        RecoveryPoolSnapshot BuildRecoveryPool(KeyFrame* seed, Map* active_map,
                const vector<KeyFrame*>& recent_keyframes, unsigned long frame_id, double timestamp,
                vector<string>* log_lines)
        {
            const auto began = std::chrono::steady_clock::now();
            RecoveryPoolSnapshot result;
            result.built = true;
            vector<pair<KeyFrame*, int>> sources;
            set<unsigned long> source_ids;
            int recent_added = 0, covis_added = 0;
            auto add_source = [&](KeyFrame* keyframe, int kind) {
                if (!keyframe || keyframe->isBad() || keyframe->GetMap() != active_map ||
                    !source_ids.insert(keyframe->mnId).second) return false;
                sources.emplace_back(keyframe, kind);
                return true;
            };
            add_source(seed, 0);
            for (KeyFrame* keyframe : recent_keyframes)
            {
                if (recent_added >= kRecoveryPoolRecentKeyFrames) break;
                if (add_source(keyframe, 1)) ++recent_added;
            }
            if (seed && !seed->isBad())
                for (KeyFrame* keyframe : seed->GetBestCovisibilityKeyFrames(kRecoveryPoolCovisKeyFrames))
                    if (add_source(keyframe, 2)) ++covis_added;

            RecoveryPointPoolBuilder<RecoveryProjectionPoint> builder(kRecoveryPoolMaxPoints);
            size_t slots = 0, null_points = 0, bad_points = 0, wrong_map = 0, no_observations = 0, bad_index = 0;
            for (const auto& source : sources)
            {
                const auto points = source.first->GetMapPointMatches();
                const size_t before_size = builder.size(), before_offered = builder.offered;
                const size_t before_duplicates = builder.duplicates, before_dropped = builder.dropped;
                slots += points.size();
                for (size_t index = 0; index < points.size(); ++index)
                {
                    MapPoint* point = points[index];
                    if (!point) { ++null_points; continue; }
                    if (point->isBad()) { ++bad_points; continue; }
                    if (point->GetMap() != active_map) { ++wrong_map; continue; }
                    if (point->Observations() < 1) { ++no_observations; continue; }
                    if (index >= source.first->mvKeysUn.size()) { ++bad_index; continue; }
                    if (builder.Add(point->mnId, {point, source.first, index, source.second}) && source.second == 0)
                        ++result.seed_points;
                }
                if (log_lines)
                    AppendCornerEvent(*log_lines, "SLAM_RELOC_POOL_SOURCE frame=%lu ts=%.9f seed=%lu source=%lu kind=%d age_s=%.6f slots=%zu offered=%zu added=%zu duplicates=%zu dropped=%zu",
                            frame_id, timestamp, seed->mnId, source.first->mnId, source.second,
                            timestamp - source.first->mTimeStamp, points.size(), builder.offered - before_offered,
                            builder.size() - before_size, builder.duplicates - before_duplicates, builder.dropped - before_dropped);
            }
            const size_t point_count = builder.size();
            result.points = builder.Take();
            for (const auto& point : result.points)
                if (point.source_kind != 0) result.extra_ids.insert(point.point->mnId);
            if (log_lines)
                AppendCornerEvent(*log_lines, "SLAM_RELOC_POOL frame=%lu ts=%.9f seed=%lu map=%lu sources=%zu recent_sources=%d covis_sources=%d slots=%zu null=%zu bad=%zu wrong_map=%zu no_obs=%zu bad_index=%zu offered=%zu duplicates=%zu dropped=%zu points=%zu seed_points=%zu extra_points=%zu capacity=%zu max_expansions=%d build_ms=%.3f",
                        frame_id, timestamp, seed->mnId, active_map->GetId(), sources.size(), recent_added, covis_added,
                        slots, null_points, bad_points, wrong_map, no_observations, bad_index, builder.offered,
                        builder.duplicates, builder.dropped, point_count, result.seed_points,
                        point_count - result.seed_points, kRecoveryPoolMaxPoints, kRecoveryPoolMaxExpansions,
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count());
            return result;
        }

        int CountRecoveryExtraInliers(const Frame& frame, const RecoveryPoolSnapshot& pool)
        {
            int count = 0;
            for (size_t i = 0; i < frame.mvpMapPoints.size(); ++i)
                if (frame.mvpMapPoints[i] && !frame.mvbOutlier[i] &&
                    !frame.mvpMapPoints[i]->isBad() && frame.mvpMapPoints[i]->Observations() > 0 &&
                    pool.extra_ids.count(frame.mvpMapPoints[i]->mnId)) ++count;
            return count;
        }

        struct RecoveryResidual { unsigned long point; size_t feature; int octave; double raw, chi2, inverse_variance; };

        RecoveryQuality MeasureRecoveryQuality(const Frame& frame, Map* active_map,
                const Sophus::SE3f* previous, const RecoveryPoolSnapshot* pool = nullptr,
                vector<RecoveryResidual>* residuals = nullptr)
        {
            RecoveryQuality q;
            const auto pose = frame.GetPose();
            vector<double> errors, depths, normalized;
            unordered_set<unsigned long> unique;
            bool cells[48] = {};
            double squared = 0, normalized_squared = 0;
            for (size_t i = 0; i < frame.mvpMapPoints.size() && i < frame.mvKeysUn.size(); ++i)
            {
                MapPoint* point = frame.mvpMapPoints[i];
                if (!point || frame.mvbOutlier[i]) continue;
                ++q.associations;
                if (point->isBad() || point->Observations() < 1 || point->GetMap() != active_map)
                { ++q.invalid; continue; }
                if (!unique.insert(point->mnId).second) { ++q.duplicates; continue; }
                const Eigen::Vector3f camera = pose * point->GetWorldPos();
                if (!RecoveryFinite(camera.x()) || !RecoveryFinite(camera.y()) ||
                    !RecoveryFinite(camera.z()) || camera.z() <= 1e-6)
                { ++q.invalid; continue; }
                const auto projected = frame.mpCamera->project(camera);
                const auto observed = frame.mvKeysUn[i].pt;
                const double error = std::hypot(double(projected.x()) - observed.x, double(projected.y()) - observed.y);
                if (!RecoveryFinite(error)) { ++q.invalid; continue; }
                const int octave = frame.mvKeysUn[i].octave;
                if (octave < 0 || octave >= static_cast<int>(frame.mvInvLevelSigma2.size()))
                { ++q.invalid; continue; }
                const double inverse_variance = frame.mvInvLevelSigma2[octave];
                const double chi2 = RecoveryChi2(error, inverse_variance);
                if (!RecoveryFinite(chi2) || chi2 < 0) { ++q.invalid; continue; }
                normalized.push_back(std::sqrt(chi2));
                normalized_squared += chi2;
                q.max_chi2 = std::max(q.max_chi2, chi2);
                if (chi2 > kRecoveryChi2 + .0001) ++q.chi2_over;
                if (residuals && residuals->size() < kRecoveryPointLogLimit)
                    residuals->push_back({point->mnId, i, octave, error, chi2, inverse_variance});
                errors.push_back(error);
                depths.push_back(camera.z());
                squared += error * error;
                ++q.unique;
                if (pool && pool->extra_ids.count(point->mnId)) ++q.extra;
                const int x = std::max(0, std::min(7, static_cast<int>((observed.x-frame.mnMinX)*8/(frame.mnMaxX-frame.mnMinX))));
                const int y = std::max(0, std::min(5, static_cast<int>((observed.y-frame.mnMinY)*6/(frame.mnMaxY-frame.mnMinY))));
                cells[y*8+x] = true;
            }
            for (bool cell : cells) if (cell) ++q.cells;
            if (!errors.empty())
            {
                sort(errors.begin(), errors.end());
                sort(depths.begin(), depths.end());
                sort(normalized.begin(), normalized.end());
                q.rmse = std::sqrt(squared/errors.size());
                q.median = errors[errors.size()/2];
                q.p95 = errors[std::min(errors.size()-1, errors.size()*95/100)];
                q.normalized_rmse = std::sqrt(normalized_squared/errors.size());
                q.normalized_p95 = normalized[std::min(normalized.size()-1, normalized.size()*95/100)];
                q.median_depth = depths[depths.size()/2];
                if (previous)
                {
                    const auto delta = pose * previous->inverse();
                    q.rotation_deg = delta.so3().log().norm()*180.0/M_PI;
                    q.translation_depth = delta.translation().norm()/q.median_depth;
                }
            }
            return q;
        }

        void AppendRecoveryResiduals(vector<string>& lines, const char* source, const Frame& frame,
                int total, const vector<RecoveryResidual>& residuals)
        {
            AppendCornerEvent(lines, "SLAM_QUALITY_POINTS frame=%lu ts=%.9f source=%s total=%d emitted=%zu limit=%d",
                    frame.mnId, frame.mTimeStamp, source, total, residuals.size(), kRecoveryPointLogLimit);
            for (const auto& p : residuals)
                AppendCornerEvent(lines, "SLAM_QUALITY_POINT frame=%lu ts=%.9f source=%s point=%lu feature=%zu octave=%d error_px=%.9f inv_sigma2=%.9f chi2=%.9f",
                        frame.mnId, frame.mTimeStamp, source, p.point, p.feature, p.octave, p.raw, p.inverse_variance, p.chi2);
        }

        vector<KeyFrame*> RecentRecoveryKeyFrames(Map* map, double timestamp)
        {
            vector<KeyFrame*> frames;
            for (KeyFrame* keyframe : map->GetAllKeyFrames())
                if (!keyframe->isBad() && keyframe->GetMap() == map && timestamp >= keyframe->mTimeStamp &&
                    timestamp-keyframe->mTimeStamp <= kRecoveryPoolRecentSeconds) frames.push_back(keyframe);
            sort(frames.begin(), frames.end(), [](KeyFrame* a, KeyFrame* b) {
                return a->mTimeStamp == b->mTimeStamp ? a->mnId > b->mnId : a->mTimeStamp > b->mTimeStamp;
            });
            return frames;
        }

        void DeduplicateRecoveryFrame(Frame& frame)
        {
            unordered_set<unsigned long> ids;
            for (auto& point : frame.mvpMapPoints)
                if (point && !ids.insert(point->mnId).second) point = nullptr;
        }

        void LogRecoveryVerification(Frame& frame, KeyFrame* seed, const RecoveryPoolSnapshot& pool,
                int optimized_inliers, vector<string>& lines)
        {
            const auto pose = frame.GetPose();
            vector<float> errors;
            double squared_error = 0;
            int invalid = 0, extra_inliers = 0;
            bool cells[48] = {};
            for (size_t i = 0; i < frame.mvpMapPoints.size() && i < frame.mvKeysUn.size(); ++i)
            {
                MapPoint* point = frame.mvpMapPoints[i];
                if (!point || frame.mvbOutlier[i] || point->isBad() || point->Observations() < 1) continue;
                const Eigen::Vector3f camera = pose * point->GetWorldPos();
                if (!CornerFinite(camera.x()) || !CornerFinite(camera.y()) || !CornerFinite(camera.z()) || camera.z() <= 0)
                { ++invalid; continue; }
                const auto pixel = frame.mpCamera->project(camera);
                const auto observed = frame.mvKeysUn[i].pt;
                const float error = std::hypot(pixel.x() - observed.x, pixel.y() - observed.y);
                if (!CornerFinite(error)) { ++invalid; continue; }
                errors.push_back(error);
                squared_error += error * error;
                if (pool.extra_ids.count(point->mnId)) ++extra_inliers;
                const int x = std::max(0, std::min(7, static_cast<int>((observed.x - frame.mnMinX) * 8 / (frame.mnMaxX - frame.mnMinX))));
                const int y = std::max(0, std::min(5, static_cast<int>((observed.y - frame.mnMinY) * 6 / (frame.mnMaxY - frame.mnMinY))));
                cells[y * 8 + x] = true;
            }
            std::sort(errors.begin(), errors.end());
            int occupied = 0;
            for (bool cell : cells) if (cell) ++occupied;
            AppendCornerEvent(lines, "SLAM_RELOC_VERIFY frame=%lu ts=%.9f seed=%lu optimized_inliers=%d checked=%zu extra_inliers=%d cells=%d invalid_projection=%d rmse_px=%.6f median_px=%.6f p95_px=%.6f stage=candidate_gate",
                    frame.mnId, frame.mTimeStamp, seed->mnId, optimized_inliers, errors.size(), extra_inliers,
                    occupied, invalid, errors.empty() ? -1 : std::sqrt(squared_error / errors.size()),
                    errors.empty() ? -1 : errors[errors.size() / 2],
                    errors.empty() ? -1 : errors[std::min(errors.size() - 1, errors.size() * 95 / 100)]);
        }
    }


    Tracking::Tracking(System* pSys,
                       ORBVocabulary* pVoc,
                       Atlas* pAtlas,
                       KeyFrameDatabase* pKFDB,
                       const string& strSettingPath,
                       const int sensor,
                       Settings* settings,
                       const string& _nameSeq)
        : mState(NO_IMAGES_YET)
        , mSensor(sensor)
        , mTrackedFr(0)
        , mbStep(false)
        , mbOnlyTracking(false)
        , mbMapUpdated(false)
        , mbVO(false)
        , mpORBVocabulary(pVoc)
        , mpKeyFrameDB(pKFDB)
        , mbReadyToInitializate(false)
        , mpSystem(pSys)
        , mpAtlas(pAtlas)
        , mnLastRelocFrameId(0)
        , time_recently_lost(5.0)
        , mnInitialFrameId(0)
        , mbCreatedMap(false)
        , mnFirstFrameId(0)
        , mpCamera2(nullptr)
        , mpLastKeyFrame(static_cast<KeyFrame*>(NULL))
    {
        // Load camera parameters from settings file
        if (settings) {
            newParameterLoader(settings);
        }
        else {
            cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

            bool b_parse_cam = ParseCamParamFile(fSettings);
            if (!b_parse_cam)
            {
                std::cout << "*Error with the camera parameters in the config file*" << std::endl;
            }

            // Load ORB parameters
            bool b_parse_orb = ParseORBParamFile(fSettings);
            if (!b_parse_orb)
            {
                std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
            }

            bool b_parse_imu = true;
            if (sensor == System::IMU_MONOCULAR || sensor == System::IMU_STEREO || sensor == System::IMU_RGBD)
            {
                b_parse_imu = ParseIMUParamFile(fSettings);
                if (!b_parse_imu)
                {
                    std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
                }

                mnFramesToResetIMU = mMaxFrames;
            }

            if (!b_parse_cam || !b_parse_orb || !b_parse_imu)
            {
                std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
                try
                {
                    throw - 1;
                }
                catch (exception& e)
                {

                }
            }
        }

        initID = 0; lastID = 0;
        mbInitWith3KFs = false;
        mnNumDataset = 0;

        vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
        std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
        for (GeometricCamera* pCam : vpCams)
        {
            std::cout << "Camera " << pCam->GetId();
            if (pCam->GetType() == GeometricCamera::CAM_PINHOLE)
            {
                std::cout << " is pinhole" << std::endl;
            }
            else if (pCam->GetType() == GeometricCamera::CAM_FISHEYE)
            {
                std::cout << " is fisheye" << std::endl;
            }
            else
            {
                std::cout << " is unknown" << std::endl;
            }
        }
    }

    Tracking::~Tracking()
    {
        //f_track_stats.close();

    }

    void Tracking::newParameterLoader(Settings* settings)
    {
        mpCamera = settings->camera1();
        mpCamera = mpAtlas->AddCamera(mpCamera);

        if (settings->needToUndistort()) {
            mDistCoef = settings->camera1DistortionCoef();
        }
        else {
            mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
        }

        //TODO: missing image scaling and rectification
        mImageScale = 1.0f;

        mK = cv::Mat::eye(3, 3, CV_32F);
        mK.at<float>(0, 0) = mpCamera->getParameter(0);
        mK.at<float>(1, 1) = mpCamera->getParameter(1);
        mK.at<float>(0, 2) = mpCamera->getParameter(2);
        mK.at<float>(1, 2) = mpCamera->getParameter(3);

        mK_.setIdentity();
        mK_(0, 0) = mpCamera->getParameter(0);
        mK_(1, 1) = mpCamera->getParameter(1);
        mK_(0, 2) = mpCamera->getParameter(2);
        mK_(1, 2) = mpCamera->getParameter(3);

        if ((mSensor == System::STEREO || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&
            settings->cameraType() == Settings::KannalaBrandt) {
            mpCamera2 = settings->camera2();
            mpCamera2 = mpAtlas->AddCamera(mpCamera2);

            mTlr = settings->Tlr();
        }

        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
            mbf = settings->bf();
            mThDepth = settings->b() * settings->thDepth();
        }

        if (mSensor == System::RGBD || mSensor == System::IMU_RGBD) {
            mDepthMapFactor = settings->depthMapFactor();
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }

        mMinFrames = 0;
        mMaxFrames = settings->fps();
        mbRGB = settings->rgb();

        //ORB parameters
        int nFeatures = settings->nFeatures();
        int nLevels = settings->nLevels();
        int fIniThFAST = settings->initThFAST();
        int fMinThFAST = settings->minThFAST();
        float fScaleFactor = settings->scaleFactor();

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
            mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        //IMU parameters
        Sophus::SE3f Tbc = settings->Tbc();
        mInsertKFsLost = settings->insertKFsWhenLost();
        mImuFreq = settings->imuFrequency();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
        float Ng = settings->noiseGyro();
        float Na = settings->noiseAcc();
        float Ngw = settings->gyroWalk();
        float Naw = settings->accWalk();

        const float sf = sqrt(mImuFreq);
        mpImuCalib = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
    }

    bool Tracking::ParseCamParamFile(cv::FileStorage& fSettings)
    {
        mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
        cout << endl << "Camera Parameters: " << endl;
        bool b_miss_params = false;

        string sCameraName = fSettings["Camera.type"];
        if (sCameraName == "PinHole")
        {
            float fx, fy, cx, cy;
            mImageScale = 1.f;

            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera.fx"];
            if (!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.fy"];
            if (!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.cx"];
            if (!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.cy"];
            if (!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera.k1"];
            if (!node.empty() && node.isReal())
            {
                mDistCoef.at<float>(0) = node.real();
            }
            else
            {
                std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.k2"];
            if (!node.empty() && node.isReal())
            {
                mDistCoef.at<float>(1) = node.real();
            }
            else
            {
                std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.p1"];
            if (!node.empty() && node.isReal())
            {
                mDistCoef.at<float>(2) = node.real();
            }
            else
            {
                std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.p2"];
            if (!node.empty() && node.isReal())
            {
                mDistCoef.at<float>(3) = node.real();
            }
            else
            {
                std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.k3"];
            if (!node.empty() && node.isReal())
            {
                mDistCoef.resize(5);
                mDistCoef.at<float>(4) = node.real();
            }

            node = fSettings["Camera.imageScale"];
            if (!node.empty() && node.isReal())
            {
                mImageScale = node.real();
            }

            if (b_miss_params)
            {
                return false;
            }

            if (mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{ fx,fy,cx,cy };

            mpCamera = new Pinhole(vCamCalib);

            mpCamera = mpAtlas->AddCamera(mpCamera);

            std::cout << "- Camera: Pinhole" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
            std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


            std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
            std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

            if (mDistCoef.rows == 5)
                std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

            mK = cv::Mat::eye(3, 3, CV_32F);
            mK.at<float>(0, 0) = fx;
            mK.at<float>(1, 1) = fy;
            mK.at<float>(0, 2) = cx;
            mK.at<float>(1, 2) = cy;

            mK_.setIdentity();
            mK_(0, 0) = fx;
            mK_(1, 1) = fy;
            mK_(0, 2) = cx;
            mK_(1, 2) = cy;
        }
        else if (sCameraName == "KannalaBrandt8")
        {
            float fx, fy, cx, cy;
            float k1, k2, k3, k4;
            mImageScale = 1.f;

            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera.fx"];
            if (!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera.fy"];
            if (!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.cx"];
            if (!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.cy"];
            if (!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera.k1"];
            if (!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera.k2"];
            if (!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.k3"];
            if (!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.k4"];
            if (!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera.imageScale"];
            if (!node.empty() && node.isReal())
            {
                mImageScale = node.real();
            }

            if (!b_miss_params)
            {
                if (mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;
                }

                vector<float> vCamCalib{ fx,fy,cx,cy,k1,k2,k3,k4 };
                mpCamera = new KannalaBrandt8(vCamCalib);
                mpCamera = mpAtlas->AddCamera(mpCamera);
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                mK = cv::Mat::eye(3, 3, CV_32F);
                mK.at<float>(0, 0) = fx;
                mK.at<float>(1, 1) = fy;
                mK.at<float>(0, 2) = cx;
                mK.at<float>(1, 2) = cy;

                mK_.setIdentity();
                mK_(0, 0) = fx;
                mK_(1, 1) = fy;
                mK_(0, 2) = cx;
                mK_(1, 2) = cy;
            }

            if (mSensor == System::STEREO || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) {
                // Right camera
                // Camera calibration parameters
                cv::FileNode node = fSettings["Camera2.fx"];
                if (!node.empty() && node.isReal())
                {
                    fx = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }
                node = fSettings["Camera2.fy"];
                if (!node.empty() && node.isReal())
                {
                    fy = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.cx"];
                if (!node.empty() && node.isReal())
                {
                    cx = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.cy"];
                if (!node.empty() && node.isReal())
                {
                    cy = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                // Distortion parameters
                node = fSettings["Camera2.k1"];
                if (!node.empty() && node.isReal())
                {
                    k1 = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }
                node = fSettings["Camera2.k2"];
                if (!node.empty() && node.isReal())
                {
                    k2 = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.k3"];
                if (!node.empty() && node.isReal())
                {
                    k3 = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.k4"];
                if (!node.empty() && node.isReal())
                {
                    k4 = node.real();
                }
                else
                {
                    std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }


                int leftLappingBegin = -1;
                int leftLappingEnd = -1;

                int rightLappingBegin = -1;
                int rightLappingEnd = -1;

                node = fSettings["Camera.lappingBegin"];
                if (!node.empty() && node.isInt())
                {
                    leftLappingBegin = node.operator int();
                }
                else
                {
                    std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
                }
                node = fSettings["Camera.lappingEnd"];
                if (!node.empty() && node.isInt())
                {
                    leftLappingEnd = node.operator int();
                }
                else
                {
                    std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
                }
                node = fSettings["Camera2.lappingBegin"];
                if (!node.empty() && node.isInt())
                {
                    rightLappingBegin = node.operator int();
                }
                else
                {
                    std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
                }
                node = fSettings["Camera2.lappingEnd"];
                if (!node.empty() && node.isInt())
                {
                    rightLappingEnd = node.operator int();
                }
                else
                {
                    std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
                }

                node = fSettings["Tlr"];
                cv::Mat cvTlr;
                if (!node.empty())
                {
                    cvTlr = node.mat();
                    if (cvTlr.rows != 3 || cvTlr.cols != 4)
                    {
                        std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                        b_miss_params = true;
                    }
                }
                else
                {
                    std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                    b_miss_params = true;
                }

                if (!b_miss_params)
                {
                    if (mImageScale != 1.f)
                    {
                        // K matrix parameters must be scaled.
                        fx = fx * mImageScale;
                        fy = fy * mImageScale;
                        cx = cx * mImageScale;
                        cy = cy * mImageScale;

                        leftLappingBegin = leftLappingBegin * mImageScale;
                        leftLappingEnd = leftLappingEnd * mImageScale;
                        rightLappingBegin = rightLappingBegin * mImageScale;
                        rightLappingEnd = rightLappingEnd * mImageScale;
                    }

                    static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                    static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                    vector<float> vCamCalib2{ fx,fy,cx,cy,k1,k2,k3,k4 };
                    mpCamera2 = new KannalaBrandt8(vCamCalib2);
                    mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                    mTlr = Converter::toSophus(cvTlr);

                    static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                    static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                    std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                    std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                    std::cout << "- Camera: Fisheye" << std::endl;
                    std::cout << "- Image scale: " << mImageScale << std::endl;
                    std::cout << "- fx: " << fx << std::endl;
                    std::cout << "- fy: " << fy << std::endl;
                    std::cout << "- cx: " << cx << std::endl;
                    std::cout << "- cy: " << cy << std::endl;
                    std::cout << "- k1: " << k1 << std::endl;
                    std::cout << "- k2: " << k2 << std::endl;
                    std::cout << "- k3: " << k3 << std::endl;
                    std::cout << "- k4: " << k4 << std::endl;

                    std::cout << "- mTlr: \n" << cvTlr << std::endl;

                    std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
                }
            }

            if (b_miss_params)
            {
                return false;
            }

        }
        else
        {
            std::cerr << "*Not Supported Camera Sensor*" << std::endl;
            std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
        }

        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            cv::FileNode node = fSettings["Camera.bf"];
            if (!node.empty() && node.isReal())
            {
                mbf = node.real();
                if (mImageScale != 1.f)
                {
                    mbf *= mImageScale;
                }
            }
            else
            {
                std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

        }

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            float fx = mpCamera->getParameter(0);
            cv::FileNode node = fSettings["ThDepth"];
            if (!node.empty() && node.isReal())
            {
                mThDepth = node.real();
                mThDepth = mbf * mThDepth / fx;
                cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
            }
            else
            {
                std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


        }

        if (mSensor == System::RGBD || mSensor == System::IMU_RGBD)
        {
            cv::FileNode node = fSettings["DepthMapFactor"];
            if (!node.empty() && node.isReal())
            {
                mDepthMapFactor = node.real();
                if (fabs(mDepthMapFactor) < 1e-5)
                    mDepthMapFactor = 1;
                else
                    mDepthMapFactor = 1.0f / mDepthMapFactor;
            }
            else
            {
                std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

        }

        if (b_miss_params)
        {
            return false;
        }

        return true;
    }

    bool Tracking::ParseORBParamFile(cv::FileStorage& fSettings)
    {
        bool b_miss_params = false;
        int nFeatures, nLevels, fIniThFAST, fMinThFAST;
        float fScaleFactor;

        cv::FileNode node = fSettings["ORBextractor.nFeatures"];
        if (!node.empty() && node.isInt())
        {
            nFeatures = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.scaleFactor"];
        if (!node.empty() && node.isReal())
        {
            fScaleFactor = node.real();
        }
        else
        {
            std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.nLevels"];
        if (!node.empty() && node.isInt())
        {
            nLevels = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.iniThFAST"];
        if (!node.empty() && node.isInt())
        {
            fIniThFAST = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.minThFAST"];
        if (!node.empty() && node.isInt())
        {
            fMinThFAST = node.operator int();
        }
        else
        {
            std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        if (b_miss_params)
        {
            return false;
        }

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
            mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        return true;
    }

    bool Tracking::ParseIMUParamFile(cv::FileStorage& fSettings)
    {
        bool b_miss_params = false;

        cv::Mat cvTbc;
        cv::FileNode node = fSettings["Tbc"];
        if (!node.empty())
        {
            cvTbc = node.mat();
            if (cvTbc.rows != 4 || cvTbc.cols != 4)
            {
                std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
                b_miss_params = true;
            }
        }
        else
        {
            std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
            b_miss_params = true;
        }
        cout << endl;
        cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
        Sophus::SE3f Tbc(eigTbc);

        node = fSettings["InsertKFsWhenLost"];
        mInsertKFsLost = true;
        if (!node.empty() && node.isInt())
        {
            mInsertKFsLost = (bool)node.operator int();
        }

        if (!mInsertKFsLost)
            cout << "Do not insert keyframes when lost visual tracking " << endl;



        float Ng, Na, Ngw, Naw;

        node = fSettings["IMU.Frequency"];
        if (!node.empty() && node.isInt())
        {
            mImuFreq = node.operator int();
            mImuPer = 0.001; //1.0 / (double) mImuFreq;
        }
        else
        {
            std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.NoiseGyro"];
        if (!node.empty() && node.isReal())
        {
            Ng = node.real();
        }
        else
        {
            std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.NoiseAcc"];
        if (!node.empty() && node.isReal())
        {
            Na = node.real();
        }
        else
        {
            std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.GyroWalk"];
        if (!node.empty() && node.isReal())
        {
            Ngw = node.real();
        }
        else
        {
            std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.AccWalk"];
        if (!node.empty() && node.isReal())
        {
            Naw = node.real();
        }
        else
        {
            std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.fastInit"];
        mFastInit = false;
        if (!node.empty())
        {
            mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
        }

        if (mFastInit)
            cout << "Fast IMU initialization. Acceleration is not checked \n";

        if (b_miss_params)
        {
            return false;
        }

        const float sf = sqrt(mImuFreq);
        cout << endl;
        cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
        cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
        cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
        cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
        cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

        mpImuCalib = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);


        return true;
    }

    void Tracking::SetLocalMapper(LocalMapping* pLocalMapper)
    {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing* pLoopClosing)
    {
        mpLoopClosing = pLoopClosing;
    }



    Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat& imRectLeft, const cv::Mat& imRectRight, const double& timestamp, string filename)
    {
        //cout << "GrabImageStereo" << endl;

        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;
        mImRight = imRectRight;

        if (mImGray.channels() == 3)
        {
            //cout << "Image with 3 channels" << endl;
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGB2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGR2GRAY);
            }
        }
        else if (mImGray.channels() == 4)
        {
            //cout << "Image with 4 channels" << endl;
            if (mbRGB)
            {
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGBA2GRAY);
            }
            else
            {
                cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
                cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGRA2GRAY);
            }
        }

        //cout << "Incoming frame creation" << endl;

        if (mSensor == System::STEREO && !mpCamera2)
        {
            //mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
            mCurrentFrame.reset(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
        }
        else if (mSensor == System::STEREO && mpCamera2)
        {
            //mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr);
            mCurrentFrame.reset(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr);
        }
        else if (mSensor == System::IMU_STEREO && !mpCamera2)
        {
            //mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);
            mCurrentFrame.reset(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);
        }
        else if (mSensor == System::IMU_STEREO && mpCamera2)
        {
            //mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr, &mLastFrame, *mpImuCalib);
            mCurrentFrame.reset(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, mpCamera2, mTlr, &mLastFrame, *mpImuCalib);
        }

        //cout << "Incoming frame ended" << endl;

        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;

        //cout << "Tracking start" << endl;
        Track();
        //cout << "Tracking end" << endl;

        return mCurrentFrame.GetPose();
    }


    Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat& imRGB, const cv::Mat& imD, const double& timestamp, string filename)
    {
        mImGray = imRGB;
        cv::Mat imDepth = imD;

        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
        }

        if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
            imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

        if (mSensor == System::RGBD)
        {
            //mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
            mCurrentFrame.reset(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera);
        }
        else if (mSensor == System::IMU_RGBD)
        {
            //mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);
            mCurrentFrame.reset(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, mpCamera, &mLastFrame, *mpImuCalib);
        }


        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;

        Track();

        return mCurrentFrame.GetPose();
    }


    Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat& im, const double& timestamp, string filename)
    {
        mFrameDiagnostics = FrameDiagnostics{};
        SetCornerFrameTime(timestamp);
        mImGray = im;
        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
        }

        if (mSensor == System::MONOCULAR)
        {
            if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET || (lastID - initID) < mMaxFrames)
            {
                //mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
                mCurrentFrame.reset(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
            }
            else
            {
                //mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
                mCurrentFrame.reset(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth);
            }
        }
        else if (mSensor == System::IMU_MONOCULAR)
        {
            if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
            {
                //mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
                mCurrentFrame.reset(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
            }
            else
            {
                //mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
                mCurrentFrame.reset(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf, mThDepth, &mLastFrame, *mpImuCalib);
            }
        }

        if (mState == NO_IMAGES_YET)
            t0 = timestamp;

        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;

        lastID = mCurrentFrame.mnId;
        mFrameDiagnostics.frame_id = mCurrentFrame.mnId;
        mFrameDiagnostics.keypoints = mCurrentFrame.N;
        bool occupied[48] = {};
        for (const auto& key : mCurrentFrame.mvKeys)
        {
            const int x = std::max(0, std::min(7, static_cast<int>(key.pt.x * 8 / mImGray.cols)));
            const int y = std::max(0, std::min(5, static_cast<int>(key.pt.y * 6 / mImGray.rows)));
            occupied[y * 8 + x] = true;
        }
        for (bool cell : occupied) if (cell) ++mFrameDiagnostics.keypoint_cells;
        Track();

        Map* debug_map = mpAtlas->GetCurrentMap();
        if (debug_map && (mState == OK || mState == RECENTLY_LOST))
        {
            const auto index = mpKeyFrameDB->GetDebugIndexStats(debug_map->GetId(), mpReferenceKF, mpLastKeyFrame);
            mFrameDiagnostics.ref_kf = mpReferenceKF ? static_cast<long>(mpReferenceKF->mnId) : -1;
            mFrameDiagnostics.last_kf = mpLastKeyFrame ? static_cast<long>(mpLastKeyFrame->mnId) : -1;
            mFrameDiagnostics.ref_in_db = index.ref_present;
            mFrameDiagnostics.last_in_db = index.last_present;
            mFrameDiagnostics.db_map_kfs = index.map_kfs;
            mFrameDiagnostics.db_newest_kf = index.newest_kf;
            if (index.newest_ts >= 0) mFrameDiagnostics.db_newest_age_s = timestamp - index.newest_ts;
        }
        return mCurrentFrame.GetPose();
    }


    void Tracking::GrabImuData(const IMU::Point& imuMeasurement)
    {
        unique_lock<mutex> lock(mMutexImuQueue);
        mlQueueImuData.push_back(imuMeasurement);
    }

    void Tracking::PreintegrateIMU()
    {
        if (!mCurrentFrame.mpPrevFrame)
        {
            Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
            mCurrentFrame.setIntegrated();
            return;
        }

        mvImuFromLastFrame.clear();
        mvImuFromLastFrame.reserve(mlQueueImuData.size());
        if (mlQueueImuData.size() == 0)
        {
            Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
            mCurrentFrame.setIntegrated();
            return;
        }

        while (true)
        {
            bool bSleep = false;
            {
                unique_lock<mutex> lock(mMutexImuQueue);
                if (!mlQueueImuData.empty())
                {
                    IMU::Point* m = &mlQueueImuData.front();
                    cout.precision(17);
                    if (m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - mImuPer)
                    {
                        mlQueueImuData.pop_front();
                    }
                    else if (m->t < mCurrentFrame.mTimeStamp - mImuPer)
                    {
                        mvImuFromLastFrame.push_back(*m);
                        mlQueueImuData.pop_front();
                    }
                    else
                    {
                        mvImuFromLastFrame.push_back(*m);
                        break;
                    }
                }
                else
                {
                    break;
                    bSleep = true;
                }
            }
            if (bSleep)
                usleep(500);
        }

        const int n = mvImuFromLastFrame.size() - 1;
        if (n == 0)
        {
            cout << "Empty IMU measurements vector!!!\n";
            return;
        }

        IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias, mCurrentFrame.mImuCalib);

        for (int i = 0; i < n; i++)
        {
            float tstep;
            Eigen::Vector3f acc, angVel;
            if ((i == 0) && (i < (n - 1)))
            {
                float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
                float tini = mvImuFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a - (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) * 0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w - (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) * 0.5f;
                tstep = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
            }
            else if (i < (n - 1))
            {
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a) * 0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w) * 0.5f;
                tstep = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            }
            else if ((i > 0) && (i == (n - 1)))
            {
                float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
                float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp;
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a - (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) * 0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w - (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) * 0.5f;
                tstep = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
            }
            else if ((i == 0) && (i == (n - 1)))
            {
                acc = mvImuFromLastFrame[i].a;
                angVel = mvImuFromLastFrame[i].w;
                tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
            }

            if (!mpImuPreintegratedFromLastKF)
                cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
            mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);
            pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
        }

        mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
        mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

        mCurrentFrame.setIntegrated();

        //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
    }


    bool Tracking::PredictStateIMU()
    {
        if (!mCurrentFrame.mpPrevFrame)
        {
            Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
            return false;
        }

        if (mbMapUpdated && mpLastKeyFrame)
        {
            const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
            const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
            const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

            const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
            const float t12 = mpImuPreintegratedFromLastKF->dT;

            Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
            Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
            Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
            mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

            mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
            mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
            return true;
        }
        else if (!mbMapUpdated)
        {
            const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
            const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
            const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
            const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
            const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

            Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
            Eigen::Vector3f twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
            Eigen::Vector3f Vwb2 = Vwb1 + t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

            mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);

            mCurrentFrame.mImuBias = mLastFrame.mImuBias;
            mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
            return true;
        }
        else
            cout << "not IMU prediction!!" << endl;

        return false;
    }

    void Tracking::ResetFrameIMU()
    {
        // TODO To implement...
    }


    void Tracking::Track()
    {
        if (mpLocalMapper->mbBadImu)
        {
            mFrameDiagnostics.event = "bad_imu_reset_requested";
            cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
            mpSystem->ResetActiveMap();
            return;
        }

        Map* pCurrentMap = mpAtlas->GetCurrentMap();
        if (!pCurrentMap)
        {
            cout << "ERROR: There is not an active map in the atlas" << endl;
        }

        if (mState != NO_IMAGES_YET)
        {
            if (mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp)
            {
                mFrameDiagnostics.event = "timestamp_backwards_new_map";
                cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
                unique_lock<mutex> lock(mMutexImuQueue);
                mlQueueImuData.clear();
                CreateMapInAtlas();
                return;
            }
            else if (mCurrentFrame.mTimeStamp > mLastFrame.mTimeStamp + 1.0)
            {
                // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
                // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
                if (mpAtlas->isInertial())
                {

                    if (mpAtlas->isImuInitialized())
                    {
                        cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                        if (!pCurrentMap->GetIniertialBA2())
                        {
                            mpSystem->ResetActiveMap();
                        }
                        else
                        {
                            CreateMapInAtlas();
                        }
                    }
                    else
                    {
                        cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                        mpSystem->ResetActiveMap();
                    }
                    return;
                }

            }
        }


        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
            mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

        if (mState == NO_IMAGES_YET)
        {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;
        if (mLastProcessedState != OK) mCornerFallbackStreak = 0;

        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
        {
            PreintegrateIMU();
        }
        mbCreatedMap = false;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

        mbMapUpdated = false;

        int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
        int nMapChangeIndex = pCurrentMap->GetLastMapChange();
        if (nCurMapChangeIndex > nMapChangeIndex)
        {
            pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
            mbMapUpdated = true;
        }


        if (mState == NOT_INITIALIZED)
        {
            if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                StereoInitialization();
            }
            else
            {
                MonocularInitialization();
            }

            if (mState != OK) // If rightly initialized, mState=OK
            {
                mLastFrame.copyFrom(mCurrentFrame);
                return;
            }

            if (mpAtlas->GetAllMaps().size() == 1)
            {
                mnFirstFrameId = mCurrentFrame.mnId;
            }
        }
        else
        {
            // System is initialized. Track Frame.
            bool bOK;

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            if (!mbOnlyTracking)
            {

                // State OK
                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only tracking" mode.
                if (mState == OK)
                {

                    // Local Mapping might have changed some MapPoints tracked in last frame
                    CheckReplacedInLastFrame();

                    if ((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                    {
                        Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                        bOK = TrackReferenceKeyFrame();
                        if(!bOK)
                            DEBUG_INFO("[Track] TrackReferenceKeyFrame Failed");
                    }
                    else
                    {
                        Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                        bOK = TrackWithMotionModel();
                        if(!bOK)
                            DEBUG_INFO("[Track] TrackWithMotionModel Failed");
                        if (!bOK)
                            bOK = TrackReferenceKeyFrame();
                    }


                    if (!bOK)
                    {
                        if (mCurrentFrame.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) &&
                            (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                        {
                            mState = LOST;
                        }
                        else if (pCurrentMap->KeyFramesInMap() > 10)
                        {
                            // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                            mState = RECENTLY_LOST;
                            mTimeStampLost = mCurrentFrame.mTimeStamp;
                        }
                        else
                        {
                            mState = LOST;
                        }
                    }
                }
                else
                {

                    if (mState == RECENTLY_LOST)
                    {
                        mFrameDiagnostics.lost_s = mCurrentFrame.mTimeStamp - mTimeStampLost;
                        Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                        bOK = true;
                        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                        {
                            if (pCurrentMap->isImuInitialized())
                                PredictStateIMU();
                            else
                                bOK = false;

                            if (mCurrentFrame.mTimeStamp - mTimeStampLost > time_recently_lost)
                            {
                                mState = LOST;
                                Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                                bOK = false;
                            }
                        }
                        else
                        {
                            // Relocalization
                            bool short_handled = false;
                            bOK = TryShortMapRecovery(short_handled);
                            if (!short_handled) bOK = Relocalization();
                            //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                            //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                            DEBUG_INFO("[Track] mCurrentFrame.mTimeStamp = %f", mCurrentFrame.mTimeStamp);
                            DEBUG_INFO("[Track] mTimeStampLost = %f", mTimeStampLost);
                            DEBUG_INFO("[Track] mCurrentFrame.mTimeStamp - mTimeStampLost = %f", mCurrentFrame.mTimeStamp - mTimeStampLost);
                            DEBUG_INFO("[Track] bOK = %d", bOK);
                            if (mCurrentFrame.mTimeStamp - mTimeStampLost > 1.5f && !bOK)//原3fs
                            {
                                mFrameDiagnostics.event = "visual_reloc_timeout";
                                mState = LOST;
                                Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                                bOK = false;
                            }
                        }
                    }
                    else if (mState == LOST)
                    {

                        Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                        if (pCurrentMap->KeyFramesInMap() < 10)
                        {
                            mFrameDiagnostics.event = "lost_small_map_reset_requested";
                            mpSystem->ResetActiveMap();
                            Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                        }
                        else
                            CreateMapInAtlas();

                        if (mpLastKeyFrame)
                            mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                        return;
                    }
                }

            }
            else
            {
                // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
                if (mState == LOST)
                {
                    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                        Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                    bOK = Relocalization();
                }
                else
                {
                    if (!mbVO)
                    {
                        // In last frame we tracked enough MapPoints in the map
                        if (mbVelocity)
                        {
                            bOK = TrackWithMotionModel();
                        }
                        else
                        {
                            bOK = TrackReferenceKeyFrame();
                        }
                    }
                    else
                    {
                        // In last frame we tracked mainly "visual odometry" points.

                        // We compute two camera poses, one from motion model and one doing relocalization.
                        // If relocalization is sucessfull we choose that solution, otherwise we retain
                        // the "visual odometry" solution.

                        bool bOKMM = false;
                        bool bOKReloc = false;
                        vector<MapPoint*> vpMPsMM;
                        vector<bool> vbOutMM;
                        Sophus::SE3f TcwMM;
                        if (mbVelocity)
                        {
                            bOKMM = TrackWithMotionModel();
                            vpMPsMM = mCurrentFrame.mvpMapPoints;
                            vbOutMM = mCurrentFrame.mvbOutlier;
                            TcwMM = mCurrentFrame.GetPose();
                        }
                        bOKReloc = Relocalization();

                        if (bOKMM && !bOKReloc)
                        {
                            mCurrentFrame.SetPose(TcwMM);
                            mCurrentFrame.mvpMapPoints = vpMPsMM;
                            mCurrentFrame.mvbOutlier = vbOutMM;

                            if (mbVO)
                            {
                                for (int i = 0; i < mCurrentFrame.N; i++)
                                {
                                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                    {
                                        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                    }
                                }
                            }
                        }
                        else if (bOKReloc)
                        {
                            mbVO = false;
                        }

                        bOK = bOKReloc || bOKMM;
                    }
                }
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

            double timePosePred = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndPosePred - time_StartPosePred).count();
            vdPosePred_ms.push_back(timePosePred);
#endif


#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if (!mbOnlyTracking)
            {
                if (bOK)
                {
                    bOK = TrackLocalMap();
                    if (mFrameDiagnostics.short_confirmed)
                    {
                        const auto q = MeasureRecoveryQuality(mCurrentFrame, pCurrentMap, &mCornerShortPreviousPose);
                        bOK = bOK && AcceptLocalFallback(q);
                        mFrameDiagnostics.short_final_ok = bOK ? 1 : 0;
                        mFrameDiagnostics.local_ok = bOK ? 1 : 0;
                        WriteCornerEvent("SlamReloc.txt", "SLAM_SHORT_FINAL frame=%lu ts=%.9f map=%lu unique=%d cells=%d invalid=%d normalized_rmse=%.9f max_chi2=%.9f chi2_over=%d rotation_deg=%.6f translation_depth=%.9f accepted=%d local_inliers=%d",
                                mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, pCurrentMap->GetId(), q.unique,
                                q.cells, q.invalid, q.normalized_rmse, q.max_chi2, q.chi2_over,
                                q.rotation_deg, q.translation_depth, bOK ? 1 : 0, mFrameDiagnostics.local_inliers);
                        ClearShortMapRecovery();
                    }
                    if (bOK) mCornerFallbackStreak = 0;
                    else if (mState == OK) bOK = TryLocalTrackingFallback();
                }
                if (!bOK)
                    //cout << "Fail to track local map!" << endl;
                    DEBUG_INFO("[Track] Fail to track local map!");
            }
            else
            {
                // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.
                if (bOK && !mbVO)
                    bOK = TrackLocalMap();
            }

            if (bOK)
            {
                ClearShortMapRecovery();
                mState = OK;
            }
            else if (mState == OK)
            {
                if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                {
                    Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                    if (!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                    {
                        cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                        mpSystem->ResetActiveMap();
                    }

                    mState = RECENTLY_LOST;
                }
                else
                    mState = RECENTLY_LOST; // visual to lost

                /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
                {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp;
                //}
            }

            // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
            if ((mCurrentFrame.mnId < (mnLastRelocFrameId + mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
                (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
            {
                // TODO check this situation
                Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
                auto pF = new Frame;
                pF->copyFrom(mCurrentFrame);
                pF->mpPrevFrame = new Frame;
                pF->mpPrevFrame->copyFrom(mLastFrame);

                // Load preintegration
                pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
            }

            if (pCurrentMap->isImuInitialized())
            {
                if (bOK)
                {
                    if (mCurrentFrame.mnId == (mnLastRelocFrameId + mnFramesToResetIMU))
                    {
                        cout << "RESETING FRAME!!!" << endl;
                        ResetFrameIMU();
                    }
                    else if (mCurrentFrame.mnId > (mnLastRelocFrameId + 30))
                        mLastBias = mCurrentFrame.mImuBias;
                }
            }

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

            double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndLMTrack - time_StartLMTrack).count();
            vdLMTrack_ms.push_back(timeLMTrack);
#endif

            // Update drawer
            if (bOK || mState == RECENTLY_LOST)
            {
                // Update motion model
                if (mLastFrame.isSet() && mCurrentFrame.isSet())
                {
                    Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse();
                    mVelocity = mCurrentFrame.GetPose() * LastTwc;
                    mbVelocity = true;
                }
                else {
                    mbVelocity = false;
                }

                // Clean VO matches
                for (int i = 0; i < mCurrentFrame.N; i++)
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1)
                        {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                        }
                }

                // Delete temporal MapPoints
                for (list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
                {
                    MapPoint* pMP = *lit;
                    delete pMP;
                }
                mlpTemporalPoints.clear();

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
                bool bNeedKF = NeedNewKeyFrame();

                // Check if we need to insert a new keyframe
                // if(bNeedKF && bOK)
                if (bNeedKF && (bOK || (mInsertKFsLost && mState == RECENTLY_LOST &&
                    (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                    CreateNewKeyFrame();

#ifdef REGISTER_TIMES
                std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

                double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndNewKF - time_StartNewKF).count();
                vdNewKF_ms.push_back(timeNewKF);
#endif

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame. Only has effect if lastframe is tracked
                for (int i = 0; i < mCurrentFrame.N;i++)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            if (mState == LOST)
            {
                if (pCurrentMap->KeyFramesInMap() <= 10)
                {
                    mFrameDiagnostics.event = "lost_small_map_reset_requested";
                    mpSystem->ResetActiveMap();
                    return;
                }
                if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    if (!pCurrentMap->isImuInitialized())
                    {
                        Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                        mpSystem->ResetActiveMap();
                        return;
                    }

                CreateMapInAtlas();

                return;
            }

            if (!mCurrentFrame.mpReferenceKF)
            {
                mCurrentFrame.mpReferenceKF = mpReferenceKF;
            }

            mLastFrame.copyFrom(mCurrentFrame);
        }




        if (mState == OK && mCurrentFrame.isSet())
        {
            mCornerLastGoodPose = mCurrentFrame.GetPose();
            mCornerLastGoodTime = mCurrentFrame.mTimeStamp;
            mCornerLastGoodMap = pCurrentMap;
        }
        if (mState == OK || mState == RECENTLY_LOST)
        {
            // Store frame pose information to retrieve the complete camera trajectory afterwards.
            if (mCurrentFrame.isSet())
            {
                Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();
                mlRelativeFramePoses.push_back(Tcr_);
                mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
                mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                mlbLost.push_back(mState == LOST);
            }
            else
            {
                // This can happen if tracking is lost
                mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                mlpReferences.push_back(mlpReferences.back());
                mlFrameTimes.push_back(mlFrameTimes.back());
                mlbLost.push_back(mState == LOST);
            }

        }

#ifdef REGISTER_LOOP
        if (Stop()) {

            // Safe area to stop
            while (isStopped())
            {
                usleep(3000);
            }
        }
#endif
    }


    void Tracking::StereoInitialization()
    {
        if (mCurrentFrame.N > 500)
        {
            if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
                {
                    cout << "not IMU meas" << endl;
                    return;
                }

                if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA - mLastFrame.mpImuPreintegratedFrame->avgA).norm() < 0.5)
                {
                    cout << "not enough acceleration" << endl;
                    return;
                }

                if (mpImuPreintegratedFromLastKF)
                    delete mpImuPreintegratedFromLastKF;

                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            }

            // Set Frame pose to the origin (In case of inertial SLAM to imu)
            if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
                Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
                Eigen::Vector3f Vwb0;
                Vwb0.setZero();
                mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
            }
            else
                mCurrentFrame.SetPose(Sophus::SE3f());

            // Create KeyFrame
            KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

            // Insert KeyFrame in the map
            mpAtlas->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            if (!mpCamera2) {
                for (int i = 0; i < mCurrentFrame.N;i++)
                {
                    float z = mCurrentFrame.mvDepth[i];
                    if (z > 0)
                    {
                        Eigen::Vector3f x3D;
                        mCurrentFrame.UnprojectStereo(i, x3D);
                        MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                        pNewMP->AddObservation(pKFini, i);
                        pKFini->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    }
                }
            }
            else {
                for (int i = 0; i < mCurrentFrame.Nleft; i++) {
                    int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                    if (rightIndex != -1) {
                        Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                        MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                        pNewMP->AddObservation(pKFini, i);
                        pNewMP->AddObservation(pKFini, rightIndex + mCurrentFrame.Nleft);

                        pKFini->AddMapPoint(pNewMP, i);
                        pKFini->AddMapPoint(pNewMP, rightIndex + mCurrentFrame.Nleft);

                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft] = pNewMP;
                    }
                }
            }

            Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

            //cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

            mpLocalMapper->InsertKeyFrame(pKFini);

            mLastFrame.copyFrom(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;
            //mnLastRelocFrameId = mCurrentFrame.mnId;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

            mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

            mState = OK;
        }
    }


    void Tracking::MonocularInitialization()
    {

        if (!mbReadyToInitializate)
        {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > 100)
            {
                mInitialFrame.copyFrom(mCurrentFrame);
                mLastFrame.copyFrom(mCurrentFrame);
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                if (mSensor == System::IMU_MONOCULAR)
                {
                    if (mpImuPreintegratedFromLastKF)
                    {
                        delete mpImuPreintegratedFromLastKF;
                    }
                    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

                }

                mbReadyToInitializate = true;

                return;
            }
        }
        else
        {
            if (((int)mCurrentFrame.mvKeys.size() <= 100) || ((mSensor == System::IMU_MONOCULAR) && (mLastFrame.mTimeStamp - mInitialFrame.mTimeStamp > 1.0)))
            {
                mbReadyToInitializate = false;

                return;
            }

            // Find correspondences
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

            // Check if there are enough correspondences
            if (nmatches < 100)
            {
                mbReadyToInitializate = false;
                return;
            }

            Sophus::SE3f Tcw;
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

            if (mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn, mCurrentFrame.mvKeysUn, mvIniMatches, Tcw, mvIniP3D, vbTriangulated))
            {
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend;i++)
                {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                    {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                mInitialFrame.SetPose(Sophus::SE3f());
                mCurrentFrame.SetPose(Tcw);

                CreateInitialMapMonocular();
            }
        }
    }



    void Tracking::CreateInitialMapMonocular()
    {
        // Create KeyFrames
        KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
        KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        if (mSensor == System::IMU_MONOCULAR)
            pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);


        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpAtlas->AddKeyFrame(pKFini);
        mpAtlas->AddKeyFrame(pKFcur);

        for (size_t i = 0; i < mvIniMatches.size();i++)
        {
            if (mvIniMatches[i] < 0)
                continue;

            //Create MapPoint.
            Eigen::Vector3f worldPos;
            worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
            MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpAtlas->GetCurrentMap());

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            //Add to Map
            mpAtlas->AddMapPoint(pMP);
        }


        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        std::set<MapPoint*> sMPs;
        sMPs = pKFini->GetMapPoints();

        // Bundle Adjustment
        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
        Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(), 20);

        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth;
        if (mSensor == System::IMU_MONOCULAR)
            invMedianDepth = 4.0f / medianDepth; // 4.0f
        else
            invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) // TODO Check, originally 100 tracks
        {
            Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
            mpSystem->ResetActiveMap();
            return;
        }

        // Scale initial baseline
        Sophus::SE3f Tc2w = pKFcur->GetPose();
        //DEBUG_INFO("[TRacking::CreateInitialMapMonocular] translation.x = %f", Tc2w.translation().x());
        //DEBUG_INFO("[TRacking::CreateInitialMapMonocular] translation.y = %f", Tc2w.translation().y());
        //DEBUG_INFO("[TRacking::CreateInitialMapMonocular] translation.z = %f", Tc2w.translation().z());
        Tc2w.translation() *= invMedianDepth;
        //DEBUG_INFO("[TRacking::CreateInitialMapMonocular] invMedianDepth = %f", invMedianDepth);
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
        {
            if (vpAllMapPoints[iMP])
            {
                MapPoint* pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
                pMP->UpdateNormalAndDepth();
            }
        }

        if (mSensor == System::IMU_MONOCULAR)
        {
            pKFcur->mPrevKF = pKFini;
            pKFini->mNextKF = pKFcur;
            pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(), pKFcur->mImuCalib);
        }


        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);
        mpLocalMapper->mFirstTs = pKFcur->mTimeStamp;

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;
        //mnLastRelocFrameId = mInitialFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        // Compute here initial velocity
        vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

        Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
        mbVelocity = false;
        Eigen::Vector3f phi = deltaT.so3().log();

        float aux = (float)(mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) / (float)(mCurrentFrame.mTimeStamp - mInitialFrame.mTimeStamp);
        phi *= aux;

        mLastFrame.copyFrom(mCurrentFrame);

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;

        initID = (int)pKFcur->mnId;
    }


    void Tracking::CreateMapInAtlas()
    {
        ClearShortMapRecovery();
        mCornerLastGoodTime = -1;
        mCornerLastGoodMap = nullptr;
        mCornerFallbackStreak = 0;
        if (std::string(mFrameDiagnostics.event) == "none")
            mFrameDiagnostics.event = "create_new_map";
        __android_log_print(ANDROID_LOG_INFO, "SLAM_DIAG",
                "SLAM_EVENT type=create_map frame=%lu ts=%.9f reason=%s",
                mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, mFrameDiagnostics.event);
        mnLastInitFrameId = mCurrentFrame.mnId;
        mpAtlas->CreateNewMap();
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
            mpAtlas->SetInertialSensor();
        mbSetInit = false;

        mnInitialFrameId = mCurrentFrame.mnId + 1;
        mState = NO_IMAGES_YET;

        // Restart the variable with information about the last KF
        mbVelocity = false;
        //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
        Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId + 1), Verbose::VERBOSITY_NORMAL);
        mbVO = false; // Init value for know if there are enough MapPoints in the last KF
        if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
        {
            mbReadyToInitializate = false;
        }

        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
        {
            delete mpImuPreintegratedFromLastKF;
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
        }

        //if (mpLastKeyFrame)
        //{
        //    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
        //}
        mpLastKeyFrame = nullptr;

        //if (mpReferenceKF)
        //{
        //    mpReferenceKF = static_cast<KeyFrame*>(NULL);
        //}
        mpReferenceKF = nullptr;

        //mLastFrame = Frame();
        //mCurrentFrame = Frame();
        mLastFrame.reset();
        mCurrentFrame.reset();
        mvIniMatches.clear();

        mbCreatedMap = true;
    }

    void Tracking::CheckReplacedInLastFrame()
    {
        for (int i = 0; i < mLastFrame.N; i++)
        {
            MapPoint* pMP = mLastFrame.mvpMapPoints[i];

            if (pMP)
            {
                MapPoint* pRep = pMP->GetReplaced();
                if (pRep)
                {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }
    }


    bool Tracking::TrackReferenceKeyFrame()
    {
        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        vector<MapPoint*> vpMapPointMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        mFrameDiagnostics.ref_raw = nmatches;

        if (nmatches < 15)
        {
            cout << "TRACK_REF_KF: Less than 15 matches!!\n";
            return false;
        }

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.SetPose(mLastFrame.GetPose());

        //mCurrentFrame.PrintPointDistribution();


        // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            //if(i >= mCurrentFrame.Nleft) break;
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    if (i < mCurrentFrame.Nleft) {
                        pMP->mbTrackInView = false;
                    }
                    else {
                        pMP->mbTrackInViewR = false;
                    }
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        mFrameDiagnostics.ref_inliers = nmatchesMap;
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return nmatchesMap >= 10;
    }

    void Tracking::UpdateLastFrame()
    {
        // Update pose according to reference keyframe
        KeyFrame* pRef = mLastFrame.mpReferenceKF;
        Sophus::SE3f Tlr = mlRelativeFramePoses.back();
        mLastFrame.SetPose(Tlr * pRef->GetPose());

        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vDepthIdx;
        const int Nfeat = mLastFrame.Nleft == -1 ? mLastFrame.N : mLastFrame.Nleft;
        vDepthIdx.reserve(Nfeat);
        for (int i = 0; i < Nfeat;i++)
        {
            float z = mLastFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (vDepthIdx.empty())
            return;

        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size();j++)
        {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            MapPoint* pMP = mLastFrame.mvpMapPoints[i];

            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1)
                bCreateNew = true;

            if (bCreateNew)
            {
                Eigen::Vector3f x3D;

                if (mLastFrame.Nleft == -1) {
                    mLastFrame.UnprojectStereo(i, x3D);
                }
                else {
                    x3D = mLastFrame.UnprojectStereoFishEye(i);
                }

                MapPoint* pNewMP = new MapPoint(x3D, mpAtlas->GetCurrentMap(), &mLastFrame, i);
                mLastFrame.mvpMapPoints[i] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            }
            else
            {
                nPoints++;
            }

            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;

        }
    }

    bool Tracking::TrackWithMotionModel()
    {
        ORBmatcher matcher(0.9, true);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU))
        {
            // Predict state with IMU if it is initialized and it doesnt need reset
            PredictStateIMU();
            return true;
        }
        else
        {
            mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
        }




        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(NULL));

        // Project points seen in previous frame
        int th;

        if (mSensor == System::STEREO)
            th = 7;
        else
            th = 15;

        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
        mFrameDiagnostics.motion_raw = nmatches;

        // If few matches, uses a wider window search
        if (nmatches < 20)
        {
            Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(NULL));

            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
            mFrameDiagnostics.motion_wide = nmatches;
            Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

        }
        //DEBUG_INFO("[TrackWithMotionModel] nmatches = %d ", nmatches);

        if (nmatches < 20)
        {
            Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                return true;
            else
                return false;
        }

        // Optimize frame pose with all matches
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (mCurrentFrame.mvbOutlier[i])
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    if (i < mCurrentFrame.Nleft) {
                        pMP->mbTrackInView = false;
                    }
                    else {
                        pMP->mbTrackInViewR = false;
                    }
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                }
                else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }
        //DEBUG_INFO("[TrackWithMotionModel] nmatchesMap = %d ", nmatchesMap);
        mFrameDiagnostics.motion_inliers = nmatchesMap;
        if (mbOnlyTracking)
        {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return nmatchesMap >= 10;
    }

    void Tracking::ClearShortMapRecovery()
    {
        mCornerShortChain.Reset();
        mCornerShortFrame.reset();
    }

    bool Tracking::TryShortMapRecovery(bool& handled)
    {
        handled = false;
        if (!CornerShortRecoveryEnabled() || mSensor != System::MONOCULAR || mbOnlyTracking ||
            mState != RECENTLY_LOST || !mCornerShortFrame || !mCornerShortChain.active) return false;
        Map* map = mpAtlas->GetCurrentMap();
        const double age = mCurrentFrame.mTimeStamp-mCornerLastGoodTime;
        const double gap = mCurrentFrame.mTimeStamp-mCornerShortChain.timestamp;
        if (mCornerLastGoodMap != map || map->GetId() != mCornerShortChain.map || age <= 0 ||
            age > kShortRecoveryWindow || gap <= 0 || gap > kFallbackMaxFrameGap)
        {
            WriteCornerEvent("SlamReloc.txt", "SLAM_SHORT_RESET frame=%lu ts=%.9f reason=window_or_map age_s=%.6f gap_s=%.6f",
                    mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, age, gap);
            ClearShortMapRecovery();
            return false;
        }
        KeyFrame* seed = mCornerShortFrame->mpReferenceKF;
        if (!seed || seed->isBad() || seed->GetMap() != map) seed = mpLastKeyFrame;
        if (!seed || seed->isBad() || seed->GetMap() != map) { ClearShortMapRecovery(); return false; }
        const auto began = std::chrono::steady_clock::now();
        Frame original; original.copyFrom(mCurrentFrame);
        const auto prior_pose = mCornerShortFrame->GetPose();
        const auto prior_frame = mCornerShortChain.frame;
        const int prior_points = mCornerShortChain.points;
        unordered_set<unsigned long> previous_ids;
        for (MapPoint* point : mCornerShortFrame->mvpMapPoints)
            if (point && !point->isBad() && point->GetMap() == map) previous_ids.insert(point->mnId);
        mCurrentFrame.SetPose(prior_pose);
        mCurrentFrame.mvpMapPoints.assign(mCurrentFrame.N, nullptr);
        mCurrentFrame.mvbOutlier.assign(mCurrentFrame.N, false);
        auto pool = BuildRecoveryPool(seed, map, RecentRecoveryKeyFrames(map, mCurrentFrame.mTimeStamp),
                mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, nullptr);
        ORBmatcher matcher(.9, true);
        set<MapPoint*> found;
        const int seed_added = matcher.SearchByProjection(mCurrentFrame, seed, found, 10, 100);
        for (MapPoint* point : mCurrentFrame.mvpMapPoints) if (point) found.insert(point);
        RecoveryProjectionDiagnostics projection;
        const int extra_added = matcher.SearchRecoveryPool(mCurrentFrame, pool.points, found, 10, 100, projection);
        DeduplicateRecoveryFrame(mCurrentFrame);
        Optimizer::PoseOptimization(&mCurrentFrame);
        vector<RecoveryResidual> residuals;
        const auto q = MeasureRecoveryQuality(mCurrentFrame, map, &prior_pose, &pool, &residuals);
        auto anchor = q;
        const auto anchor_delta = mCurrentFrame.GetPose()*mCornerLastGoodPose.inverse();
        anchor.rotation_deg = anchor_delta.so3().log().norm()*180.0/M_PI;
        anchor.translation_depth = q.median_depth > 0 ? anchor_delta.translation().norm()/q.median_depth : -1;
        const bool valid = AcceptLocalFallback(q) && AcceptShortAnchor(anchor, age);
        int shared = 0;
        for (size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); ++i)
            if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i] &&
                previous_ids.count(mCurrentFrame.mvpMapPoints[i]->mnId)) ++shared;
        bool confirmed = false;
        if (valid)
        {
            confirmed = mCornerShortChain.Observe(map->GetId(), mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, q.unique, shared);
            mCornerShortPreviousPose = prior_pose;
            for (size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); ++i)
                if (mCurrentFrame.mvbOutlier[i]) mCurrentFrame.mvpMapPoints[i] = nullptr;
            mCornerShortFrame->copyFrom(mCurrentFrame);
            mCornerShortFrame->mpReferenceKF = seed;
            mFrameDiagnostics.short_pending = 1;
            mFrameDiagnostics.short_confirmed = confirmed ? 1 : 0;
            if (confirmed) { mpReferenceKF = seed; mCurrentFrame.mpReferenceKF = seed; }
            handled = true;
        }
        WriteCornerEvent("SlamReloc.txt", "SLAM_SHORT_TRY frame=%lu ts=%.9f map=%lu seed=%lu age_s=%.6f gap_s=%.6f seed_added=%d extra_added=%d unique=%d cells=%d invalid=%d normalized_rmse=%.9f max_chi2=%.9f chi2_over=%d rotation_deg=%.6f translation_depth=%.9f anchor_rotation_deg=%.6f anchor_translation_depth=%.9f shared=%d streak=%d valid=%d confirmed=%d total_ms=%.3f previous_frame=%llu previous_points=%d",
                mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, map->GetId(), seed->mnId, age, gap, seed_added,
                extra_added, q.unique, q.cells, q.invalid, q.normalized_rmse, q.max_chi2, q.chi2_over,
                q.rotation_deg, q.translation_depth, anchor.rotation_deg, anchor.translation_depth, shared,
                valid ? mCornerShortChain.streak : 0, valid ? 1 : 0, confirmed ? 1 : 0,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-began).count(),
                static_cast<unsigned long long>(prior_frame), prior_points);
        vector<string> lines;
        AppendRecoveryResiduals(lines, "short", mCurrentFrame, q.unique, residuals);
        WriteCornerEvents("SlamReloc.txt", lines);
        if (!valid) { mCurrentFrame.copyFrom(original); ClearShortMapRecovery(); }
        return confirmed;
    }

    bool Tracking::TryLocalTrackingFallback()
    {
        const double gap = mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp;
        const int original_inliers = mFrameDiagnostics.local_inliers;
        if (mbOnlyTracking || mState != OK || !mCurrentFrame.isSet() || !mLastFrame.isSet() ||
            !LocalFallbackEligible(mSensor == System::MONOCULAR, mLastProcessedState == OK,
                mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames, original_inliers, gap, mCornerFallbackStreak))
            return false;
        Map* active_map = mpAtlas->GetCurrentMap();
        KeyFrame* seed = mpLastKeyFrame;
        if (!seed || seed->isBad() || seed->GetMap() != active_map) return false;
        const auto began = std::chrono::steady_clock::now();
        ++mCornerFallbackStreak;
        mFrameDiagnostics.fallback_attempted = 1;
        mFrameDiagnostics.fallback_original_inliers = original_inliers;
        const auto saved_pose = mCurrentFrame.GetPose();
        const auto saved_points = mCurrentFrame.mvpMapPoints;
        const auto saved_outliers = mCurrentFrame.mvbOutlier;
        unordered_set<unsigned long> old_inliers, unique;
        set<MapPoint*> found;
        for (size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); ++i)
        {
            MapPoint* point = mCurrentFrame.mvpMapPoints[i];
            if (point && !mCurrentFrame.mvbOutlier[i] && !point->isBad() && point->Observations() > 0 &&
                point->GetMap() == active_map && unique.insert(point->mnId).second)
            { found.insert(point); old_inliers.insert(point->mnId); }
            else mCurrentFrame.mvpMapPoints[i] = nullptr;
            mCurrentFrame.mvbOutlier[i] = false;
        }
        vector<KeyFrame*> recent;
        for (KeyFrame* keyframe : active_map->GetAllKeyFrames())
        {
            const double age = mCurrentFrame.mTimeStamp-keyframe->mTimeStamp;
            if (!keyframe->isBad() && age >= 0 && age <= kRecoveryPoolRecentSeconds) recent.push_back(keyframe);
        }
        sort(recent.begin(), recent.end(), [](KeyFrame* a, KeyFrame* b) {
            return a->mTimeStamp == b->mTimeStamp ? a->mnId > b->mnId : a->mTimeStamp > b->mTimeStamp;
        });
        auto pool = BuildRecoveryPool(seed, active_map, recent, mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, nullptr);
        ORBmatcher matcher(.9, true);
        const int seed_added = matcher.SearchByProjection(mCurrentFrame, seed, found, 10, 100);
        found.clear();
        for (MapPoint* point : mCurrentFrame.mvpMapPoints) if (point) found.insert(point);
        RecoveryProjectionDiagnostics projection;
        const int extra_added = matcher.SearchRecoveryPool(mCurrentFrame, pool.points, found, 10, 100, projection);
        // A seed KF may itself contain repeated point associations. Never let
        // these multiply the constraints submitted to the retry optimizer.
        unique.clear();
        for (auto& point : mCurrentFrame.mvpMapPoints)
            if (point && !unique.insert(point->mnId).second) point = nullptr;
        const int optimized = Optimizer::PoseOptimization(&mCurrentFrame);
        const auto previous_pose = mLastFrame.GetPose();
        vector<RecoveryResidual> residuals;
        const auto quality = MeasureRecoveryQuality(mCurrentFrame, active_map, &previous_pose, &pool, &residuals);
        const bool accepted = AcceptLocalFallback(quality);
        WriteCornerEvent("SlamReloc.txt", "SLAM_LOCAL_FALLBACK_PROJ frame=%lu ts=%.9f tested=%d seed_excluded=%d already_found=%d invalid=%d depth=%d bounds=%d distance=%d no_features=%d no_free=%d descriptor=%d orientation=%d matched=%d",
                mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, projection.tested, projection.seed_excluded,
                projection.already_found, projection.invalid, projection.depth, projection.bounds, projection.distance,
                projection.no_features, projection.no_free, projection.descriptor, projection.orientation, projection.matched);
        WriteCornerEvent("SlamReloc.txt", "SLAM_LOCAL_FALLBACK frame=%lu ts=%.9f map=%lu seed=%lu initial_inliers=%d frame_gap_s=%.6f streak=%d pool_points=%zu seed_added=%d extra_added=%d optimized=%d associations=%d unique=%d duplicates=%d invalid=%d cells=%d extra_inliers=%d rmse_px=%.6f median_px=%.6f p95_px=%.6f rotation_deg=%.6f translation_depth=%.9f accepted=%d total_ms=%.3f stage=before_recently_lost normalized_rmse=%.9f normalized_p95=%.9f max_chi2=%.9f chi2_over=%d quality_policy=octave_chi2",
                mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, active_map->GetId(), seed->mnId, original_inliers,
                gap, mCornerFallbackStreak, pool.points.size(), seed_added, extra_added, optimized, quality.associations,
                quality.unique, quality.duplicates, quality.invalid, quality.cells, quality.extra,
                quality.rmse, quality.median, quality.p95, quality.rotation_deg, quality.translation_depth,
                accepted ? 1 : 0, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-began).count(),
                quality.normalized_rmse, quality.normalized_p95, quality.max_chi2, quality.chi2_over);
        vector<string> residual_lines;
        AppendRecoveryResiduals(residual_lines, "local", mCurrentFrame, quality.unique, residuals);
        WriteCornerEvents("SlamReloc.txt", residual_lines);
        if (!accepted)
        {
            mCurrentFrame.SetPose(saved_pose);
            mCurrentFrame.mvpMapPoints = saved_points;
            mCurrentFrame.mvbOutlier = saved_outliers;
            return false;
        }
        mFrameDiagnostics.fallback_accepted = 1;
        mnMatchesInliers = quality.unique;
        mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
        mFrameDiagnostics.local_inliers = mnMatchesInliers;
        mFrameDiagnostics.local_matches = 0;
        mFrameDiagnostics.local_outliers = 0;
        mFrameDiagnostics.recent_inliers = 0;
        mFrameDiagnostics.local_ok = 1;
        mFrameDiagnostics.inlier_cells = quality.cells;
        mFrameDiagnostics.map_match_labels.assign(mCurrentFrame.N, 0);
        for (size_t i = 0; i < mCurrentFrame.mvpMapPoints.size(); ++i)
        {
            MapPoint* point = mCurrentFrame.mvpMapPoints[i];
            if (!point) continue;
            ++mFrameDiagnostics.local_matches;
            if (mCurrentFrame.mvbOutlier[i])
            { ++mFrameDiagnostics.local_outliers; mFrameDiagnostics.map_match_labels[i] = 2; continue; }
            mFrameDiagnostics.map_match_labels[i] = 1;
            if (!old_inliers.count(point->mnId)) point->IncreaseFound();
            point->mnLastFrameSeen = mCurrentFrame.mnId;
            if (point->mDebugBirthTime >= 0 && mCurrentFrame.mTimeStamp-point->mDebugBirthTime <= 2.0)
                ++mFrameDiagnostics.recent_inliers;
        }
        return true;
    }

    bool Tracking::TrackLocalMap()
    {
        const auto debugResult = [this](bool ok) {
            mFrameDiagnostics.local_ok = ok ? 1 : 0;
            return ok;
        };

        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.
        mTrackedFr++;

        UpdateLocalMap();
        mFrameDiagnostics.local_points = static_cast<int>(mvpLocalMapPoints.size());
        SearchLocalPoints();

        // TOO check outliers before PO
        int aux1 = 0, aux2 = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i])
            {
                aux1++;
                if (mCurrentFrame.mvbOutlier[i])
                    aux2++;
            }

        int inliers;
        if (!mpAtlas->isImuInitialized())
            Optimizer::PoseOptimization(&mCurrentFrame);
        else
        {
            if (mCurrentFrame.mnId <= mnLastRelocFrameId + mnFramesToResetIMU)
            {
                Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
                Optimizer::PoseOptimization(&mCurrentFrame);
            }
            else
            {
                // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
                if (!mbMapUpdated) //  && (mnMatchesInliers>30))
                {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                }
                else
                {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                }
            }
        }

        aux1 = 0, aux2 = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i])
            {
                aux1++;
                if (mCurrentFrame.mvbOutlier[i])
                    aux2++;
            }

        mFrameDiagnostics.local_matches = aux1;
        mFrameDiagnostics.local_outliers = aux2;
        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvpMapPoints[i])
            {
                if (!mCurrentFrame.mvbOutlier[i])
                {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking)
                    {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                        {
                            mnMatchesInliers++;
                            const double birth = mCurrentFrame.mvpMapPoints[i]->mDebugBirthTime;
                            if (birth >= 0 && mCurrentFrame.mTimeStamp - birth <= 2.0)
                                ++mFrameDiagnostics.recent_inliers;
                        }
                    }
                    else
                        mnMatchesInliers++;
                }
                else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
            }
        }

        // Decide if the tracking was succesful
        mFrameDiagnostics.local_inliers = mnMatchesInliers;
        mFrameDiagnostics.map_match_labels.assign(mCurrentFrame.N, 0);
        for (int i = 0; i < mCurrentFrame.N; ++i)
            if (mCurrentFrame.mvpMapPoints[i])
                mFrameDiagnostics.map_match_labels[i] = mCurrentFrame.mvbOutlier[i] ? 2 :
                        (mCurrentFrame.mvpMapPoints[i]->Observations() > 0 ? 1 : 0);
        bool occupied[48] = {};
        for (int i = 0; i < mCurrentFrame.N; ++i)
        {
            if (i >= static_cast<int>(mCurrentFrame.mvKeysUn.size())) continue;
            if (!mCurrentFrame.mvpMapPoints[i] || mCurrentFrame.mvbOutlier[i]) continue;
            const auto& point = mCurrentFrame.mvKeysUn[i].pt;
            const int x = std::max(0, std::min(7, static_cast<int>(point.x * 8 / mImGray.cols)));
            const int y = std::max(0, std::min(5, static_cast<int>(point.y * 6 / mImGray.rows)));
            occupied[y * 8 + x] = true;
        }
        mFrameDiagnostics.inlier_cells = 0;
        for (bool cell : occupied) if (cell) ++mFrameDiagnostics.inlier_cells;
        // More restrictive if there was a relocalization recently
        mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
            return debugResult(false);

        if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST) && !mFrameDiagnostics.short_confirmed)
            return debugResult(true);


        if (mSensor == System::IMU_MONOCULAR)
        {
            if ((mnMatchesInliers < 15 && mpAtlas->isImuInitialized()) || (mnMatchesInliers < 50 && !mpAtlas->isImuInitialized()))
            {
                return debugResult(false);
            }
            else
                return debugResult(true);
        }
        else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (mnMatchesInliers < 15)
            {
                return debugResult(false);
            }
            else
                return debugResult(true);
        }
        else
        {
            if (mnMatchesInliers < 30)
                return debugResult(false);
            else
                return debugResult(true);
        }
    }

    bool Tracking::NeedNewKeyFrame()
    {
        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
        {
            if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
                return true;
            else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
                return true;
            else
                return false;
        }

        if (mbOnlyTracking)
            return false;

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
            /*if(mSensor == System::MONOCULAR)
            {
                std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
            }*/
            return false;
        }

        const int nKFs = mpAtlas->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation
        if (mCurrentFrame.mnId<mnLastRelocFrameId + mMaxFrames && nKFs>mMaxFrames)
        {
            return false;
        }

        // Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose = 0;

        if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR)
        {
            int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
            for (int i = 0; i < N; i++)
            {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
                {
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                        nTrackedClose++;
                    else
                        nNonTrackedClose++;

                }
            }
            //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
        }

        bool bNeedToInsertClose;
        bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

        // Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
        const int thStereoClosedPoints = 15;
        if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
        {
            //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
            thRefRatio = 0.9f;
        }*/

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        if (mpCamera2) thRefRatio = 0.75f;

        if (mSensor == System::IMU_MONOCULAR)
        {
            if (mnMatchesInliers > 350) // Points tracked from the local map
                thRefRatio = 0.75f;
            else
                thRefRatio = 0.90f;
        }

        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = ((mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);
        //Condition 1c: tracking is weak
        const bool c1c = mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR && mSensor != System::IMU_STEREO && mSensor != System::IMU_RGBD && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = (((mnMatchesInliers < nRefMatches* thRefRatio || bNeedToInsertClose)) && mnMatchesInliers > 15);

        //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
        // Temporal condition for Inertial cases
        bool c3 = false;
        if (mpLastKeyFrame)
        {
            if (mSensor == System::IMU_MONOCULAR)
            {
                if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
                    c3 = true;
            }
            else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
                    c3 = true;
            }
        }

        bool c4 = false;
        if ((((mnMatchesInliers < 75) && (mnMatchesInliers > 15)) || mState == RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
            c4 = true;
        else
            c4 = false;

        if (((c1a || c1b || c1c) && c2) || c3 || c4)
        {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if (bLocalMappingIdle || mpLocalMapper->IsInitializing())
            {
                return true;
            }
            else
            {
                mpLocalMapper->InterruptBA();
                if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR)
                {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                        return true;
                    else
                        return false;
                }
                else
                {
                    //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                    return false;
                }
            }
        }
        else
            return false;
    }

    void Tracking::CreateNewKeyFrame()
    {
        if (mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
            return;

        if (!mpLocalMapper->SetNotStop(true))
            return;

        KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        if (mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
            pKF->bImu = true;

        pKF->SetNewBias(mCurrentFrame.mImuBias);
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        if (mpLastKeyFrame)
        {
            pKF->mPrevKF = mpLastKeyFrame;
            mpLastKeyFrame->mNextKF = pKF;
        }
        else
            Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

        // Reset preintegration from last KF (Create new object)
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(), pKF->mImuCalib);
        }

        if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
        {
            mCurrentFrame.UpdatePoseMatrices();
            // cout << "create new MPs" << endl;
            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            int maxPoint = 100;
            if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                maxPoint = 100;

            vector<pair<float, int> > vDepthIdx;
            int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
            vDepthIdx.reserve(mCurrentFrame.N);
            for (int i = 0; i < N; i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0)
                {
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty())
            {
                sort(vDepthIdx.begin(), vDepthIdx.end());

                int nPoints = 0;
                for (size_t j = 0; j < vDepthIdx.size();j++)
                {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1)
                    {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                    }

                    if (bCreateNew)
                    {
                        Eigen::Vector3f x3D;

                        if (mCurrentFrame.Nleft == -1) {
                            mCurrentFrame.UnprojectStereo(i, x3D);
                        }
                        else {
                            x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                        }

                        MapPoint* pNewMP = new MapPoint(x3D, pKF, mpAtlas->GetCurrentMap());
                        pNewMP->AddObservation(pKF, i);

                        //Check if it is a stereo observation in order to not
                        //duplicate mappoints
                        if (mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0) {
                            mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]] = pNewMP;
                            pNewMP->AddObservation(pKF, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                            pKF->AddMapPoint(pNewMP, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        }

                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;
                    }
                    else
                    {
                        nPoints++;
                    }

                    if (vDepthIdx[j].first > mThDepth && nPoints > maxPoint)
                    {
                        break;
                    }
                }
                //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
            }
        }


        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);
        mFrameDiagnostics.new_keyframe = 1;

        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
        WriteCornerEvent("SlamKeyframes.txt", "SLAM_KF_CREATE frame=%lu kf=%lu ts=%.9f map=%lu inliers=%d",
                mCurrentFrame.mnId, pKF->mnId, pKF->mTimeStamp, pKF->GetMap()->GetId(), mnMatchesInliers);
    }

    void Tracking::SearchLocalPoints()
    {
        // Do not search map points already matched
        for (vector<MapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
        {
            MapPoint* pMP = *vit;
            if (pMP)
            {
                if (pMP->isBad())
                {
                    *vit = static_cast<MapPoint*>(NULL);
                }
                else
                {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                    pMP->mbTrackInViewR = false;
                }
            }
        }

        int nToMatch = 0;

        // Project points in frame and check its visibility
        for (vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
        {
            MapPoint* pMP = *vit;

            const bool recent = pMP->mDebugBirthTime >= 0 &&
                    mCurrentFrame.mTimeStamp - pMP->mDebugBirthTime <= 2.0;
            if (recent) ++mFrameDiagnostics.recent_points;
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            {
                if (recent) ++mFrameDiagnostics.recent_preassociated;
                continue;
            }
            if (pMP->isBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            int reason = 5;
            ++mFrameDiagnostics.projection_tested;
            if (recent) ++mFrameDiagnostics.recent_tested;
            if (mCurrentFrame.isInFrustum(pMP, 0.5, &reason))
            {
                pMP->IncreaseVisible();
                nToMatch++;
                if (recent) ++mFrameDiagnostics.recent_visible;
            }
            ++mFrameDiagnostics.projection_rejected[reason];
            if (recent) ++mFrameDiagnostics.recent_projection[reason];
            if (pMP->mbTrackInView)
            {
                mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
            }
        }

        mFrameDiagnostics.local_visible = nToMatch;
        mFrameDiagnostics.local_added = 0;
        if (nToMatch > 0)
        {
            ORBmatcher matcher(0.8);
            int th = 1;
            if (mSensor == System::RGBD || mSensor == System::IMU_RGBD)
                th = 3;
            if (mpAtlas->isImuInitialized())
            {
                if (mpAtlas->GetCurrentMap()->GetIniertialBA2())
                    th = 2;
                else
                    th = 6;
            }
            else if (!mpAtlas->isImuInitialized() && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
            {
                th = 10;
            }

            // If the camera has been relocalised recently, perform a coarser search
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;

            if (mState == LOST || mState == RECENTLY_LOST) // Lost for less than 1 second
                th = 15; // 15

            int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints,
                    mpLocalMapper->mThFarPoints, &mFrameDiagnostics.projection_match);
            mFrameDiagnostics.local_added = matches;
        }
    }

    void Tracking::UpdateLocalMap()
    {
        // This is for visualization
        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        // Update
        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

    void Tracking::UpdateLocalPoints()
    {
        mvpLocalMapPoints.clear();

        int count_pts = 0;

        for (vector<KeyFrame*>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(), itEndKF = mvpLocalKeyFrames.rend(); itKF != itEndKF; ++itKF)
        {
            KeyFrame* pKF = *itKF;
            const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
            {

                MapPoint* pMP = *itMP;
                if (!pMP)
                    continue;
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad())
                {
                    count_pts++;
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }


    void Tracking::UpdateLocalKeyFrames()
    {
        // Each map point vote for the keyframes in which it has been observed
        map<KeyFrame*, int> keyframeCounter;
        if (!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2))
        {
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                {
                    if (!pMP->isBad())
                    {
                        const map<KeyFrame*, tuple<int, int>> observations = pMP->GetObservations();
                        for (map<KeyFrame*, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                            keyframeCounter[it->first]++;
                    }
                    else
                    {
                        mCurrentFrame.mvpMapPoints[i] = NULL;
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < mLastFrame.N; i++)
            {
                // Using lastframe since current frame has not matches yet
                if (mLastFrame.mvpMapPoints[i])
                {
                    MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                    if (!pMP)
                        continue;
                    if (!pMP->isBad())
                    {
                        const map<KeyFrame*, tuple<int, int>> observations = pMP->GetObservations();
                        for (map<KeyFrame*, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                            keyframeCounter[it->first]++;
                    }
                    else
                    {
                        // MODIFICATION
                        mLastFrame.mvpMapPoints[i] = NULL;
                    }
                }
            }
        }


        int max = 0;
        KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
        {
            KeyFrame* pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max)
            {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(pKF);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }

        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
        {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80) // 80
                break;

            KeyFrame* pKF = *itKF;

            const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


            for (vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
            {
                KeyFrame* pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad())
                {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            const set<KeyFrame*> spChilds = pKF->GetChilds();
            for (set<KeyFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
            {
                KeyFrame* pChildKF = *sit;
                if (!pChildKF->isBad())
                {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                    {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            KeyFrame* pParent = pKF->GetParent();
            if (pParent)
            {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // Add 10 last temporal KFs (mainly for IMU)
        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mvpLocalKeyFrames.size() < 80)
        {
            KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

            const int Nd = 20;
            for (int i = 0; i < Nd; i++) {
                if (!tempKeyFrame)
                    break;
                if (tempKeyFrame->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(tempKeyFrame);
                    tempKeyFrame->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    tempKeyFrame = tempKeyFrame->mPrevKF;
                }
            }
        }

        if (pKFmax)
        {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    bool Tracking::Relocalization()
    {
        Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
        // Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());
        mFrameDiagnostics.reloc_candidates = static_cast<int>(vpCandidateKFs.size());

        if (vpCandidateKFs.empty()) {
            Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
            return false;
        }

        const int nKFs = vpCandidateKFs.size();

        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);

        vector<unique_ptr<MLPnPsolver>> vpMLPnPsolvers;
        vpMLPnPsolvers.resize(nKFs);

        vector<vector<MapPoint*> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        struct SolverDiagnostics
        {
            int eligible = 0, attempts = 0, poses = 0, pnp_best = -1, initial_good_best = -1;
            int coarse_calls = 0, coarse_added_best = -1, coarse_gate_best = -1, coarse_good_best = -1;
            int fine_calls = 0, fine_added_best = -1, fine_gate_best = -1, final_good_best = -1, recovered = 0;
            int pool_points = -1, extra_points = -1, pool_short = 0, expand_calls = 0, expand_budget_skips = 0;
            int expand_added_best = -1, expand_gate_best = -1, extra_inliers_best = -1;
            int quality_best = -1;
            int short_opt_calls = 0;
            double pnp_ms = 0, optimize_ms = 0;
        };
        vector<SolverDiagnostics> solver_diagnostics(nKFs);

        int nCandidates = 0;

        for (int i = 0; i < nKFs; i++)
        {
            KeyFrame* pKF = vpCandidateKFs[i];
            if (i < 8)
                WriteCornerEvent("SlamReloc.txt", "SLAM_RELOC_CANDIDATE frame=%lu ts=%.9f rank=%d kf=%lu kf_ts=%.9f age_s=%.6f bad=%d available_mps=%d common_words=%d score=%.6f",
                        mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, i, pKF->mnId, pKF->mTimeStamp,
                        mCurrentFrame.mTimeStamp - pKF->mTimeStamp, pKF->isBad() ? 1 : 0,
                        pKF->TrackedMapPoints(1), pKF->mnRelocWords, pKF->mRelocScore);
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else
            {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                mFrameDiagnostics.reloc_bow_best = std::max(mFrameDiagnostics.reloc_bow_best, nmatches);
                if (i < 8)
                    WriteCornerEvent("SlamReloc.txt", "SLAM_RELOC_BOW frame=%lu ts=%.9f kf=%lu matches=%d pnp_eligible=%d",
                            mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, pKF->mnId, nmatches, nmatches >= 15 ? 1 : 0);
                if (nmatches < 15)
                {
                    vbDiscarded[i] = true;
                    continue;
                }
                else
                {
                    solver_diagnostics[i].eligible = 1;
                    auto pSolver = make_unique<MLPnPsolver>(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 6, 0.5, 5.991);  //This solver needs at least 6 points
                    vpMLPnPsolvers[i] = std::move(pSolver);
                    nCandidates++;
                }
            }
        }

        const int debug_pnp_candidates = nCandidates;
        // Alternatively perform some iterations of P4P RANSAC
        // Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);
        const bool expand_enabled = mSensor == System::MONOCULAR;
        Map* recovery_map = mpAtlas->GetCurrentMap();
        vector<KeyFrame*> recent_keyframes;
        if (expand_enabled && nCandidates > 0)
        {
            for (KeyFrame* keyframe : recovery_map->GetAllKeyFrames())
            {
                const double age = mCurrentFrame.mTimeStamp - keyframe->mTimeStamp;
                if (!keyframe->isBad() && keyframe->GetMap() == recovery_map && age >= 0 && age <= kRecoveryPoolRecentSeconds)
                    recent_keyframes.push_back(keyframe);
            }
            sort(recent_keyframes.begin(), recent_keyframes.end(), [](KeyFrame* a, KeyFrame* b) {
                return a->mTimeStamp == b->mTimeStamp ? a->mnId > b->mnId : a->mTimeStamp > b->mTimeStamp;
            });
        }
        vector<RecoveryPoolSnapshot> recovery_pools(nKFs);
        vector<string> recovery_lines;
        const bool short_query = CornerShortRecoveryEnabled() && expand_enabled && !mbOnlyTracking &&
                mState == RECENTLY_LOST && mCornerLastGoodMap == recovery_map && mCornerLastGoodTime >= 0 &&
                mCurrentFrame.mTimeStamp > mCornerLastGoodTime &&
                mCurrentFrame.mTimeStamp-mCornerLastGoodTime <= kShortRecoveryWindow;
        vector<vector<string>> quality_lines(nKFs);
        int best_residual_count = -1, short_seed_unique = -1;
        vector<RecoveryResidual> best_residuals;
        vector<MapPoint*> short_seed_points;
        vector<bool> short_seed_outliers;
        Sophus::SE3f short_seed_pose;
        KeyFrame* short_seed = nullptr;
        RecoveryQuality short_seed_quality;
        auto optimize_candidate = [&](int candidate, const char* stage) {
            const auto began = std::chrono::steady_clock::now();
            const int good = Optimizer::PoseOptimization(&mCurrentFrame);
            auto& d = solver_diagnostics[candidate];
            d.optimize_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-began).count();
            if (candidate < 8 && good > d.quality_best)
            {
                d.quality_best = good;
                const bool has_reference = mCornerLastGoodTime >= 0 && mCornerLastGoodMap == recovery_map &&
                        mCornerLastGoodTime <= mCurrentFrame.mTimeStamp;
                vector<RecoveryResidual> residuals;
                const auto q = MeasureRecoveryQuality(mCurrentFrame, recovery_map,
                        has_reference ? &mCornerLastGoodPose : nullptr, &recovery_pools[candidate], &residuals);
                if (q.unique > best_residual_count) { best_residual_count = q.unique; best_residuals = residuals; }
                const double reference_age = has_reference ? mCurrentFrame.mTimeStamp-mCornerLastGoodTime : -1;
                const double seed_age = mCurrentFrame.mTimeStamp-vpCandidateKFs[candidate]->mTimeStamp;
                if (CornerShortRecoveryEnabled() && expand_enabled && !mbOnlyTracking && mState == RECENTLY_LOST &&
                    seed_age >= 0 && seed_age <= kRecoveryPoolRecentSeconds && q.unique < 50 &&
                    AcceptShortAnchor(q, reference_age) && q.unique > short_seed_unique)
                {
                    short_seed_unique = q.unique; short_seed = vpCandidateKFs[candidate]; short_seed_quality = q;
                    short_seed_pose = mCurrentFrame.GetPose(); short_seed_points = mCurrentFrame.mvpMapPoints;
                    short_seed_outliers = mCurrentFrame.mvbOutlier;
                }
                auto& lines = quality_lines[candidate];
                lines.clear();
                AppendCornerEvent(lines, "SLAM_RELOC_QUALITY frame=%lu ts=%.9f seed=%lu stage=%s optimized_inliers=%d associations=%d unique=%d duplicates=%d invalid=%d cells=%d extra_inliers=%d rmse_px=%.6f median_px=%.6f p95_px=%.6f rotation_deg=%.6f translation_depth=%.9f reference_age_s=%.6f decision=diagnostic_only normalized_rmse=%.9f normalized_p95=%.9f max_chi2=%.9f chi2_over=%d quality_policy=octave_chi2",
                        mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, vpCandidateKFs[candidate]->mnId, stage,
                        good, q.associations, q.unique, q.duplicates, q.invalid, q.cells, q.extra,
                        q.rmse, q.median, q.p95, q.rotation_deg, q.translation_depth,
                        reference_age, q.normalized_rmse, q.normalized_p95, q.max_chi2, q.chi2_over);
            }
            return good;
        };
        auto expand_projection = [&](int candidate, const char* stage, int existing_good, int seed_added,
                float radius, int descriptor_limit) {
            auto& debug = solver_diagnostics[candidate];
            if (!expand_enabled || candidate >= kRecoveryPoolCandidateLimit || existing_good + seed_added >= 50) return 0;
            if (debug.expand_calls >= kRecoveryPoolMaxExpansions) { ++debug.expand_budget_skips; return 0; }
            auto& pool = recovery_pools[candidate];
            if (!pool.built)
            {
                pool = BuildRecoveryPool(vpCandidateKFs[candidate], recovery_map, recent_keyframes,
                        mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, candidate < 8 ? &recovery_lines : nullptr);
                debug.pool_points = static_cast<int>(pool.points.size());
                debug.extra_points = static_cast<int>(pool.extra_ids.size());
            }
            if (pool.points.size() < 50) { debug.pool_short = 1; return 0; }
            set<MapPoint*> found;
            for (MapPoint* point : mCurrentFrame.mvpMapPoints) if (point) found.insert(point);
            RecoveryProjectionDiagnostics projection;
            const auto began = std::chrono::steady_clock::now();
            const int added = matcher2.SearchRecoveryPool(mCurrentFrame, pool.points, found, radius, descriptor_limit, projection);
            ++debug.expand_calls;
            debug.expand_added_best = std::max(debug.expand_added_best, added);
            debug.expand_gate_best = std::max(debug.expand_gate_best, existing_good + seed_added + added);
            if (candidate < 8)
                AppendCornerEvent(recovery_lines, "SLAM_RELOC_EXPAND frame=%lu ts=%.9f seed=%lu stage=%s round=%d seed_good=%d seed_added=%d baseline_total=%d extra_added=%d total_matches=%d required=50 tested=%d seed_excluded=%d already_found=%d invalid=%d depth=%d bounds=%d distance=%d no_features=%d no_free=%d descriptor=%d orientation=%d matched=%d search_ms=%.3f",
                        mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, vpCandidateKFs[candidate]->mnId,
                        stage, debug.expand_calls, existing_good, seed_added, existing_good + seed_added,
                        added, existing_good + seed_added + added, projection.tested, projection.seed_excluded,
                        projection.already_found, projection.invalid, projection.depth, projection.bounds,
                        projection.distance, projection.no_features, projection.no_free, projection.descriptor,
                        projection.orientation, projection.matched,
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count());
            return added;
        };

        while (nCandidates > 0 && !bMatch)
        {
            for (int i = 0; i < nKFs; i++)
            {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                MLPnPsolver* pSolver = vpMLPnPsolvers[i].get();
                auto& solver_debug = solver_diagnostics[i];
                ++solver_debug.attempts;
                Eigen::Matrix4f eigTcw;
                const auto pnp_start = std::chrono::steady_clock::now();
                bool bTcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers, eigTcw);
                solver_debug.pnp_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-pnp_start).count();

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore)
                {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (bTcw)
                {
                    ++solver_debug.poses;
                    solver_debug.pnp_best = std::max(solver_debug.pnp_best, nInliers);
                    Sophus::SE3f Tcw(eigTcw);
                    mCurrentFrame.SetPose(Tcw);
                    // Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint*> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++)
                    {
                        if (vbInliers[j])
                        {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        }
                        else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = optimize_candidate(i, "initial");
                    solver_debug.initial_good_best = std::max(solver_debug.initial_good_best, nGood);
                    mFrameDiagnostics.reloc_good_best = std::max(mFrameDiagnostics.reloc_good_best, nGood);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint*>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50)
                    {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);
                        nadditional += expand_projection(i, "coarse", nGood, nadditional, 10, 100);
                        ++solver_debug.coarse_calls;
                        solver_debug.coarse_added_best = std::max(solver_debug.coarse_added_best, nadditional);
                        solver_debug.coarse_gate_best = std::max(solver_debug.coarse_gate_best, nadditional + nGood);

                        if (AllowRecoveryOptimization(nadditional+nGood, short_query, solver_debug.short_opt_calls))
                        {
                            nGood = optimize_candidate(i, "coarse");
                            if (recovery_pools[i].built)
                                solver_debug.extra_inliers_best = std::max(solver_debug.extra_inliers_best,
                                        CountRecoveryExtraInliers(mCurrentFrame, recovery_pools[i]));
                            solver_debug.coarse_good_best = std::max(solver_debug.coarse_good_best, nGood);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50)
                            {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);
                                nadditional += expand_projection(i, "fine", nGood, nadditional, 3, 64);
                                ++solver_debug.fine_calls;
                                solver_debug.fine_added_best = std::max(solver_debug.fine_added_best, nadditional);
                                solver_debug.fine_gate_best = std::max(solver_debug.fine_gate_best, nGood + nadditional);

                                // Final optimization
                                if (AllowRecoveryOptimization(nGood+nadditional, short_query, solver_debug.short_opt_calls))
                                {
                                    nGood = optimize_candidate(i, "fine");
                                    if (recovery_pools[i].built)
                                        solver_debug.extra_inliers_best = std::max(solver_debug.extra_inliers_best,
                                                CountRecoveryExtraInliers(mCurrentFrame, recovery_pools[i]));

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                            }
                        }
                    }


                    solver_debug.final_good_best = std::max(solver_debug.final_good_best, nGood);
                    mFrameDiagnostics.reloc_good_best = std::max(mFrameDiagnostics.reloc_good_best, nGood);
                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50)
                    {
                        solver_debug.recovered = 1;
                        if (expand_enabled)
                            LogRecoveryVerification(mCurrentFrame, vpCandidateKFs[i], recovery_pools[i], nGood, recovery_lines);
                        bMatch = true;
                        break;
                    }
                }
            }
        }

        WriteCornerEvents("SlamReloc.txt", recovery_lines);
        vector<string> solver_lines;
        if (best_residual_count >= 0)
            AppendRecoveryResiduals(solver_lines, "reloc", mCurrentFrame, best_residual_count, best_residuals);
        for (int i = 0; i < nKFs && i < 8; ++i)
        {
            const auto& d = solver_diagnostics[i];
            for (const auto& line : quality_lines[i]) solver_lines.push_back(line);
            if (vpMLPnPsolvers[i])
            {
                const auto& p = vpMLPnPsolvers[i]->GetDiagnostics();
                AppendCornerEvent(solver_lines, "SLAM_PNP frame=%lu ts=%.9f seed=%lu calls=%d samples=%d last_samples=%d limit=%d inputs=%d invalid_inputs=%d duplicate_inputs=%d invalid_poses=%d refine_calls=%d refined=%d depth_rejected=%d exhausted=%d reason=%s batch_limit=5 sample_size=6 pnp_ms=%.3f optimize_ms=%.3f",
                        mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, vpCandidateKFs[i]->mnId,
                        p.calls, p.samples, p.last_samples, p.limit, p.inputs, p.invalid_inputs, p.duplicate_inputs,
                        p.invalid_poses, p.refine_calls, p.refined, p.depth_rejected, p.exhausted ? 1 : 0,
                        p.reason, d.pnp_ms, d.optimize_ms);
            }
            AppendCornerEvent(solver_lines, "SLAM_RELOC_SOLVER frame=%lu ts=%.9f kf=%lu eligible=%d pnp_attempts=%d pnp_poses=%d pnp_inliers_best=%d initial_good_best=%d coarse_calls=%d coarse_added_best=%d coarse_gate_best=%d coarse_good_best=%d fine_calls=%d fine_added_best=%d fine_gate_best=%d final_good_best=%d required_good=50 recovered=%d pool_points=%d extra_points=%d pool_short=%d expand_calls=%d expand_budget_skips=%d expand_added_best=%d expand_gate_best=%d extra_inliers_best=%d short_opt_calls=%d short_opt_budget=%d",
                    mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, vpCandidateKFs[i]->mnId,
                    d.eligible, d.attempts, d.poses, d.pnp_best, d.initial_good_best,
                    d.coarse_calls, d.coarse_added_best, d.coarse_gate_best, d.coarse_good_best,
                    d.fine_calls, d.fine_added_best, d.fine_gate_best, d.final_good_best, d.recovered,
                    d.pool_points, d.extra_points, d.pool_short, d.expand_calls, d.expand_budget_skips,
                    d.expand_added_best, d.expand_gate_best, d.extra_inliers_best, d.short_opt_calls, kShortOptimizationBudget);
        }
        WriteCornerEvents("SlamReloc.txt", solver_lines);
        WriteCornerEvent("SlamReloc.txt", "SLAM_RELOC_RESULT frame=%lu ts=%.9f candidates=%d pnp_candidates=%d best_bow=%d best_good=%d recovered=%d",
                mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, nKFs, debug_pnp_candidates,
                mFrameDiagnostics.reloc_bow_best, mFrameDiagnostics.reloc_good_best, bMatch ? 1 : 0);
        if (!bMatch)
        {
            if (short_seed)
            {
                mCurrentFrame.SetPose(short_seed_pose);
                mCurrentFrame.mvpMapPoints = short_seed_points;
                mCurrentFrame.mvbOutlier = short_seed_outliers;
                DeduplicateRecoveryFrame(mCurrentFrame);
                for (size_t j = 0; j < mCurrentFrame.mvpMapPoints.size(); ++j)
                    if (mCurrentFrame.mvbOutlier[j]) mCurrentFrame.mvpMapPoints[j] = nullptr;
                mCornerShortFrame = make_unique<Frame>();
                mCornerShortFrame->copyFrom(mCurrentFrame);
                mCornerShortFrame->mpReferenceKF = short_seed;
                mCornerShortChain.Reset();
                mCornerShortChain.Observe(recovery_map->GetId(), mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, short_seed_unique, 0);
                mFrameDiagnostics.short_pending = 1;
                WriteCornerEvent("SlamReloc.txt", "SLAM_SHORT_SEED frame=%lu ts=%.9f map=%lu seed=%lu unique=%d cells=%d invalid=%d normalized_rmse=%.9f max_chi2=%.9f chi2_over=%d rotation_deg=%.6f translation_depth=%.9f age_s=%.6f streak=1 confirmed=0",
                        mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, recovery_map->GetId(), short_seed->mnId,
                        short_seed_unique, short_seed_quality.cells, short_seed_quality.invalid, short_seed_quality.normalized_rmse,
                        short_seed_quality.max_chi2, short_seed_quality.chi2_over, short_seed_quality.rotation_deg,
                        short_seed_quality.translation_depth, mCurrentFrame.mTimeStamp-mCornerLastGoodTime);
            }
            return false;
        }
        else
        {
            ClearShortMapRecovery();
            mnLastRelocFrameId = mCurrentFrame.mnId;
            cout << "Relocalized!!" << endl;
            return true;
        }

    }

    void Tracking::Reset(bool bLocMap)
    {
        ClearShortMapRecovery();
        mCornerLastGoodTime = -1;
        mCornerLastGoodMap = nullptr;
        mCornerFallbackStreak = 0;
        __android_log_print(ANDROID_LOG_INFO, "SLAM_DIAG", "SLAM_EVENT type=reset_all");
        Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

        // Reset Local Mapping
        if (!bLocMap)
        {
            Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
            mpLocalMapper->RequestReset();
            Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
        }


        // Reset Loop Closing
        Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
        mpLoopClosing->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear BoW Database
        Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
        mpKeyFrameDB->clear();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear Map (this erase MapPoints and KeyFrames)
        mpAtlas->clearAtlas();
        mpAtlas->CreateNewMap();
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
            mpAtlas->SetInertialSensor();
        mnInitialFrameId = 0;

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        mbReadyToInitializate = false;
        mbSetInit = false;

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();
        //mCurrentFrame = Frame();
        mCurrentFrame.reset();
        mnLastRelocFrameId = 0;
        //mLastFrame = Frame();
        mLastFrame.reset();
        mpReferenceKF = nullptr;
        mpLastKeyFrame = nullptr;
        mvIniMatches.clear();

        Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
    }

    void Tracking::ResetActiveMap(bool bLocMap)
    {
        ClearShortMapRecovery();
        mCornerLastGoodTime = -1;
        mCornerLastGoodMap = nullptr;
        mCornerFallbackStreak = 0;
        __android_log_print(ANDROID_LOG_INFO, "SLAM_DIAG", "SLAM_EVENT type=reset_active_map");
        Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);

        Map* pMap = mpAtlas->GetCurrentMap();

        if (!bLocMap)
        {
            Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
            mpLocalMapper->RequestResetActiveMap(pMap);
            Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
        }

        // Reset Loop Closing
        Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
        mpLoopClosing->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear BoW Database
        Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
        mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear Map (this erase MapPoints and KeyFrames)
        mpAtlas->clearMap();


        //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
        //Frame::nNextId = mnLastInitFrameId;
        mnLastInitFrameId = Frame::nNextId;
        //mnLastRelocFrameId = mnLastInitFrameId;
        mState = NO_IMAGES_YET; //NOT_INITIALIZED;

        mbReadyToInitializate = false;

        list<bool> lbLost;
        // lbLost.reserve(mlbLost.size());
        unsigned int index = mnFirstFrameId;
        cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
        for (Map* pMap : mpAtlas->GetAllMaps())
        {
            if (pMap->GetAllKeyFrames().size() > 0)
            {
                if (index > pMap->GetLowerKFID())
                    index = pMap->GetLowerKFID();
            }
        }

        //cout << "First Frame id: " << index << endl;
        int num_lost = 0;
        cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

        for (bool ilbL : mlbLost)
        {
            if (index < mnInitialFrameId)
            {
                lbLost.push_back(ilbL);
            }
            else
            {
                lbLost.push_back(true);
                num_lost += 1;
            }

            index++;
        }
        cout << num_lost << " Frames set to lost" << endl;

        mlbLost = lbLost;

        mnInitialFrameId = mCurrentFrame.mnId;
        mnLastRelocFrameId = mCurrentFrame.mnId;

        mCurrentFrame.reset();
        mLastFrame.reset();
        mpReferenceKF = nullptr;
        mpLastKeyFrame = nullptr;
        mvIniMatches.clear();

        mbVelocity = false;

        Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
    }

    vector<MapPoint*> Tracking::GetLocalMapMPS()
    {
        return mvpLocalMapPoints;
    }

    void Tracking::ChangeCalibration(const string& strSettingPath)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        mK_.setIdentity();
        mK_(0, 0) = fx;
        mK_(1, 1) = fy;
        mK_(0, 2) = cx;
        mK_(1, 2) = cy;

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool& flag)
    {
        mbOnlyTracking = flag;
    }

    void Tracking::UpdateFrameIMU(const float s, const IMU::Bias& b, KeyFrame* pCurrentKeyFrame)
    {
        Map* pMap = pCurrentKeyFrame->GetMap();
        unsigned int index = mnFirstFrameId;
        list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
        list<bool>::iterator lbL = mlbLost.begin();
        for (auto lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end();lit != lend;lit++, lRit++, lbL++)
        {
            if (*lbL)
                continue;

            KeyFrame* pKF = *lRit;

            while (pKF->isBad())
            {
                pKF = pKF->GetParent();
            }

            if (pKF->GetMap() == pMap)
            {
                (*lit).translation() *= s;
            }
        }

        mLastBias = b;

        mpLastKeyFrame = pCurrentKeyFrame;

        mLastFrame.SetNewBias(mLastBias);
        mCurrentFrame.SetNewBias(mLastBias);

        while (!mCurrentFrame.imuIsPreintegrated())
        {
            usleep(500);
        }


        if (mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
        {
            mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                mLastFrame.mpLastKeyFrame->GetImuPosition(),
                mLastFrame.mpLastKeyFrame->GetVelocity());
        }
        else
        {
            const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
            const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
            const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
            const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
            float t12 = mLastFrame.mpImuPreintegrated->dT;

            mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                Vwb1 + Gz * t12 + Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }

        if (mCurrentFrame.mpImuPreintegrated)
        {
            const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

            const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
            const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
            const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
            float t12 = mCurrentFrame.mpImuPreintegrated->dT;

            mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                Vwb1 + Gz * t12 + Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }

        mnFirstImuFrameId = mCurrentFrame.mnId;
    }

    void Tracking::NewDataset()
    {
        mnNumDataset++;
    }

    int Tracking::GetNumberDataset()
    {
        return mnNumDataset;
    }

    int Tracking::GetMatchesInliers()
    {
        return mnMatchesInliers;
    }

    void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
    {
        mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
        //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
    }

    void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
    {
        mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
        if (!strNameFile_kf.empty())
            mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
    }

    float Tracking::GetImageScale()
    {
        return mImageScale;
    }

#ifdef REGISTER_LOOP
    void Tracking::RequestStop()
    {
        unique_lock<mutex> lock(mMutexStop);
        mbStopRequested = true;
    }

    bool Tracking::Stop()
    {
        unique_lock<mutex> lock(mMutexStop);
        if (mbStopRequested && !mbNotStop)
        {
            mbStopped = true;
            cout << "Tracking STOP" << endl;
            return true;
        }

        return false;
    }

    bool Tracking::stopRequested()
    {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopRequested;
    }

    bool Tracking::isStopped()
    {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopped;
    }

    void Tracking::Release()
    {
        unique_lock<mutex> lock(mMutexStop);
        mbStopped = false;
        mbStopRequested = false;
    }
#endif

} //namespace ORB_SLAM
