#include "core/SlamScene.h"

#include "core/App.h"

#include "render/ImageTexture.h"
#include "render/Shader.h"
#include "render/Plane2D.h"

#include "utils/Log.h"
#include "utils/AssetManager.h"
#include "utils/ImageLoader.h"
#include "utils/GnssGui.h"
#include <cstdio>
#include <sstream>

namespace android_slam
{

    void SlamScene::init()
    {
        m_need_update_image = true;
        tracking_res = TrackingResult{};
        m_tracking_result = TrackingResult{};
        m_comm_has_new_data = false;
        m_file_manager = std::make_unique<AssetManager>();

        Image first_image;
        if(!m_from_datasets) {
            // Camera image converter.
            m_image_pool = std::make_unique<ImagePool>(
                    k_sensor_camera_width, k_sensor_camera_height, "shader/yuv2rgb.vert",
                    "shader/yuv2rgb.frag"
            );
            first_image = m_image_pool->getImage();

            //IMU data converter
            m_imu_pool = std::make_unique<SensorIMU>();
        }
        else{
            DatasetImage = ImageLoader::loadDatasetImage("dataset/imgdata_indoor3_15hz_test3");
            first_image = DatasetImage[0];
            //DatasetImu = ImageLoader::loadDatasetImu("data10/imu.txt");
        }

        m_slam_renderer = std::make_unique<SlamRenderer>(
        k_fps_camera_fov, m_app_ref.getWindow().getAspectRatio(), k_fps_camera_z_min, k_fps_camera_z_max
        );

        // Create slam kernel.
        {
            DEBUG_INFO("[Android Slam App Info] Starts to create slam kernel.");

            AAsset* asset = AAssetManager_open(AssetManager::get(), "vocabulary/ORBVoc.txt", AASSET_MODE_BUFFER);
            assert(asset && "[Android Slam App Info] Failed to open ORBVoc.txt.");

            size_t size = AAsset_getLength(asset);
            auto buffer = (const char*) AAsset_getBuffer(asset);
            std::string voc_buffer(buffer, buffer + size);
            AAsset_close(asset);

            m_slam_kernel = std::make_unique<SlamKernel>(
            k_sensor_camera_width, k_sensor_camera_height, std::move(voc_buffer), first_image.time_stamp
            );

            DEBUG_INFO("[Android Slam App Info] Creates slam kernel successfully.");
        }

        // 读取真实世界锚点坐标（/storage/emulated/0/Documents/AnchorPos.txt，每行 x y z）
        {
            std::string anchor_str = AssetManager::readFile("/storage/emulated/0/Documents/AnchorPos.txt");
            std::istringstream iss(anchor_str);
            std::vector<Eigen::Vector3f> world_anchors;
            float ax = 0.0f, ay = 0.0f, az = 0.0f;
            while (iss >> ax >> ay >> az)
                world_anchors.emplace_back(ax, ay, az);
            if (!world_anchors.empty())
            {
                m_slam_kernel->setWorldAnchors(world_anchors);
                DEBUG_INFO("[Android Slam App Info] Loaded %zu world anchors.", world_anchors.size());
            }
            else
                DEBUG_INFO("[Android Slam App Info] AnchorPos.txt missing or empty.");
        }

        m_slam_has_new_data = false; // Whether the main thread generates new image for slam thread.
        m_is_running_slam = true;     // Whether the slam scene is running, also indicates the slam thread is running.

        // 每次开启新实验时覆盖旧日志，避免文件随实验次数持续膨胀。
        m_file_manager->clearFile("/storage/emulated/0/Documents/SlamLog.txt");
        AssetManager::appendLog("========== SLAM session start ==========");
        AssetManager::appendLog("debug_version=corner_diag_v6 mode=MONOCULAR imu_fused=0 unique_images_only=1 recovery_policy=expanded_projection_pool reloc_score_policy=current_query_only pnp_policy=bounded_refine_applied fallback_policy=one_wide_local_retry quality_policy=octave_chi2");
        m_log_frame_index = 0;
        m_debug_turn_index = 0;

        // Create slam thread.
        m_slam_thread = std::make_unique<std::thread>(
        [this]()
        {
            //std::vector<Image> images;
            //std::vector<ImuPoint> imu_points;
            while (m_is_running_slam)
            {
                if (m_slam_has_new_data)
                {
                    // Acquire new images.
                    std::vector<Image> images;
                    {
                        std::unique_lock<std::mutex> lock(m_image_mutex);
                        images = std::move(m_images);
                        //imu_points = std::move(m_imu_points);
                    }

                    // Call slam tracking function.
                    //auto res = m_slam_kernel->handleData(images, imu_points);
                    auto res = m_slam_kernel->handleData(images, {});

                    // Synchronize tracking result to main thread, move the data because this thread doesn't need it.
                    {
                        std::unique_lock<std::mutex> lock(m_tracking_res_mutex);
                        m_tracking_result = std::move(res);
                    }
                    // Publish the result before allowing the producer to submit
                    // another image. This keeps diagnostic results in order.
                    m_slam_has_new_data = false;

                }
                else
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        );
        // Create communication thread.
        if(m_open_comm) {
            std::string start_str = AssetManager::readFile("/storage/emulated/0/Documents/StartPos.txt");
            m_comm = std::make_unique<Communicator>();
            m_comm->getStartPos(start_str);
            m_comm_thread = std::make_unique<std::thread>(
                    [this]() {
                        std::string port = AssetManager::readFile("/storage/emulated/0/Documents/Port.txt");
                        std::string ip = AssetManager::readFile("/storage/emulated/0/Documents/IP.txt");
                        {
                            char comm_buf[256];
                            snprintf(comm_buf, sizeof(comm_buf), "[comm] thread start ip='%s' port='%s'", ip.c_str(), port.c_str());
                            DEBUG_INFO("[Android Slam App Info] %s", comm_buf);
                            AssetManager::appendLog(comm_buf);
                        }
                        int send_ok = 0;
                        int send_fail = 0;
                        //DEBUG_INFO("[Android Slam App Info] port: %s", port.c_str());
                        //DEBUG_INFO("[Android Slam App Info] ip: %s", ip.c_str());
                        while (m_is_running_slam) {
                            TrackingResult tracking_res_comm;
                            if (m_comm_has_new_data) {
                                {
                                    std::unique_lock<std::mutex> lock(m_comm_mutex);
                                    tracking_res_comm = m_comm_result;
                                }
                                if (tracking_res_comm.tracking_status == "OK") {
                                    /*
                                    m_comm->Run(tracking_res_comm.trajectory.back().x,
                                             tracking_res_comm.trajectory.back().y,
                                             tracking_res_comm.trajectory.back().z,
                                                ip, port);
                                     */
                                    if (m_comm->Run(tracking_res_comm, ip, port))
                                        ++send_ok;
                                    else
                                    {
                                        ++send_fail;
                                        if (send_fail == 1)
                                        {
                                            char comm_buf[256];
                                            snprintf(comm_buf, sizeof(comm_buf), "[comm] first send FAILED ip='%s' port='%s'", ip.c_str(), port.c_str());
                                            DEBUG_INFO("[Android Slam App Info] %s", comm_buf);
                                            AssetManager::appendLog(comm_buf);
                                        }
                                    }
                                    if ((send_ok + send_fail) % 200 == 0)
                                    {
                                        char comm_buf[256];
                                        snprintf(comm_buf, sizeof(comm_buf), "[comm] sent=%d failed=%d", send_ok, send_fail);
                                        DEBUG_INFO("[Android Slam App Info] %s", comm_buf);
                                        AssetManager::appendLog(comm_buf);
                                    }
                                }
                                m_comm_has_new_data = false;
                            }

                        }
                    }
            );
        }
    }

    void SlamScene::exit()
    {
        m_is_running_slam = false;
        m_need_update_image = false;

        if (m_slam_thread && m_slam_thread->joinable())
            m_slam_thread->join();
        m_slam_thread.reset(nullptr);

        if (m_comm_thread && m_comm_thread->joinable())
            m_comm_thread->join();
        m_comm_thread.reset(nullptr);

        m_comm.reset(nullptr);
        m_slam_kernel.reset(nullptr);
        m_slam_renderer.reset(nullptr);
        m_image_pool.reset(nullptr);
        m_imu_pool.reset(nullptr);

        AssetManager::appendLog("========== SLAM session end ==========");
    }

    void SlamScene::update(float dt)
    {
        // Update image data if not paused and slam thread needs new image. Render data will be set at the same time.
        if (m_need_update_image && !m_slam_has_new_data)
        {
            bool has_result = false;
            {
                std::unique_lock<std::mutex> lock(m_tracking_res_mutex);
                if (!m_tracking_result.tracking_status.empty())
                {
                    tracking_res = std::move(m_tracking_result);
                    m_tracking_result = TrackingResult{};
                    has_result = true;
                }
            }

            // Slam handling.
            std::vector<Image> images;
            //std::vector<ImuPoint> imus;
            if(!m_from_datasets) {
                images.push_back(m_image_pool->getImage());
                if (!images[0].data.empty())
                {
                    std::unique_lock<std::mutex> lock(m_image_mutex);
                    m_images = images;
                    //m_imu_points = m_imu_pool->getImuData();
                    m_slam_has_new_data = true;
                }
            }
            else{// outline dataset
                {
                    images.push_back(DatasetImage[m_datasets_num]);
                    //imus = ImageLoader::getImuDataset(DatasetImu,
                    //                                  DatasetImage[m_datasets_num].time_stamp,
                    //                                  DatasetImage[m_datasets_num-1].time_stamp);
                    m_datasets_num++;
                    usleep(20000);
                    std::unique_lock<std::mutex> lock(m_image_mutex);
                    m_images = images;
                    //m_imu_points = imus;
                }
                m_slam_has_new_data = true;
            }
            // Set slam data.
            if (!images[0].data.empty())
                m_slam_renderer->setImage(k_sensor_camera_width, k_sensor_camera_height, images[0]);

            // 实验日志：打印并写入 /storage/emulated/0/Documents/SlamLog.txt
            if (has_result)
            {
                m_slam_renderer->setData(tracking_res);
                m_app_ref.m_last_process_delta_time = tracking_res.processing_delta_time;
                float px = 0.0f, py = 0.0f, pz = 0.0f;
                if (!tracking_res.trajectory.empty())
                {
                    px = tracking_res.trajectory.back().x;
                    py = tracking_res.trajectory.back().y;
                    pz = tracking_res.trajectory.back().z;
                }
                char line[256];
                snprintf(line, sizeof(line),
                         "frame=%d ts=%lld state=%s dt=%.4f scale=%.4f pos=(%.4f,%.4f,%.4f) traj=%lu mp=%lu valid=%d",
                         m_log_frame_index++,
                         (long long) tracking_res.time_stamp,
                         tracking_res.tracking_status.c_str(),
                         tracking_res.processing_delta_time,
                         m_slam_kernel->getSimScale(),
                         px, py, pz,
                         (unsigned long) tracking_res.trajectory.size(),
                         (unsigned long) tracking_res.map_points.size(), tracking_res.pose_valid ? 1 : 0);
                DEBUG_INFO("[Android Slam App Info] %s", line);
                AssetManager::appendLog(line);
            }

            //m_file_manager->getTraj(tracking_res);

            //set comm data
            if(has_result && !m_comm_has_new_data)
            {
                {
                    std::unique_lock<std::mutex> lock(m_comm_mutex);
                    //m_comm_result = std::move(tracking_res);
                    m_comm_result = tracking_res;
                }
                m_comm_has_new_data = true;
            }
        }

        // Draw trajectory.
        m_slam_renderer->clearColor();

        int32_t screen_width = m_app_ref.getWindow().getWidth();
        int32_t screen_height = m_app_ref.getWindow().getHeight();

        m_slam_renderer->drawMapPoints(0, 0, screen_width, screen_height);
        m_slam_renderer->drawKeyFrames(0, 0, screen_width, screen_height);

        int32_t img_width = screen_height * 4 / 7;
        int32_t img_height = screen_height * 3 / 7;
        m_slam_renderer->drawImage(0, 0, img_width, img_height); // Aspect ratio: 4 : 3

        m_slam_renderer->drawTotalTrajectory(0, img_height, img_width, screen_height - img_height); // Aspect ratio: 1 : 1
    }

    void SlamScene::drawGui(float dt)
    {
        drawGnssPanel();
        if (ImGui::TreeNode(u8"SLAM控制"))
        {
            ImGui::Text("Diagnostics: corner_diag_v6");
            //auto pos = m_slam_renderer->temp_last_kf_position;
            //ImGui::Text("Position: (%f, %f, %f)", pos.x, pos.y, pos.z);

            if (ImGui::Button(m_need_update_image ? u8"暂停" : u8"继续"))
            {
                m_need_update_image = !m_need_update_image;
            }

            if (ImGui::Button(u8"转弯开始"))
            {
                char marker[160];
                snprintf(marker, sizeof(marker), "SLAM_MARK type=turn_start turn=%d ts=%lld state=%s",
                        ++m_debug_turn_index, (long long)tracking_res.time_stamp,
                        tracking_res.tracking_status.c_str());
                AssetManager::appendLog(marker);
                __android_log_print(ANDROID_LOG_INFO, "SLAM_DIAG", "%s", marker);
            }
            ImGui::SameLine();
            if (ImGui::Button(u8"转弯结束"))
            {
                char marker[160];
                snprintf(marker, sizeof(marker), "SLAM_MARK type=turn_end turn=%d ts=%lld state=%s",
                        m_debug_turn_index, (long long)tracking_res.time_stamp,
                        tracking_res.tracking_status.c_str());
                AssetManager::appendLog(marker);
                __android_log_print(ANDROID_LOG_INFO, "SLAM_DIAG", "%s", marker);
            }

            //if (ImGui::Button(u8"重置"))
            //{
            //    m_slam_kernel->reset();
            //}

            if (ImGui::Button(u8"退出"))
            {
                m_app_ref.setActiveScene("Init");
            }

            {
                const bool aligned = m_slam_kernel->isAligned();

                if (ImGui::Button(u8"记录锚点"))
                {
                    if (!tracking_res.trajectory.empty())
                    {
                        const auto& p = tracking_res.trajectory.back();
                        m_slam_kernel->addSlamAnchor(p.x, p.y, p.z);
                        char buf[128];
                        snprintf(buf, sizeof(buf), "anchor=%d pos=(%.3f,%.3f,%.3f)",
                                 m_slam_kernel->getSlamAnchorCount(), p.x, p.y, p.z);
                        DEBUG_INFO("[Android Slam App Info] %s", buf);
                        AssetManager::appendLog(buf);
                    }
                }
                ImGui::SameLine();
                ImGui::Text(u8"锚点 %d/%d",
                            m_slam_kernel->getSlamAnchorCount(),
                            m_slam_kernel->getWorldAnchorCount());

                if (ImGui::Button(u8"清除锚点"))
                {
                    m_slam_kernel->clearAnchors();
                    DEBUG_INFO("[Android Slam App Info] anchors cleared.");
                    AssetManager::appendLog("anchors cleared");
                }

                if (ImGui::Button(aligned ? u8"还原坐标" : u8"对齐坐标"))
                {
                    if (aligned)
                    {
                        m_slam_kernel->resetAlignment();
                        DEBUG_INFO("[Android Slam App Info] alignment reset.");
                        AssetManager::appendLog("alignment reset");
                    }
                    else
                    {
                        if (m_slam_kernel->computeAlignment())
                        {
                            char buf[128];
                            snprintf(buf, sizeof(buf), "alignment ok scale=%.4f",
                                     m_slam_kernel->getSimScale());
                            DEBUG_INFO("[Android Slam App Info] %s", buf);
                            AssetManager::appendLog(buf);
                        }
                        else
                        {
                            DEBUG_INFO("[Android Slam App Info] alignment failed (need >=3 non-collinear anchors).");
                            AssetManager::appendLog("alignment failed (need >=3 non-collinear anchors)");
                        }
                    }
                }
            }
            if (ImGui::Button(u8"保存轨迹"))
            {
                m_file_manager->getTraj(tracking_res);
                m_file_manager->saveTraj();
            }

            char* save_mark;
            switch(markNum) {
                case 1: save_mark = "保存点1"; break;
                case 2: save_mark = "保存点2"; break;
                case 3: save_mark = "保存点3"; break;
                case 4: save_mark = "保存点4"; break;
                case 5: save_mark = "保存点5"; break;
                case 6: save_mark = "保存点6"; break;
                case 7: save_mark = "保存点7"; break;
                case 8: save_mark = "保存点8"; break;
                default: save_mark = "保存点9+"; break;
            }
            if (ImGui::Button(save_mark))
            {
                if(markNum == 1)
                    m_file_manager->clearFile("/storage/emulated/0/Documents/MarkPos.txt");
                m_file_manager->getTraj(tracking_res);
                m_file_manager->saveCurPos();
                markNum++;
            }


            ImGui::TreePop();
        }

        if (ImGui::TreeNode(u8"绘制选项"))
        {
            if (ImGui::TreeNode(u8"地图点"))
            {
                if (ImGui::Button(m_slam_renderer->m_show_mappoints ? u8"隐藏" : u8"显示"))
                {
                    m_slam_renderer->m_show_mappoints = !m_slam_renderer->m_show_mappoints;
                }

                ImGui::SliderFloat(u8"地图点大小", &m_slam_renderer->m_point_size, 1.0f, 5.0f);

                ImGui::ColorEdit3(u8"地图点颜色", reinterpret_cast<float*>(&m_slam_renderer->m_mp_color));

                ImGui::TreePop();
            }

            if (ImGui::TreeNode(u8"相机轨迹"))
            {
                if (ImGui::Button(m_slam_renderer->m_show_keyframes ? u8"隐藏" : u8"显示"))
                {
                    m_slam_renderer->m_show_keyframes = !m_slam_renderer->m_show_keyframes;
                }

                ImGui::SliderFloat(u8"轨迹线粗细", &m_slam_renderer->m_line_width, 1.0f, 5.0f);

                ImGui::ColorEdit3(u8"轨迹线颜色", reinterpret_cast<float*>(&m_slam_renderer->m_kf_color));

                ImGui::TreePop();
            }

            if (ImGui::TreeNode(u8"其他"))
            {
                if (ImGui::Button(m_slam_renderer->m_show_image ? u8"隐藏相机图像" : u8"显示相机图像"))
                {
                    m_slam_renderer->m_show_image = !m_slam_renderer->m_show_image;
                }

                if (ImGui::Button(m_slam_renderer->m_show_total_trajectory ? u8"隐藏全局轨迹" : u8"显示全局轨迹"))
                {
                    m_slam_renderer->m_show_total_trajectory = !m_slam_renderer->m_show_total_trajectory;
                }

                ImGui::ColorEdit3(u8"背景颜色", reinterpret_cast<float*>(&m_slam_renderer->m_clear_color));

                ImGui::TreePop();
            }

            ImGui::TreePop();
        }
    }

}
