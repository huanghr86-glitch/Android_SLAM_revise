#include "sensor/SensorCamera.h"
#include <string>
#include <thread>
#include <cassert>

#include "utils/Log.h"
#include "sensor/SensorGnss.h"
#include <time.h>
#include <cstdio>

namespace android_slam
{

    SensorCamera::SensorCamera(int32_t width, int32_t height, int32_t format, uint64_t usage)
        : m_img_reader{ nullptr, AImageReader_delete }
        , m_camera_manager{ nullptr, ACameraManager_delete }
        , m_camera_id_list{ nullptr, ACameraManager_deleteCameraIdList }
        , m_camera_device{ nullptr, ACameraDevice_close }
        , m_capture_session_output_container{ nullptr, ACaptureSessionOutputContainer_free }
        , m_image_reader_output{ nullptr, ACaptureSessionOutput_free }
        , m_capture_session{ nullptr, ACameraCaptureSession_close }
        , m_capture_request{nullptr, ACaptureRequest_free }
        , m_camera_output_target{ nullptr, ACameraOutputTarget_free }
        , m_camera_device_state_callbacks{}
        , m_capture_session_state_callbacks{}
    {
        media_status_t ms{};

        // Create images, and allocate image buffers.
        {
            for(size_t i = 0; i < k_max_image_count; ++i)
            {
                m_images.emplace_back(nullptr, AImage_delete);
            }
            m_image_buffers.resize(k_max_image_count, nullptr);
        }

        // Create ImageReader.
        {
            decltype(m_img_reader.get()) ptr;
            ms = AImageReader_newWithUsage(width, height, format, usage, k_max_image_count + 2, &ptr);
            assert((ms == AMEDIA_OK) && "[Android Slam Sensor Info] Failed to create ImageReader.");

            m_img_reader.reset(ptr);
            assert(m_img_reader);
        }

        //Create ImageReader's window.
        {
            ms = AImageReader_getWindow(m_img_reader.get(), &m_img_reader_window);
            assert((ms == AMEDIA_OK) && "[Android Slam Sensor Info] Failed to create ImageReader's Native Window.");
            assert(m_img_reader_window && "[Android Slam Sensor Info] ImageReader's Native Window is NULL.");
        }


        camera_status_t cs{};

        // Create camera manager;
        {
            m_camera_manager.reset(ACameraManager_create());
            assert(m_camera_manager && "[Android Slam Sensor Info] Failed to create camera manager.");
        }

        // Enumerate cameras;
        {
            decltype(m_camera_id_list.get()) ptr;
            cs = ACameraManager_getCameraIdList(m_camera_manager.get(), &ptr);
            assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to get camera id list.");

            m_camera_id_list.reset(ptr);
            assert(m_camera_id_list && "[Android Slam Sensor Info] CameraIdList is NULL.");
            assert((m_camera_id_list->numCameras > 0) && "[Android Slam Sensor Info] No available camera.");
        }

        // Choose one back facing camera
        std::string active_camera_id;
        {
            for (int32_t i = 0; i < m_camera_id_list->numCameras; ++i)
            {
                const char* id = m_camera_id_list->cameraIds[i];

                ACameraMetadata* ptr;
                cs = ACameraManager_getCameraCharacteristics(m_camera_manager.get(), id, &ptr);
                assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to get camera characteristics.");

                // Using RAII feature to delete ptr memory.
                CameraMetadata metadata{ ptr, ACameraMetadata_free };
                assert(metadata && "[Android Slam Sensor Info] Metadata is NULL.");

                ACameraMetadata_const_entry lens_facing{};
                cs = ACameraMetadata_getConstEntry(metadata.get(), ACAMERA_LENS_FACING, &lens_facing);
                assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to get camera lens info.");
                auto facing = static_cast<acamera_metadata_enum_android_lens_facing_t>(lens_facing.data.u8[0]);

                if(facing == ACAMERA_LENS_FACING_BACK)
                {
                    active_camera_id = id;
                    ACameraMetadata_const_entry timestamp_source{};
                    const auto timestamp_status = ACameraMetadata_getConstEntry(
                            metadata.get(), ACAMERA_SENSOR_INFO_TIMESTAMP_SOURCE, &timestamp_source);
                    const int source = timestamp_status == ACAMERA_OK && timestamp_source.count > 0
                            ? timestamp_source.data.u8[0] : 0;
                    timespec now{};
                    clock_gettime(CLOCK_BOOTTIME, &now);
                    SensorGnss::instance().setCameraTimestampSource(source,
                            static_cast<int64_t>(now.tv_sec) * 1000000000LL + now.tv_nsec);
                    DEBUG_INFO("[Android Slam Camera Info] timestamp source=%d (1=boot clock, 0=unknown)", source);
                    __android_log_print(ANDROID_LOG_INFO, "SLAM_DIAG",
                            "SLAM_CAMERA_CONFIG id=%s width=%d height=%d timestamp_source=%d template=PREVIEW",
                            id, width, height, source);
                    break;
                }
            }

            assert((!active_camera_id.empty()) && "[Android Slam Sensor Info] Failed to get an active camera.");
        }

        // Create camera device.
        {
            m_camera_device_state_callbacks.context = this;
            m_camera_device_state_callbacks.onDisconnected = [](void* context, ACameraDevice* device)
            {
                DEBUG_INFO("[Android Slam Camera Info] Camera(ID: %s) is disconnected.", ACameraDevice_getId(device));
            };
            m_camera_device_state_callbacks.onError = [](void* context, ACameraDevice* device, int error)
            {
                DEBUG_INFO("[Android Slam Camera Info] Camera(ID: %s)\nError Code: %d", ACameraDevice_getId(device), error);
            };

            decltype(m_camera_device.get()) ptr;
            cs = ACameraManager_openCamera(m_camera_manager.get(),
                                           active_camera_id.c_str(),
                                           &m_camera_device_state_callbacks,
                                           &ptr);
            while(cs == ACAMERA_ERROR_PERMISSION_DENIED)
            {
                static int count = 0;
                if(count == 10)
                {
                    assert(false && "[Android Slam Sensor Info] Run out of time when opening camera.");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                cs = ACameraManager_openCamera(m_camera_manager.get(),
                                               active_camera_id.c_str(),
                                               &m_camera_device_state_callbacks,
                                               &ptr);
                ++count;
            }
            assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to open active camera.");

            m_camera_device.reset(ptr);
            assert(m_camera_device && "[Android Slam Sensor Info] CameraDevice is NULL.");
        }

        // Create capture session output container.
        {
            decltype(m_capture_session_output_container.get()) ptr;
            cs = ACaptureSessionOutputContainer_create(&ptr);
            assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to create capture session output container.");

            m_capture_session_output_container.reset(ptr);
            assert(m_capture_session_output_container && "[Android Slam Sensor Info] Capture Session Output Container is NULL.");
        }

        //  Create capture session output, which is the place to save camera images.
        {
            decltype(m_image_reader_output.get()) ptr;
            cs = ACaptureSessionOutput_create(m_img_reader_window, &ptr);
            assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to create capture session output.");

            m_image_reader_output.reset(ptr);
            assert(m_image_reader_output && "[Android Slam Sensor Info] ImageReaderOutput is NULL.");
        }

        // Add image reader output to output container.
        {
            cs = ACaptureSessionOutputContainer_add(m_capture_session_output_container.get(), m_image_reader_output.get());
            assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to add ImageReaderOutput to CaptureSessionOutputContainer.");
        }

        // Create capture session.
        {
            m_capture_session_state_callbacks.context = this;
            m_capture_session_state_callbacks.onClosed = [](void* context, ACameraCaptureSession* session)
            {
                DEBUG_INFO("[Android Slam Camera Info] Session(ptr == %p) is closed.", session);
            };
            m_capture_session_state_callbacks.onReady = [](void* context, ACameraCaptureSession* session)
            {
                DEBUG_INFO("[Android Slam Camera Info] Session(ptr == %p) is ready.", session);
            };
            m_capture_session_state_callbacks.onActive = [](void* context, ACameraCaptureSession* session)
            {
                DEBUG_INFO("[Android Slam Camera Info] Session(ptr == %p) is activated.", session);
            };

            decltype(m_capture_session.get()) ptr;
            cs = ACameraDevice_createCaptureSession(m_camera_device.get(),
                                                    m_capture_session_output_container.get(),
                                                    &m_capture_session_state_callbacks,
                                                    &ptr);
            assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to create capture session.");

            m_capture_session.reset(ptr);
            assert(m_capture_session && "[Android Slam Sensor Info] CaptureSession is NULL.");
        }

        // Create capture request.
        {
            decltype(m_capture_request.get()) ptr;
            cs = ACameraDevice_createCaptureRequest(m_camera_device.get(), TEMPLATE_PREVIEW, &ptr);
            assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to create capture request.");

            m_capture_request.reset(ptr);
            assert(m_capture_request && "[Android Slam Sensor Info] CaptureRequest is NULL.");
        }

        // Create camera output target.
        {
            decltype(m_camera_output_target.get()) ptr;
            cs = ACameraOutputTarget_create(m_img_reader_window, &ptr);
            assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to create camera output target.");

            m_camera_output_target.reset(ptr);
            assert(m_camera_output_target && "[Android Slam Sensor Info] CameraOutputTarget is NULL.");
        }

        // Add target to capture request.
        {
            cs = ACaptureRequest_addTarget(m_capture_request.get(), m_camera_output_target.get());
            assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to add camera output target to capture request.");
        }
    }

    void SensorCamera::startCapture()
    {
        if (FILE* file = std::fopen("/storage/emulated/0/Documents/SlamCamera.txt", "w"))
            std::fclose(file);
        // Read capture results without changing AE, AF, frame rate, or OIS/EIS.
        // Timestamps join these records to SLAM_FRAME.ts in the debug log.
        m_capture_callbacks.onCaptureCompleted = [](void*, ACameraCaptureSession*,
                ACaptureRequest*, const ACameraMetadata* result) {
            const auto read = [result](uint32_t tag, int kind) -> double {
                ACameraMetadata_const_entry entry{};
                if (ACameraMetadata_getConstEntry(result, tag, &entry) != ACAMERA_OK || entry.count == 0)
                    return -1.0;
                switch (kind)
                {
                    case 0: return entry.data.u8[0];
                    case 1: return entry.data.i32[0];
                    case 2: return static_cast<double>(entry.data.i64[0]);
                    default: return entry.data.f[0];
                }
            };
            ACameraMetadata_const_entry timestamp_entry{};
            const long long timestamp = ACameraMetadata_getConstEntry(result, ACAMERA_SENSOR_TIMESTAMP,
                    &timestamp_entry) == ACAMERA_OK && timestamp_entry.count > 0 ? timestamp_entry.data.i64[0] : -1;
            const double exposure = read(ACAMERA_SENSOR_EXPOSURE_TIME, 2);
            const double skew = read(ACAMERA_SENSOR_ROLLING_SHUTTER_SKEW, 2);
            const double duration = read(ACAMERA_SENSOR_FRAME_DURATION, 2);
            char line[640];
            std::snprintf(line, sizeof(line),
                    "SLAM_CAMERA ts=%lld exposure_ms=%.3f frame_duration_ms=%.3f rolling_shutter_ms=%.3f iso=%.0f af=%.0f ae=%.0f focus_diopters=%.4f focal_mm=%.3f eis=%.0f ois=%.0f",
                    timestamp, exposure < 0 ? -1.0 : exposure * 1e-6,
                    duration < 0 ? -1.0 : duration * 1e-6, skew < 0 ? -1.0 : skew * 1e-6,
                    read(ACAMERA_SENSOR_SENSITIVITY, 1), read(ACAMERA_CONTROL_AF_STATE, 0),
                    read(ACAMERA_CONTROL_AE_STATE, 0), read(ACAMERA_LENS_FOCUS_DISTANCE, 3),
                    read(ACAMERA_LENS_FOCAL_LENGTH, 3), read(ACAMERA_CONTROL_VIDEO_STABILIZATION_MODE, 0),
                    read(ACAMERA_LENS_OPTICAL_STABILIZATION_MODE, 0));
            __android_log_print(ANDROID_LOG_INFO, "SLAM_DIAG", "%s", line);
            if (FILE* file = std::fopen("/storage/emulated/0/Documents/SlamCamera.txt", "a"))
            {
                std::fprintf(file, "%s\n", line);
                std::fclose(file);
            }
        };
        std::vector<ACaptureRequest*> capture_request_list{ m_capture_request.release() };
        camera_status_t cs = ACameraCaptureSession_setRepeatingRequest(
                m_capture_session.get(),
                &m_capture_callbacks,
                (int)capture_request_list.size(),
                capture_request_list.data(),
                nullptr
                );
        assert((cs == ACAMERA_OK) && "[Android Slam Sensor Info] Failed to set repeating request.");

        m_capture_request.reset(capture_request_list[0]);
        assert(m_capture_request && "[Android Slam Sensor Info] Capture Request is NULL when start capturing.");


        while(!getLatestImage())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000u));
        }
        assert(getLatestImage());

        DEBUG_INFO("[Android Slam Sensor Info] Camera Start Capturing.");
    }

    void SensorCamera::stopCapture()
    {
        ACameraCaptureSession_stopRepeating(m_capture_session.get());

        DEBUG_INFO("[Android Slam Sensor Info] Camera Stop Capturing.");
    }

    AImage* SensorCamera::getLatestImage() const
    {
        AImage* image = nullptr;
        media_status_t ms = AImageReader_acquireLatestImage(m_img_reader.get(), &image);
        if(ms == AMEDIA_OK && image)
        {
            m_curr_img_idx = (m_curr_img_idx + 1) % k_max_image_count;
            m_images[m_curr_img_idx].reset(image);
            m_image_buffers[m_curr_img_idx] = image;
        }

        return m_image_buffers[m_curr_img_idx];
    }

}
