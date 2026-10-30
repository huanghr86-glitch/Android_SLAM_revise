#pragma once

#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace android_slam
{
    // Values shared with GnssCollector.java; see docs/gnss-integration.md.
    enum class GnssState : int {
        Starting = 0, PermissionRequired = 1, PrecisePermissionRequired = 2,
        ProviderUnavailable = 3, ProviderDisabled = 4, Listening = 5,
        Paused = 6, Error = 7, Closed = 8
    };

    enum GnssFlags : uint32_t {
        HasAltitude = 1u << 0, HasHorizontalAccuracy = 1u << 1,
        HasVerticalAccuracy = 1u << 2, HasSpeed = 1u << 3,
        HasBearing = 1u << 4, HasTimeUncertainty = 1u << 5,
        MockLocation = 1u << 6, HasSpeedAccuracy = 1u << 7,
        HasBearingAccuracy = 1u << 8
    };

    struct GnssFix {
        // WGS84, degrees and ellipsoid height in meters. Zero optional values
        // are placeholders only; flags determine whether a field is present.
        double latitude_deg = 0, longitude_deg = 0, altitude_m = 0;
        double horizontal_accuracy_m = 0, vertical_accuracy_m = 0;
        double speed_mps = 0, bearing_deg = 0;
        double time_uncertainty_ns = 0, speed_accuracy_mps = 0, bearing_accuracy_deg = 0;
        int64_t measurement_boot_ns = 0, received_boot_ns = 0, utc_ms = 0;
        uint32_t flags = 0;
        // 1 = Android GPS_PROVIDER. Future receiver adapters must convert to
        // this clock/datum contract before submitting. Android exposes no RTK status.
        int source = 1;
    };

    struct GnssSnapshot {
        GnssState state = GnssState::Starting;
        GnssFix latest;
        bool has_fix = false, logging_ok = false;
        uint64_t received = 0, accepted = 0, rejected = 0, queue_overflows = 0;
        size_t queued = 0;
        std::string log_directory;
        // -1 = not queried, 0 = UNKNOWN, 1 = REALTIME (boot clock).
        int camera_timestamp_source = -1;
    };

    // JNI callbacks and the render/SLAM threads never share a Scene pointer.
    // This process-lifetime mailbox is protected by one mutex, including logging.
    class SensorGnss {
    public:
        static SensorGnss& instance();
        static constexpr size_t kQueueCapacity = 512;
        void beginSession(const std::string& directory, int64_t now_boot_ns);
        void endSession(int64_t now_boot_ns);
        void setState(GnssState state, int64_t now_boot_ns);
        void setCameraTimestampSource(int source, int64_t now_boot_ns);
        bool submit(const GnssFix& fix);
        GnssSnapshot snapshot() const;
        std::vector<GnssFix> drain();

    private:
        SensorGnss() = default;
        ~SensorGnss();
        void closeFiles();
        void writeState(int64_t now_boot_ns);
        mutable std::mutex m_mutex;
        GnssSnapshot m_snapshot;
        std::deque<GnssFix> m_queue;
        FILE* m_fix_file = nullptr;
        FILE* m_state_file = nullptr;
        bool m_session_open = false;
    };
    const char* gnssStateName(GnssState state);
}
