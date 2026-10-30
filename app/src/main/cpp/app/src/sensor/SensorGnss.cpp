#include "sensor/SensorGnss.h"

#include <cmath>
#include <cinttypes>

namespace android_slam
{
    namespace {
        bool nonnegative(double value) { return std::isfinite(value) && value >= 0; }

        const char* invalidReason(const GnssFix& f) {
            if (!std::isfinite(f.latitude_deg) || !std::isfinite(f.longitude_deg) ||
                std::abs(f.latitude_deg) > 90 || std::abs(f.longitude_deg) > 180)
                return "invalid_coordinates";
            if (f.measurement_boot_ns <= 0 || f.received_boot_ns < f.measurement_boot_ns || f.utc_ms <= 0)
                return "invalid_time";
            if (f.source != 1) return "unsupported_source";
            if ((f.flags & HasAltitude) && !std::isfinite(f.altitude_m)) return "invalid_altitude";
            if ((f.flags & HasHorizontalAccuracy) && !nonnegative(f.horizontal_accuracy_m)) return "invalid_hacc";
            if ((f.flags & HasVerticalAccuracy) && !nonnegative(f.vertical_accuracy_m)) return "invalid_vacc";
            if ((f.flags & HasSpeed) && !nonnegative(f.speed_mps)) return "invalid_speed";
            if ((f.flags & HasBearing) && (!nonnegative(f.bearing_deg) || f.bearing_deg >= 360)) return "invalid_bearing";
            if ((f.flags & HasTimeUncertainty) && !nonnegative(f.time_uncertainty_ns)) return "invalid_time_uncertainty";
            if ((f.flags & HasSpeedAccuracy) && !nonnegative(f.speed_accuracy_mps)) return "invalid_speed_accuracy";
            if ((f.flags & HasBearingAccuracy) && !nonnegative(f.bearing_accuracy_deg)) return "invalid_bearing_accuracy";
            if (f.flags & MockLocation) return "mock_location";
            return nullptr;
        }
    }

    const char* gnssStateName(GnssState state) {
        switch (state) {
            case GnssState::Starting: return "STARTING";
            case GnssState::PermissionRequired: return "LOCATION_PERMISSION_REQUIRED";
            case GnssState::PrecisePermissionRequired: return "PRECISE_LOCATION_REQUIRED";
            case GnssState::ProviderUnavailable: return "GPS_PROVIDER_UNAVAILABLE";
            case GnssState::ProviderDisabled: return "GPS_PROVIDER_DISABLED";
            case GnssState::Listening: return "LISTENING";
            case GnssState::Paused: return "PAUSED";
            case GnssState::Error: return "ERROR";
            case GnssState::Closed: return "CLOSED";
        }
        return "UNKNOWN";
    }

    SensorGnss& SensorGnss::instance() { static SensorGnss sensor; return sensor; }
    SensorGnss::~SensorGnss() { closeFiles(); }

    void SensorGnss::closeFiles() {
        if (m_fix_file) std::fclose(m_fix_file);
        if (m_state_file) std::fclose(m_state_file);
        m_fix_file = m_state_file = nullptr;
    }

    void SensorGnss::beginSession(const std::string& directory, int64_t now_boot_ns) {
        std::lock_guard<std::mutex> lock(m_mutex);
        closeFiles();
        const int camera_source = m_snapshot.camera_timestamp_source;
        m_snapshot = GnssSnapshot{};
        m_snapshot.camera_timestamp_source = camera_source;
        m_snapshot.log_directory = directory;
        m_queue.clear();
        m_session_open = true;
        if (!directory.empty()) {
            m_fix_file = std::fopen((directory + "/gnss.csv").c_str(), "w");
            m_state_file = std::fopen((directory + "/gnss_status.csv").c_str(), "w");
        }
        m_snapshot.logging_ok = m_fix_file && m_state_file;
        if (m_fix_file) {
            if (std::fprintf(m_fix_file, "measurement_boot_ns,received_boot_ns,utc_ms,latitude_deg,longitude_deg,altitude_m,horizontal_accuracy_m,vertical_accuracy_m,speed_mps,bearing_deg,time_uncertainty_ns,speed_accuracy_mps,bearing_accuracy_deg,flags,source,accepted,reason\n") < 0 ||
                std::fflush(m_fix_file) != 0) m_snapshot.logging_ok = false;
        }
        if (m_state_file) {
            if (std::fprintf(m_state_file, "event_boot_ns,state,camera_timestamp_source\n") < 0)
                m_snapshot.logging_ok = false;
        }
        writeState(now_boot_ns);
    }

    void SensorGnss::writeState(int64_t now_boot_ns) {
        if (m_state_file && (std::fprintf(m_state_file, "%" PRId64 ",%s,%d\n", now_boot_ns,
                gnssStateName(m_snapshot.state), m_snapshot.camera_timestamp_source) < 0 ||
                std::fflush(m_state_file) != 0)) m_snapshot.logging_ok = false;
    }

    void SensorGnss::endSession(int64_t now_boot_ns) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_session_open) return;
        m_snapshot.state = GnssState::Closed;
        writeState(now_boot_ns);
        closeFiles();
        m_queue.clear();
        m_session_open = false;
    }

    void SensorGnss::setState(GnssState state, int64_t now_boot_ns) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_session_open || m_snapshot.state == state) return;
        m_snapshot.state = state;
        // Prevent future consumers from replaying queued pre-pause samples.
        if (state != GnssState::Listening) m_queue.clear();
        writeState(now_boot_ns);
    }

    void SensorGnss::setCameraTimestampSource(int source, int64_t now_boot_ns) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_snapshot.camera_timestamp_source = source;
        if (m_session_open) writeState(now_boot_ns);
    }

    bool SensorGnss::submit(const GnssFix& fix) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_session_open) return false;
        ++m_snapshot.received;
        const char* reason = invalidReason(fix);
        if (!reason && m_snapshot.state != GnssState::Listening) reason = "not_listening";
        if (!reason && m_snapshot.has_fix && fix.measurement_boot_ns <= m_snapshot.latest.measurement_boot_ns)
            reason = "duplicate_or_out_of_order";
        if (!reason) {
            m_snapshot.latest = fix;
            m_snapshot.has_fix = true;
            ++m_snapshot.accepted;
            if (m_queue.size() == kQueueCapacity) {
                m_queue.pop_front();
                ++m_snapshot.queue_overflows;
            }
            m_queue.push_back(fix);
        } else ++m_snapshot.rejected;
        if (m_fix_file && (std::fprintf(m_fix_file,
                "%" PRId64 ",%" PRId64 ",%" PRId64 ",%.10f,%.10f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.4f,%.4f,%u,%d,%d,%s\n",
                fix.measurement_boot_ns, fix.received_boot_ns, fix.utc_ms,
                fix.latitude_deg, fix.longitude_deg, fix.altitude_m,
                fix.horizontal_accuracy_m, fix.vertical_accuracy_m, fix.speed_mps, fix.bearing_deg,
                fix.time_uncertainty_ns, fix.speed_accuracy_mps, fix.bearing_accuracy_deg,
                fix.flags, fix.source, reason ? 0 : 1, reason ? reason : "recorded") < 0 ||
                std::fflush(m_fix_file) != 0)) m_snapshot.logging_ok = false;
        return !reason;
    }

    GnssSnapshot SensorGnss::snapshot() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto result = m_snapshot;
        result.queued = m_queue.size();
        return result;
    }

    std::vector<GnssFix> SensorGnss::drain() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<GnssFix> result(m_queue.begin(), m_queue.end());
        m_queue.clear();
        return result;
    }
}
