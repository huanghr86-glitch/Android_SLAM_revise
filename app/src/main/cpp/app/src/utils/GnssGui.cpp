#include "utils/GnssGui.h"
#include "sensor/SensorGnss.h"
#include <imgui.h>
#include <time.h>

namespace android_slam {
    void drawGnssPanel() {
        if (!ImGui::CollapsingHeader(u8"GNSS 采集状态", ImGuiTreeNodeFlags_DefaultOpen)) return;
        const auto data = SensorGnss::instance().snapshot();
        ImGui::Text("GNSS: %s", gnssStateName(data.state));
        ImGui::TextUnformatted("GPS_PROVIDER / WGS84 / collection only");
        ImGui::Text("Samples: %llu received / %llu recorded / %llu rejected",
                static_cast<unsigned long long>(data.received),
                static_cast<unsigned long long>(data.accepted),
                static_cast<unsigned long long>(data.rejected));
        if (data.has_fix) {
            timespec now{};
            clock_gettime(CLOCK_BOOTTIME, &now);
            const int64_t boot_ns = static_cast<int64_t>(now.tv_sec) * 1000000000LL + now.tv_nsec;
            const double age_s = static_cast<double>(boot_ns - data.latest.measurement_boot_ns) / 1e9;
            const bool fresh = data.state == GnssState::Listening && age_s >= 0 && age_s <= 3;
            ImGui::Text("Last sample: %s / age %.1f s", fresh ? "RECENT" : "STALE", age_s);
            ImGui::Text("Lat %.9f / Lon %.9f", data.latest.latitude_deg, data.latest.longitude_deg);
            if (data.latest.flags & HasHorizontalAccuracy)
                ImGui::Text("Horizontal accuracy (68%%): %.2f m", data.latest.horizontal_accuracy_m);
            else ImGui::TextUnformatted("Horizontal accuracy: unavailable");
            if (data.latest.flags & HasAltitude)
                ImGui::Text("Ellipsoid height: %.2f m", data.latest.altitude_m);
            else ImGui::TextUnformatted("Ellipsoid height: unavailable");
            if (data.latest.flags & HasVerticalAccuracy)
                ImGui::Text("Vertical accuracy (68%%): %.2f m", data.latest.vertical_accuracy_m);
            ImGui::Text("Delivery delay: %.1f ms", static_cast<double>(
                    data.latest.received_boot_ns - data.latest.measurement_boot_ns) / 1e6);
        } else ImGui::TextUnformatted("No recorded satellite fix yet");
        ImGui::Text("Camera clock: %s", data.camera_timestamp_source == 1 ? "BOOT / REALTIME" :
                data.camera_timestamp_source == -1 ? "not queried yet" : "UNKNOWN (alignment required)");
        ImGui::Text("GNSS CSV: %s", data.logging_ok ? "OK" : "NOT WRITING / ERROR");
        ImGui::TextWrapped("%s", data.log_directory.c_str());
    }
}
