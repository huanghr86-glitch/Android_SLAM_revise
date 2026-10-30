#include "sensor/SensorGnss.h"
#include <jni.h>

using android_slam::SensorGnss;

extern "C" JNIEXPORT void JNICALL
Java_cntlab_f304_androidslam_GnssCollector_nativeBeginSession(JNIEnv* env, jclass, jstring directory, jlong now) {
    if (!directory) return;
    const char* path = env->GetStringUTFChars(directory, nullptr);
    if (!path) return; // Pending JVM allocation exception.
    SensorGnss::instance().beginSession(path, now);
    env->ReleaseStringUTFChars(directory, path);
}

extern "C" JNIEXPORT void JNICALL
Java_cntlab_f304_androidslam_GnssCollector_nativeEndSession(JNIEnv*, jclass, jlong now) {
    SensorGnss::instance().endSession(now);
}

extern "C" JNIEXPORT void JNICALL
Java_cntlab_f304_androidslam_GnssCollector_nativeSetState(JNIEnv*, jclass, jint state, jlong now) {
    if (state >= 0 && state <= 8)
        SensorGnss::instance().setState(static_cast<android_slam::GnssState>(state), now);
}

extern "C" JNIEXPORT void JNICALL
Java_cntlab_f304_androidslam_GnssCollector_nativeOnFix(JNIEnv*, jclass,
        jlong measurement, jlong received, jlong utc, jdouble latitude, jdouble longitude,
        jdouble altitude, jdouble hacc, jdouble vacc, jdouble speed, jdouble bearing,
        jdouble time_uncertainty, jdouble speed_accuracy, jdouble bearing_accuracy, jint flags) {
    android_slam::GnssFix fix;
    fix.measurement_boot_ns = measurement;
    fix.received_boot_ns = received;
    fix.utc_ms = utc;
    fix.latitude_deg = latitude;
    fix.longitude_deg = longitude;
    fix.altitude_m = altitude;
    fix.horizontal_accuracy_m = hacc;
    fix.vertical_accuracy_m = vacc;
    fix.speed_mps = speed;
    fix.bearing_deg = bearing;
    fix.time_uncertainty_ns = time_uncertainty;
    fix.speed_accuracy_mps = speed_accuracy;
    fix.bearing_accuracy_deg = bearing_accuracy;
    fix.flags = static_cast<uint32_t>(flags);
    SensorGnss::instance().submit(fix);
}
