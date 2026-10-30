package cntlab.f304.androidslam;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;

import androidx.core.content.ContextCompat;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/** Foreground-only Android GNSS position solution, not raw satellite observations. */
final class GnssCollector implements LocationListener {
    private static final String TAG = "AndroidSlamGnss";
    // Keep these values identical to SensorGnss.h.
    private static final int PERMISSION_REQUIRED = 1, PRECISE_REQUIRED = 2,
            PROVIDER_UNAVAILABLE = 3, PROVIDER_DISABLED = 4, LISTENING = 5,
            PAUSED = 6, ERROR = 7;
    private static final int ALTITUDE = 1, HACC = 2, VACC = 4, SPEED = 8,
            BEARING = 16, TIME_UNCERTAINTY = 32, MOCK = 64,
            SPEED_ACCURACY = 128, BEARING_ACCURACY = 256;

    static { System.loadLibrary("android_slam"); }

    private final Context context;
    private final LocationManager manager;
    private boolean resumed;
    private boolean registered;
    private boolean closed;

    GnssCollector(Context context) {
        this.context = context;
        manager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);
        String session = "session_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                .format(new Date()) + "_" + SystemClock.elapsedRealtimeNanos();
        File root = context.getExternalFilesDir(null);
        File directory = root == null ? null : new File(root, "gnss/" + session);
        if (directory == null || (!directory.isDirectory() && !directory.mkdirs())) {
            directory = new File(context.getFilesDir(), "gnss/" + session);
        }
        String path = directory.isDirectory() || directory.mkdirs() ? directory.getAbsolutePath() : "";
        nativeBeginSession(path, SystemClock.elapsedRealtimeNanos());
        if (path.isEmpty()) Log.e(TAG, "Cannot create GNSS log directory; collection will continue");
        else Log.i(TAG, "GNSS session directory: " + path);
    }

    /** All lifecycle and LocationListener methods execute on the main looper. */
    @SuppressLint("MissingPermission") // Explicit fine-permission check immediately below.
    void resume() {
        if (closed) return;
        resumed = true;
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
            unregister();
            boolean coarse = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION)
                    == PackageManager.PERMISSION_GRANTED;
            state(coarse ? PRECISE_REQUIRED : PERMISSION_REQUIRED);
            return;
        }
        try {
            if (manager == null || !manager.getAllProviders().contains(LocationManager.GPS_PROVIDER)) {
                state(PROVIDER_UNAVAILABLE);
                return;
            }
            if (!registered) {
                // Request ~1 Hz. Actual rate is device-dependent. Do not inject last-known fixes.
                manager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000L, 0.0f,
                        this, Looper.getMainLooper());
                registered = true;
            }
            // Stay subscribed while GPS is disabled so enabling it can resume acquisition.
            state(manager.isProviderEnabled(LocationManager.GPS_PROVIDER) ? LISTENING : PROVIDER_DISABLED);
        } catch (SecurityException e) {
            unregister();
            state(PERMISSION_REQUIRED);
            Log.w(TAG, "Location permission unavailable", e);
        } catch (RuntimeException e) {
            unregister();
            state(ERROR);
            Log.e(TAG, "Cannot subscribe to GPS_PROVIDER", e);
        }
    }

    void pause() {
        if (closed) return;
        resumed = false;
        unregister();
        state(PAUSED);
    }

    void close() {
        if (closed) return;
        pause();
        nativeEndSession(SystemClock.elapsedRealtimeNanos());
        closed = true;
    }

    private void unregister() {
        if (manager != null && registered) {
            try { manager.removeUpdates(this); }
            catch (RuntimeException e) { Log.w(TAG, "Cannot remove GPS updates", e); }
        }
        registered = false;
    }

    private void state(int state) {
        nativeSetState(state, SystemClock.elapsedRealtimeNanos());
    }

    @Override
    public void onLocationChanged(Location location) {
        final long received = SystemClock.elapsedRealtimeNanos();
        if (!resumed || !registered || closed ||
                !LocationManager.GPS_PROVIDER.equals(location.getProvider())) return;
        int flags = (location.hasAltitude() ? ALTITUDE : 0)
                | (location.hasAccuracy() ? HACC : 0)
                | (location.hasVerticalAccuracy() ? VACC : 0)
                | (location.hasSpeed() ? SPEED : 0)
                | (location.hasBearing() ? BEARING : 0)
                | (location.hasElapsedRealtimeUncertaintyNanos() ? TIME_UNCERTAINTY : 0)
                | (location.isFromMockProvider() ? MOCK : 0)
                | (location.hasSpeedAccuracy() ? SPEED_ACCURACY : 0)
                | (location.hasBearingAccuracy() ? BEARING_ACCURACY : 0);
        nativeOnFix(location.getElapsedRealtimeNanos(), received, location.getTime(),
                location.getLatitude(), location.getLongitude(),
                location.hasAltitude() ? location.getAltitude() : 0,
                location.hasAccuracy() ? location.getAccuracy() : 0,
                location.hasVerticalAccuracy() ? location.getVerticalAccuracyMeters() : 0,
                location.hasSpeed() ? location.getSpeed() : 0,
                location.hasBearing() ? location.getBearing() : 0,
                location.hasElapsedRealtimeUncertaintyNanos() ? location.getElapsedRealtimeUncertaintyNanos() : 0,
                location.hasSpeedAccuracy() ? location.getSpeedAccuracyMetersPerSecond() : 0,
                location.hasBearingAccuracy() ? location.getBearingAccuracyDegrees() : 0, flags);
    }

    @Override public void onProviderEnabled(String provider) {
        if (resumed && LocationManager.GPS_PROVIDER.equals(provider)) resume();
    }
    @Override public void onProviderDisabled(String provider) {
        if (resumed && LocationManager.GPS_PROVIDER.equals(provider)) state(PROVIDER_DISABLED);
    }
    @Override public void onStatusChanged(String provider, int status, Bundle extras) {
        // Deprecated provider status is not a fix-quality indicator.
    }

    private static native void nativeBeginSession(String directory, long nowBootNs);
    private static native void nativeEndSession(long nowBootNs);
    private static native void nativeSetState(int state, long nowBootNs);
    private static native void nativeOnFix(long measurementBootNs, long receivedBootNs, long utcMs,
            double latitude, double longitude, double altitude, double horizontalAccuracy,
            double verticalAccuracy, double speed, double bearing, double timeUncertainty,
            double speedAccuracy, double bearingAccuracy, int flags);
}
