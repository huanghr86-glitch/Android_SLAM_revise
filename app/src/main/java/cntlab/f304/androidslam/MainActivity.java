package cntlab.f304.androidslam;

import android.Manifest;
import android.app.NativeActivity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.provider.Settings;
import android.os.Environment;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends NativeActivity
{
    private static final int REQUEST_SENSOR_PERMISSIONS = 5;
    private GnssCollector gnssCollector;
    private boolean activityResumed;
    private boolean allFilesAccessChecked;

    @Override
    protected void onCreate(Bundle state)
    {
        super.onCreate(state);

        setContentView(R.layout.activity_main);
        gnssCollector = new GnssCollector(this);

        // One sensor permission request, then the existing storage requests.
        // Android 12 requires fine and coarse location in the same request.
        java.util.ArrayList<String> permissions = new java.util.ArrayList<>();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) permissions.add(Manifest.permission.CAMERA);
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.ACCESS_COARSE_LOCATION);
            permissions.add(Manifest.permission.ACCESS_FINE_LOCATION);
        }
        if (!permissions.isEmpty()) {
            ActivityCompat.requestPermissions(this, permissions.toArray(new String[0]), REQUEST_SENSOR_PERMISSIONS);
        } else {
            requestStorageAfterSensors();
        }
    }

    @Override protected void onResume() {
        super.onResume();
        activityResumed = true;
        if (gnssCollector != null) gnssCollector.resume();
    }

    @Override protected void onPause() {
        activityResumed = false;
        if (gnssCollector != null) gnssCollector.pause();
        super.onPause();
    }

    @Override protected void onDestroy() {
        if (gnssCollector != null) gnssCollector.close();
        super.onDestroy();
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_SENSOR_PERMISSIONS) {
            if (activityResumed && gnssCollector != null) gnssCollector.resume();
            requestStorageAfterSensors();
        } else if (requestCode == REQUEST_EXTERNAL_STORAGE) {
            requestAllFilesAccess(this);
        }
    }

    private void requestStorageAfterSensors() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            verifyStoragePermission(this);
        } else {
            requestAllFilesAccess(this);
        }
    }

    private void requestAllFilesAccess(NativeActivity activity) {
        if (allFilesAccessChecked) return;
        allFilesAccessChecked = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                try {
                    Intent intent = new Intent(
                            Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                            Uri.parse("package:" + activity.getPackageName()));
                    activity.startActivity(intent);
                } catch (Exception e) {
                    try {
                        Intent intent = new Intent(
                                Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
                        activity.startActivity(intent);
                    } catch (Exception ignored) {}
                }
            }
        }
    }

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE",
            "android.permission.MANAGE_EXTERNAL_STORAGE",
            "android.permission.MOUNT_UNMOUNT_FILESYSTEMS"
    };
    public void verifyStoragePermission(NativeActivity activity){
        try{
            int permission = ActivityCompat.checkSelfPermission(activity,"android.permission.WRITE_EXTERNAL_STORAGE");
            if(permission!= PackageManager.PERMISSION_GRANTED){
                ActivityCompat.requestPermissions(activity,PERMISSIONS_STORAGE, REQUEST_EXTERNAL_STORAGE);
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

}
