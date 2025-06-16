package com.programminghut.yolo_deploy;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_REQUEST_CODE = 200;
    private static final long DETECTION_INTERVAL_MS = 500; // Detection interval in milliseconds
    private static final long BOUNDING_BOX_RELOAD = 1000;  // Bounding box timeout in milliseconds (5 seconds)
    private static final int NOTIFICATION_PERMISSION_CODE = 1001;  // Custom request code for notifications
    private PreviewView previewView;
    private ImageView overlayView;  // ImageView to display bounding boxes overlay
    private Yolov5TFLiteDetector yolov5TFLiteDetector;
    private Paint boxPaint = new Paint();
    private Paint textPaint = new Paint();
    private ExecutorService cameraExecutor;
    private boolean detectionActive = false;
    private boolean isUsingFrontCamera = false;
    private ImageAnalysis imageAnalysis;
    private Handler boundingBoxHandler = new Handler(Looper.getMainLooper()); // Handler to manage bounding box updates
    private ArrayList<Recognition> lastRecognitions = new ArrayList<>();  // Store the last detection results globally
    private long lastBoundingBoxUpdateTime = 0;  // Track the last bounding box update time
    private MediaPlayer player;  // Declare MediaPlayer as a class variable to control it outside the method
    // Variables for microsleep detection
    private long microsleepStartTime = 0;
    private boolean microsleepDetected = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        overlayView = findViewById(R.id.overlayView);  // This will show the detection overlay
        Button toggleDetectionButton = findViewById(R.id.predict);
        Button switchCameraButton = findViewById(R.id.switchCamera);  // Switch camera button

        // Initialize the YOLOv5 TFLite detector
        yolov5TFLiteDetector = new Yolov5TFLiteDetector();
        yolov5TFLiteDetector.setModelFile("best-fp16 (2).tflite");
        yolov5TFLiteDetector.initialModel(this);

        // Set up paint for bounding boxes
        boxPaint.setStrokeWidth(5);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setColor(Color.RED);

        textPaint.setTextSize(50);
        textPaint.setColor(Color.GREEN);
        textPaint.setStyle(Paint.Style.FILL);

        // Executor for running camera-related tasks
        cameraExecutor = Executors.newSingleThreadExecutor();

        // Request camera permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_REQUEST_CODE);
        } else {
            startCamera();
        }

        toggleDetectionButton.setOnClickListener(this::toggleDetection);
        switchCameraButton.setOnClickListener(this::switchCamera);

        // Create notification channel
        createNotificationChannel();
    }

    public void toggleDetection(View view) {
        Button toggleDetectionButton = (Button) view;
        Log.i("Detection_active", "" + detectionActive);
        if (detectionActive) {
            detectionActive = false;
            toggleDetectionButton.setText("Detection Disabled");
            stopDetection();
        } else {
            detectionActive = true;
            toggleDetectionButton.setText("Detection Active");
            startDetection();
        }
    }

    public void switchCamera(View view) {
        isUsingFrontCamera = !isUsingFrontCamera;
        startCamera();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraUseCases(ProcessCameraProvider cameraProvider) {
        cameraProvider.unbindAll();

        Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(isUsingFrontCamera ? CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    private void startDetection() {
        Log.i("Detection_actives", "run");
        imageAnalysis.setAnalyzer(cameraExecutor, new ImageAnalysis.Analyzer() {
            private long lastDetectionTime = 0;

            @Override
            public void analyze(@NonNull ImageProxy imageProxy) {
                long currentTime = System.currentTimeMillis();

                if (currentTime - lastDetectionTime >= DETECTION_INTERVAL_MS && detectionActive) {
                    lastDetectionTime = currentTime;
                    runObjectDetection(imageProxy);
                } else {
                    updateOverlayWithStoredRecognitions(imageProxy);
                    imageProxy.close();
                }
            }
        });
    }

    private void stopDetection() {
        Log.i("Detection_actives", "stop");
        boundingBoxHandler.removeCallbacksAndMessages(null);
        if (imageAnalysis != null) {
            imageAnalysis.clearAnalyzer();
        }
        runOnUiThread(() -> overlayView.setImageBitmap(null));
        lastRecognitions.clear();
    }

    private void runObjectDetection(ImageProxy imageProxy) {
        Bitmap bitmap = convertImageProxyToBitmap(imageProxy);

        int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
        Bitmap rotatedBitmap = rotateAndMirrorBitmap(bitmap, rotationDegrees, isUsingFrontCamera);

        if (rotatedBitmap != null) {
            ArrayList<Recognition> recognitions = yolov5TFLiteDetector.detect(rotatedBitmap);
            Log.i("Detection_s", "" + recognitions);

            lastRecognitions = new ArrayList<>(recognitions);

            updateBoundingBoxes(rotatedBitmap, recognitions);

            lastBoundingBoxUpdateTime = System.currentTimeMillis();

            clearBoundingBoxesAfterInterval();
        }

        imageProxy.close();
    }

    // Filter overlapping recognitions and retain the ones with the highest confidence
    private ArrayList<Recognition> filterOverlappingRecognitions(ArrayList<Recognition> recognitions) {
        ArrayList<Recognition> filteredRecognitions = new ArrayList<>();

        for (Recognition current : recognitions) {
            boolean overlapFound = false;

            for (Recognition filtered : filteredRecognitions) {
                // Check if the bounding boxes overlap
                if (RectF.intersects(current.getLocation(), filtered.getLocation())) {
                    // Keep the one with the highest confidence
                    if (current.getConfidence() > filtered.getConfidence()) {
                        // Replace the lower confidence recognition
                        filteredRecognitions.remove(filtered);
                        filteredRecognitions.add(current);
                    }
                    overlapFound = true;
                    break;
                }
            }

            // If no overlap found, add the current recognition
            if (!overlapFound) {
                filteredRecognitions.add(current);
            }
        }

        return recognitions;
    }

    private void updateBoundingBoxes(Bitmap bitmap, ArrayList<Recognition> recognitions) {
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        ArrayList<Recognition> filteredRecognitions = filterOverlappingRecognitions(recognitions);

        boolean microsleepFound = false;

        for (Recognition recognition : filteredRecognitions) {
            if (recognition.getConfidence() > 0.5) {
                RectF location = recognition.getLocation();
                canvas.drawRect(location, boxPaint);
                canvas.drawText(recognition.getLabelName() + ": " + String.format("%.2f", recognition.getConfidence()), location.left, location.top, textPaint);

                if (recognition.getLabelName().equalsIgnoreCase("microsleep")) {
                    microsleepFound = true;
                    if (!microsleepDetected) {
                        microsleepDetected = true;
                        microsleepStartTime = System.currentTimeMillis();
                    }
                }

                Log.d("Detection", "Detected " + recognition.getLabelName() + " with confidence " + recognition.getConfidence());
            }
        }

        if (microsleepDetected && microsleepFound) {
            long currentTime = System.currentTimeMillis();
            if (currentTime - microsleepStartTime >= 5000) {  // 5 seconds threshold
                showMicrosleepWarning();
                playWarningSound();
                microsleepDetected = false;
            }
        } else {
            microsleepDetected = false;
            stopWarningSound();  // Stop sound if microsleep is no longer detected
        }

        runOnUiThread(() -> overlayView.setImageBitmap(mutableBitmap));
    }


    private void updateOverlayWithStoredRecognitions(ImageProxy imageProxy) {
        long currentTime = System.currentTimeMillis();

        Bitmap bitmap = convertImageProxyToBitmap(imageProxy);

        int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
        Bitmap rotatedBitmap = rotateAndMirrorBitmap(bitmap, rotationDegrees, isUsingFrontCamera);

        int imageWidth = rotatedBitmap.getWidth();
        int imageHeight = rotatedBitmap.getHeight();

        imageProxy.close();

        if (currentTime - lastBoundingBoxUpdateTime <= BOUNDING_BOX_RELOAD && !lastRecognitions.isEmpty()) {
            Bitmap emptyBitmap = Bitmap.createBitmap(previewView.getWidth(), previewView.getHeight(), Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(emptyBitmap);

            int previewWidth = previewView.getWidth();
            int previewHeight = previewView.getHeight();

            float widthScaleFactor = (float) previewWidth / imageWidth;
            float heightScaleFactor = (float) previewHeight / imageHeight;

            ArrayList<Recognition> filteredRecognitions = filterOverlappingRecognitions(lastRecognitions);

            for (Recognition recognition : filteredRecognitions) {
                RectF location = recognition.getLocation();

                float scaledLeft = location.left * widthScaleFactor;
                float scaledTop = location.top * heightScaleFactor;
                float scaledRight = location.right * widthScaleFactor;
                float scaledBottom = location.bottom * heightScaleFactor;

                RectF scaledLocation = new RectF(scaledLeft, scaledTop, scaledRight, scaledBottom);

                canvas.drawRect(scaledLocation, boxPaint);
                canvas.drawText(recognition.getLabelName() + ": " + String.format("%.2f", recognition.getConfidence()),
                        scaledLeft, scaledTop, textPaint);
            }

            runOnUiThread(() -> overlayView.setImageBitmap(emptyBitmap));
        } else {
            runOnUiThread(() -> overlayView.setImageBitmap(null));
        }
    }

    private void clearBoundingBoxesAfterInterval() {
        boundingBoxHandler.postDelayed(() -> {
            long currentTime = System.currentTimeMillis();
            if (currentTime - lastBoundingBoxUpdateTime >= BOUNDING_BOX_RELOAD) {
                runOnUiThread(() -> overlayView.setImageBitmap(null));
            }
        }, BOUNDING_BOX_RELOAD);
    }

    private Bitmap convertImageProxyToBitmap(ImageProxy imageProxy) {
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();

        YuvImage yuvImage = new YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, width, height), 100, out);
        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private Bitmap rotateAndMirrorBitmap(Bitmap bitmap, int rotationDegrees, boolean mirror) {
        Matrix matrix = new Matrix();

        if (rotationDegrees != 0) {
            matrix.postRotate(rotationDegrees);
        }

        if (mirror) {
            matrix.postScale(-1, 1, bitmap.getWidth() / 2f, bitmap.getHeight() / 2f);
        }

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
        boundingBoxHandler.removeCallbacksAndMessages(null);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == NOTIFICATION_PERMISSION_CODE) {
            // Check if the permission is granted
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted, you can now show the notification
                showMicrosleepWarning();
            } else {
                // Permission denied, handle accordingly (e.g., show a message)
                Toast.makeText(this, "Notification permission denied. Microsleep warnings will not appear.", Toast.LENGTH_SHORT).show();
            }
        }
    }


    private void showMicrosleepWarning() {
        NotificationCompat.Builder builder = new NotificationCompat.Builder(this, "MICROSLEEP_CHANNEL")
                .setSmallIcon(android.R.drawable.ic_dialog_alert)  // Using built-in alert icon
                .setContentTitle("Microsleep Detected")
                .setContentText("Warning: 10 seconds of microsleep detected!")
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setVibrate(new long[]{1000, 1000})
                .setAutoCancel(true);

        NotificationManagerCompat notificationManager = NotificationManagerCompat.from(this);
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return;
        }
        notificationManager.notify(1, builder.build());
    }


    private void playWarningSound() {
        try {
            if (player == null) {  // Initialize the player if it's not already initialized
                AssetFileDescriptor afd = getAssets().openFd("Danger Alarm Sound Effect.mp3");
                player = new MediaPlayer();
                player.setDataSource(afd.getFileDescriptor(), afd.getStartOffset(), afd.getLength());
                player.prepare();
                player.setLooping(true);  // Set the sound to loop continuously
            }
            player.start();  // Start playing the sound
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void stopWarningSound() {
        if (player != null && player.isPlaying()) {
            player.stop();  // Stop the sound
            player.reset();  // Reset the player for the next use
            player.release();  // Release resources
            player = null;  // Set player to null so it can be reinitialized when needed
        }
    }


    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            String channelId = "MICROSLEEP_CHANNEL";
            CharSequence name = "Microsleep Warning";
            String description = "Notification for microsleep warning";
            int importance = NotificationManager.IMPORTANCE_HIGH;
            NotificationChannel channel = new NotificationChannel(channelId, name, importance);
            channel.setDescription(description);

            NotificationManager notificationManager = getSystemService(NotificationManager.class);
            notificationManager.createNotificationChannel(channel);
        }
    }
}
