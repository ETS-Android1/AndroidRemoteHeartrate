package com.example.mytest;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.commons.math3.filter.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.android.JavaCameraView;
import org.opencv.imgproc.Imgproc;

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.anychart.AnyChart;
import com.anychart.AnyChartView;
import com.anychart.chart.common.dataentry.DataEntry;
import com.anychart.chart.common.dataentry.ValueDataEntry;
import com.anychart.charts.Cartesian;
import com.anychart.core.cartesian.series.Line;
import com.anychart.data.Mapping;
import com.anychart.data.Set;
import com.anychart.enums.Anchor;
import com.anychart.enums.TooltipPositionMode;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.solutions.hands.Hands;
import com.google.mediapipe.solutions.hands.HandsOptions;

import com.google.common.primitives.Floats;

import uk.me.berndporr.iirj.*;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private int REQUEST_CODE_PERMISSIONS = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA"};

    Camera camera;
    ExecutorService executorService;
    ProcessCameraProvider provider;
    ImageAnalysis analysis;
    CameraSelector cameraSelector;
    ImageView imageView;

    Button torchlight;

    private boolean hand_detected = false;
    private int gc_count = 0;
    private int rectsize = 0;
    private int wid = 0, heig = 0;
    private long old_time = 0;
    private double prev_frame_value = 0.0;
    private double alpha_value = 1.0;
    private ArrayList<Float> effective_value_list;
    private float[] afteremd = new float[128];
    private ArrayList<Integer> output_list = new ArrayList<>();

    private Point rt, rb, lt, lb;
    private Scalar colorspace_Count = new Scalar(0,0,0);
    private Mat perspective;

    //kalman filter
    private double dt = 1d;
    private double measurementNoise = 2d;
    RealMatrix A = new Array2DRowRealMatrix(new double[][] { { 1, dt }, { 0, 1 } });
    RealMatrix B = null;
    RealMatrix H = new Array2DRowRealMatrix(new double[][] { { 1d, 0d } });
    RealVector x = new ArrayRealVector(new double[] { 80, 0 });
    RealMatrix Q = WaveProcess.whiteNoise(2,1);
    RealMatrix P0 = new Array2DRowRealMatrix(new double[][] { { 2, 1 }, { 1, 2 } });
    RealMatrix Rnc = new Array2DRowRealMatrix(new double[] { Math.pow(measurementNoise, 2) });
    ProcessModel pm = new DefaultProcessModel(A, B, Q, x, P0);
    MeasurementModel mm = new DefaultMeasurementModel(H, Rnc);
    KalmanFilter filter = new KalmanFilter(pm, mm);

    private TextView outputText;
    private JavaCameraView mOpenCvCameraView;

    //bandpass filter
    private Butterworth butterworth = new Butterworth();

    //mediapipe
    HandsOptions handsOptions;
    Hands mphands;
    NormalizedLandmark root, index, pinky, prev_root = null, prev_pinky = null;

    //line chart
    private AnyChartView anyChartView;
    private Cartesian cartesian;
    private List<DataEntry> seriesData;
    private Set set;
    private Mapping mapping;
    private Line series1;

    static {
        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded");
    }

    private void SetupPipeline(){
        handsOptions =
                HandsOptions.builder()
                        .setStaticImageMode(true)
                        .setMaxNumHands(1)
                        .setRunOnGpu(true)
                        .build();
        mphands = new Hands(this, handsOptions);
        mphands.setResultListener(
                handsResult -> {
                    if(handsResult.multiHandLandmarks().isEmpty()){
                        hand_detected=false;
                        prev_root = null;
                        prev_pinky = null;
                    }else{
                        root = handsResult.multiHandLandmarks().get(0).getLandmark(0);
                        index = handsResult.multiHandLandmarks().get(0).getLandmark(5);
                        pinky = handsResult.multiHandLandmarks().get(0).getLandmark(17);

                        rt = new Point(index.getX() * wid, index.getY() * heig);
                        lt = new Point(pinky.getX() * wid, pinky.getY() * heig);

                        if(prev_root == null || prev_pinky == null){
                            prev_root = root;
                            prev_pinky = pinky;
                            hand_detected = false;
                            return;
                        }else{
                            double root_tmp = Math.sqrt(Math.pow(prev_root.getX() - root.getX(), 2)+Math.pow(prev_root.getY() - root.getY(), 2));
                            double pinky_tmp = Math.sqrt(Math.pow(prev_pinky.getX() - pinky.getX(), 2)+Math.pow(prev_pinky.getY() - pinky.getY(), 2));
                            if(root_tmp < 30 && pinky_tmp < 30){
                                hand_detected = true;
                            }else{
                                hand_detected = false;
                                return;
                            }

                        }

                        float disx = (root.getX() - ((index.getX() + pinky.getX()) / 2)) / 2;
                        float disy = (root.getY() - ((index.getY() + pinky.getY()) / 2)) / 2 ;

                        rb = new Point((index.getX() + disx) * wid, (index.getY() + disy) * heig);
                        lb = new Point((pinky.getX() + disx) * wid, (pinky.getY() + disy) * heig);

                        MatOfPoint2f src = new MatOfPoint2f(
                                rt, lt, rb, lb
                        );
                        MatOfPoint2f dst = new MatOfPoint2f(
                                new Point(0, 0),
                                new Point(200,0),
                                new Point(0, 200),
                                new Point(200,200)
                        );
                        perspective = Imgproc.getPerspectiveTransform(src,dst);

                        src.release();
                        dst.release();

                    }
        });
        mphands.setErrorListener((message, e) -> Log.e(TAG, "MediaPipe Hands error:" + message));
    }

    private void LoadMatToPipeline(Mat input){
        Bitmap bmp = Bitmap.createBitmap(input.cols(),input.rows(), Bitmap.Config.ARGB_8888  );
        Utils.matToBitmap(input, bmp);
        mphands.send(bmp);
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);
        
        imageView = findViewById(R.id.image_view);
        torchlight = findViewById(R.id.button2);
        torchlight.setOnClickListener(new Button.OnClickListener(){
            @Override
            public void onClick(View v){
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if(torchlight.getText() == "Open"){
                            camera.getCameraControl().enableTorch(true);
                            torchlight.setText("Close");
                        }else{
                            camera.getCameraControl().enableTorch(false);
                            torchlight.setText("Open");
                        }

                    }
                });

            }
        });


        executorService = Executors.newSingleThreadExecutor();

        outputText = (TextView) findViewById(R.id.textView2);
        outputText.setText(Integer.toString((int)filter.getStateEstimation()[0]));

        anyChartView = findViewById(R.id.any_chart_view);
        anyChartView.setProgressBar(findViewById(R.id.progress_bar));
        cartesian = AnyChart.line();
        cartesian.animation(true);
        cartesian.padding(10d, 20d, 5d, 20d);
        cartesian.tooltip().positionMode(TooltipPositionMode.POINT);
        seriesData = new ArrayList<>();
        for(int i = 0; i < 128; i++){
            seriesData.add(new ValueDataEntry(i,0));
        }
        set = Set.instantiate();
        set.data(seriesData);
        mapping = set.mapAs("{ x: 'x', value: 'value' }");
        series1 = cartesian.line(mapping);
        series1.name("HeartRate");
        series1.tooltip().position("right")
                .anchor(Anchor.LEFT_CENTER)
                .offsetX(5d)
                .offsetY(5d);
        anyChartView.setChart(cartesian);

        if (allPermissionsGranted()) {
            Log.i("Create","Start Camera");
            startCamera();
        } else {
            Log.i("Create","Require Permissions");
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    @Override
    public void onResume() {
        super.onResume();
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                provider = cameraProviderFuture.get();
                analysis = setImageAnalysis();
                cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();
                camera = provider.bindToLifecycle(
                        ((LifecycleOwner) this),
                        cameraSelector,
                        analysis);
            } catch (InterruptedException | ExecutionException e) {
            }
        }, ContextCompat.getMainExecutor(this));

        SetupPipeline();
        effective_value_list = new ArrayList<Float>();
        old_time = System.currentTimeMillis();
        butterworth.bandPass(5, 30, 1.85,1.15);
    }

    public ImageAnalysis setImageAnalysis() {

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setImageQueueDepth(1).build();

        imageAnalysis.setAnalyzer(executorService,
                new ImageAnalysis.Analyzer() {
                    @Override
                    public void analyze(ImageProxy image) {
                        //Analyzing live camera feed begins.
                        heig = image.getHeight();
                        wid = image.getWidth();

                        @SuppressLint("UnsafeExperimentalUsageError") Image tmp = image.getImage();
                        if(tmp==null)
                            return;
                        Mat mat = ImageToMat(tmp);
                        Mat previewMat = onCameraFrame(mat);
                        Matrix matrix = new Matrix();
                        matrix.postRotate(90);
                        Bitmap outtmp = Bitmap.createBitmap(previewMat.cols(),previewMat.rows(),Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(previewMat, outtmp);
                        Bitmap outview = Bitmap.createBitmap(outtmp, 0, 0, outtmp.getWidth(), outtmp.getHeight(), matrix,true);

                        image.close();
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                imageView.setImageBitmap(outview);

                            }
                        });

                    }
                });


        return imageAnalysis;

    }

    private Mat ImageToMat(Image input){
        Image.Plane[] planes = input.getPlanes();
        int w = input.getWidth();
        int h = input.getHeight();
        int chromaPixelStride = planes[1].getPixelStride();
        Mat mRgba = new Mat();

        if (chromaPixelStride == 2) { // Chroma channels are interleaved
            assert(planes[0].getPixelStride() == 1);
            assert(planes[2].getPixelStride() == 2);
            ByteBuffer y_plane = planes[0].getBuffer();
            int y_plane_step = planes[0].getRowStride();
            ByteBuffer uv_plane1 = planes[1].getBuffer();
            int uv_plane1_step = planes[1].getRowStride();
            ByteBuffer uv_plane2 = planes[2].getBuffer();
            int uv_plane2_step = planes[2].getRowStride();
            Mat y_mat = new Mat(h, w, CvType.CV_8UC1, y_plane, y_plane_step);
            Mat uv_mat1 = new Mat(h / 2, w / 2, CvType.CV_8UC2, uv_plane1, uv_plane1_step);
            Mat uv_mat2 = new Mat(h / 2, w / 2, CvType.CV_8UC2, uv_plane2, uv_plane2_step);
            long addr_diff = uv_mat2.dataAddr() - uv_mat1.dataAddr();
            if (addr_diff > 0) {
                assert(addr_diff == 1);
                Imgproc.cvtColorTwoPlane(y_mat, uv_mat1, mRgba, Imgproc.COLOR_YUV2RGBA_NV12);
            } else {
                assert(addr_diff == -1);
                Imgproc.cvtColorTwoPlane(y_mat, uv_mat2, mRgba, Imgproc.COLOR_YUV2RGBA_NV21);
            }
            return mRgba;
        } else { // Chroma channels are not interleaved
            byte[] yuv_bytes = new byte[w * (h + h / 2)];
            ByteBuffer y_plane = planes[0].getBuffer();
            ByteBuffer u_plane = planes[1].getBuffer();
            ByteBuffer v_plane = planes[2].getBuffer();

            int yuv_bytes_offset = 0;

            int y_plane_step = planes[0].getRowStride();
            if (y_plane_step == w) {
                y_plane.get(yuv_bytes, 0, w * h);
                yuv_bytes_offset = w * h;
            } else {
                int padding = y_plane_step - w;
                for (int i = 0; i < h; i++) {
                    y_plane.get(yuv_bytes, yuv_bytes_offset, w);
                    yuv_bytes_offset += w;
                    if (i < h - 1) {
                        y_plane.position(y_plane.position() + padding);
                    }
                }
                assert (yuv_bytes_offset == w * h);
            }

            int chromaRowStride = planes[1].getRowStride();
            int chromaRowPadding = chromaRowStride - w / 2;

            if (chromaRowPadding == 0) {
                // When the row stride of the chroma channels equals their width, we can copy
                // the entire channels in one go
                u_plane.get(yuv_bytes, yuv_bytes_offset, w * h / 4);
                yuv_bytes_offset += w * h / 4;
                v_plane.get(yuv_bytes, yuv_bytes_offset, w * h / 4);
            } else {
                // When not equal, we need to copy the channels row by row
                for (int i = 0; i < h / 2; i++) {
                    u_plane.get(yuv_bytes, yuv_bytes_offset, w / 2);
                    yuv_bytes_offset += w / 2;
                    if (i < h / 2 - 1) {
                        u_plane.position(u_plane.position() + chromaRowPadding);
                    }
                }
                for (int i = 0; i < h / 2; i++) {
                    v_plane.get(yuv_bytes, yuv_bytes_offset, w / 2);
                    yuv_bytes_offset += w / 2;
                    if (i < h / 2 - 1) {
                        v_plane.position(v_plane.position() + chromaRowPadding);
                    }
                }
            }

            Mat yuv_mat = new Mat(h + h / 2, w, CvType.CV_8UC1);
            yuv_mat.put(0, 0, yuv_bytes);
            Imgproc.cvtColor(yuv_mat, mRgba, Imgproc.COLOR_YUV2RGBA_I420, 4);

            return mRgba;
        }
    }

    public Mat onCameraFrame(Mat inputFrame) {
        LoadMatToPipeline(inputFrame);

        if(hand_detected && perspective!=null ) {
            Mat crop = new Mat();
            Imgproc.warpPerspective(inputFrame, crop, perspective, new Size(200,200));
            PreProcess(crop);
            if (effective_value_list.size() >= 128) {
                int pass_time = (int) (System.currentTimeMillis() - old_time);
                afteremd = applyEmd(Floats.toArray(effective_value_list));
                double[] freq = WaveProcess.WaveProcess(afteremd);
                gainEstimate(freq);
                old_time = System.currentTimeMillis();
                effective_value_list.clear();
            }

            Imgproc.circle(inputFrame, rt, 25, new Scalar(255,0,0), 8);
            Imgproc.circle(inputFrame, rb, 25, new Scalar(255,0,0), 8);
            Imgproc.circle(inputFrame, lt, 25, new Scalar(0,255,0), 8);
            Imgproc.circle(inputFrame, lb, 25, new Scalar(0,255,0), 8);
        }else{
            filter.correct(new double[]{80d});
            effective_value_list.clear();
            output_list.clear();
        }
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                int outputValue = 0;
                int divide_count = 0;
                if (output_list.size() >= 3) {
                    Log.i("Output", "Start output");
                    for(int i = output_list.size() - 1; i >= 0; i--){
                        if(output_list.get(i) != 0){
                            outputValue += output_list.get(i);
                            Log.i("Output", Integer.toString(output_list.get(i)));
                            divide_count++;
                        }
                        if(divide_count >= 3)
                            break;
                    }
                    if(divide_count != 0)
                        outputValue /= 3;
                } else {
                    outputValue = 0;
                }

                if (outputValue != 0) {
                    String output_suffix = "";
                    for(int i = divide_count; i < 3; i++){
                        output_suffix = output_suffix + "!";
                    }
                    outputText.setText(Integer.toString(outputValue) + output_suffix);
                } else {
                    outputText.setText("NaN");
                }
                if(afteremd.length != 0){
                    seriesData = new ArrayList<>();
                    for(int i = 0; i < 128; i++){
                        seriesData.add(new ValueDataEntry(i,afteremd[i]));
                    }
                    series1.data(seriesData);
                }
            }
        });
        return inputFrame;
    }

    public void PreProcess(Mat input){
        float current_frame_value = 0;

        colorspace_Count = Core.sumElems(input);

        current_frame_value=(float)colorspace_Count.val[1]/(input.rows()*input.cols());
        current_frame_value=(float)butterworth.filter(current_frame_value);

        float eff_value = current_frame_value + (float)(alpha_value * (current_frame_value - prev_frame_value) / 4);

        effective_value_list.add(eff_value);

        prev_frame_value = current_frame_value;
    }

    public void gainEstimate(double[] freq){
        double prevEstimate = 0;
        if (output_list.size() != 0) {
            for(int i = 0; i < output_list.size(); i++){
                prevEstimate += output_list.get(i);
            }
            prevEstimate /= output_list.size();
        } else {
            prevEstimate = filter.getStateEstimation()[0];
        }
        int o = 0;
        if(freq.length == 0){
            output_list.add(0);
            x.setEntry(1,80d);
        }else{
            double best = 200d;
            for(int i = 0; i < freq.length; i++){
                double ff = freq[i]*60;
                if(Math.abs(ff - prevEstimate) < Math.abs(best)){
                    o = i;
                    best = ff - prevEstimate;
                }
            }
            if(Math.abs(best) > 30){
                output_list.add(0);
            }else{
                filter.predict();
                filter.correct(new double[]{freq[o] * 60});
                output_list.add((int)(filter.getStateEstimation()[0]));
            }
        }
    }

    public float[] applyEmd(float[] input){
        //EMD
        Emd.EmdData emd = Emd.decompose(input,2,5, 0);
        float[] imfout = emd.imfs[1];
        return imfout;
    }

    private boolean allPermissionsGranted() {

        for (String permission: REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

}