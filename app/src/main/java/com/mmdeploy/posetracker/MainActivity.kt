package com.mmdeploy.posetracker

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.blankj.utilcode.util.PathUtils
import com.blankj.utilcode.util.ResourceUtils
import com.mmdeploy.posetracker.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.InternalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable.isActive
import kotlinx.coroutines.cancel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import mmdeploy.Context
import mmdeploy.Device
import mmdeploy.Model
import mmdeploy.PoseTracker
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import java.io.File

class MainActivity : AppCompatActivity() {
    private var poseTracker: PoseTracker? = null
    private var processingJob: Job? = null
    private val coroutineScope = CoroutineScope(Dispatchers.Main + Job())
    private var stateHandle: Long = 0

    // 添加FPS计算相关变量
    private var frameCount = 0
    private var lastFPSComputeTime = System.currentTimeMillis()
    private var currentFPS = 0f

    private var videoCapture: VideoCapture? = null
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        stateHandle = initMMDeploy()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.addVideoButton.setOnClickListener(object : View.OnClickListener {
            override fun onClick(v: View?) {
                if (ContextCompat.checkSelfPermission(
                        this@MainActivity, Manifest.permission
                            .ACCESS_MEDIA_LOCATION
                    ) != PackageManager.PERMISSION_GRANTED
                ) {
                    ActivityCompat.requestPermissions(
                        this@MainActivity,
                        arrayOf<String>(
                            Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.ACCESS_MEDIA_LOCATION,
                            Manifest.permission.READ_MEDIA_IMAGES,
                            Manifest.permission.READ_MEDIA_VIDEO
                        ),
                        1
                    )
                } else {
                    openAlbum()
                }
            }
        })
    }

    private fun openAlbum() {
        val intent = Intent()
        intent.setAction(Intent.ACTION_PICK)
        intent.setType("video/*")
        startActivityForResult(intent, 2)
    }

    private fun initPoseTracker(workDir: String?): Long {
        val detModelPath = "$workDir/rtmdet-nano-ncnn-fp16"
        val poseModelPath = "$workDir/rtmpose-tiny-ncnn-fp16"
        val deviceName = "cpu"
        val deviceID = 0
        val detModel = Model(detModelPath)
        val poseModel = Model(poseModelPath)
        val device = Device(deviceName, deviceID)
        val context = Context()
        context.add(device)
        this.poseTracker = PoseTracker(detModel, poseModel, context)
        val params = this.poseTracker!!.initParams()
        params.detInterval = 5
        params.poseMaxNumBboxes = 6
        val stateHandle = this.poseTracker!!.createState(params)
        return stateHandle
    }

    private fun initMMDeploy(): Long {
        val workDir = (PathUtils.getExternalAppFilesPath() + File.separator
                + "file")
        if (ResourceUtils.copyFileFromAssets("models", workDir)) {
            return initPoseTracker(workDir)
        }
        return -1
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String?>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) openAlbum()
            else Toast.makeText(this@MainActivity, "Invite memory refused.", Toast.LENGTH_SHORT)
                .show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 2) {
            var path: String? = null
            val uri = data!!.data
            System.out.printf("debugging what is uri scheme: %s\n", uri!!.scheme)
            val cursor = contentResolver.query(uri, null, null, null, null)
            if (cursor != null) {
                if (cursor.moveToFirst()) {
                    path = cursor.getString(
                        cursor.getColumnIndex(MediaStore.Images.Media.DATA).coerceAtLeast(0)
                    )
                }
                cursor.close()
            }

            this.videoCapture = VideoCapture(path, Videoio.CAP_ANDROID)
            if (this.videoCapture?.isOpened == true) {
                System.out.printf("failed to open video: %s", path)
            }
            // 取消之前的处理任务
            processingJob?.cancel()

            // 启动新的处理任务
            processingJob = coroutineScope.launch {
                processVideoFrames()
            }
        }
    }

    private suspend fun processVideoFrames() {
        val frame = Mat()

        coroutineScope {
            withContext(Dispatchers.Default) {
                while (currentCoroutineContext().isActive && videoCapture?.read(frame) == true) {
                    if (frame.empty()) continue

                    val processedFrame = processFrame(frame.clone())

                    withContext(Dispatchers.Main) {
                        updateUI(processedFrame)
                    }
                }
            }
        }
    }

    private fun processFrame(frame: Mat): Mat {
        val cvMat = Mat()
        Imgproc.cvtColor(frame, cvMat, Imgproc.COLOR_RGB2BGR)
        val mat = MatUtils.cvMatToMat(cvMat)

        if (stateHandle == -1L) {
            println("State create failed!")
            return frame
        }

        val results = poseTracker!!.apply(stateHandle, mat, -1)
        Draw.drawPoseTrackerResult(frame, results)

        // 计算和更新FPS
        frameCount++
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastFPSComputeTime >= 1000) { // 每秒更新一次FPS
            currentFPS = frameCount * 1000f / (currentTime - lastFPSComputeTime)
            frameCount = 0
            lastFPSComputeTime = currentTime
        }

        // 在画面左上角绘制FPS
        val fpsText = String.format("FPS: %.1f", currentFPS)
        Imgproc.putText(
            frame,
            fpsText,
            org.opencv.core.Point(30.0, 50.0), // 文本位置
            Imgproc.FONT_HERSHEY_SIMPLEX, // 字体
            1.5, // 字体大小
            org.opencv.core.Scalar(0.0, 255.0, 0.0), // 绿色
            2 // 线条粗细
        )

        return frame
    }

    private fun updateUI(frame: Mat) {
        val bitmap = Bitmap.createBitmap(
            frame.width(),
            frame.height(),
            Bitmap.Config.ARGB_8888
        )
        Utils.matToBitmap(frame, bitmap)
        binding.videoFrameView.setImageBitmap(bitmap)
    }

    override fun onDestroy() {
        super.onDestroy()
        processingJob?.cancel()
        coroutineScope.cancel()
        videoCapture?.release()
    }

    companion object {
        init {
            System.loadLibrary("opencv_java4")
        }
    }
}
