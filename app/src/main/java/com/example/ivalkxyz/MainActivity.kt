package com.example.ivalkxyz

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Rect
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Paint
import android.graphics.YuvImage
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.AttributeSet
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import android.widget.ImageView
import android.widget.TextView
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.NativeCameraView.TAG
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.features2d.SIFT
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.Features2d
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var viewFinder: PreviewView
    //private lateinit var resultImageView: ImageView
    private lateinit var infoTextView: TextView
    //private lateinit var pointCloudView: PointCloudView

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        //resultImageView = findViewById(R.id.resultImageView)
        infoTextView = findViewById(R.id.infoTextView)
        //pointCloudView = findViewById(R.id.pointCloudView)

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "Unable to load OpenCV", Toast.LENGTH_LONG).show()
            return
        }

        if (allPermissionsGranted()) {

            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )

        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewFinder.surfaceProvider)
                }





            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, FeatureMatchingAnalyzer())
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    inner class FeatureMatchingAnalyzer : ImageAnalysis.Analyzer {
        private val sift = SIFT.create()
        private val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE)
        private var previousFrame: Mat? = null
        private var previousKeypoints: MatOfKeyPoint? = null
        private var previousDescriptors: Mat? = null


        @SuppressLint("SetTextI18n")
        override fun analyze(image: ImageProxy) {
            val currentFrame = image.toBitmap().toMat()
            val grayFrame = Mat()

            Imgproc.cvtColor(currentFrame, grayFrame, Imgproc.COLOR_RGBA2GRAY)

            val currentKeypoints = MatOfKeyPoint()
            //val previouskeypoints = MatOfKeyPoint()

            val keypoints1 = MatOfKeyPoint()

            val keypoints2 = MatOfKeyPoint()

            val pointCloudView = findViewById<PointCloudView>(R.id.pointCloudView)

            val currentDescriptors = Mat()
            sift.detectAndCompute(grayFrame, Mat(), currentKeypoints, currentDescriptors)

            if (previousFrame != null && previousKeypoints != null && previousDescriptors != null) {
                // Match features
                val matches = MatOfDMatch()
                matcher.match(previousDescriptors, currentDescriptors, matches)

                // Filter good matches
                val goodMatches = matches.toArray().sortedBy { it.distance }.take(50)
                //println("mat")


                // Extract matched keypoints
                val listOfMatches = matches.toList()
                println("list fo mathed points :$goodMatches")
                println("list of previous points:${previousKeypoints!!.toList()}")
                println("list of current points: ${currentKeypoints.toList()}")
                val srcPoints = mutableListOf<Point>()
                val dstPoints = mutableListOf<Point>()
                for (match in goodMatches) {
                    srcPoints.add(previousKeypoints!!.toList()[match.queryIdx].pt)
                    dstPoints.add(currentKeypoints.toList()[match.trainIdx].pt)
                }
                println("points :")
                println("src points : $srcPoints")
                println("dst points : $dstPoints")



                // Convert points to MatOfPoint2f
                val srcMatOfPoint2f = MatOfPoint2f(*srcPoints.toTypedArray())
                val dstMatOfPoint2f = MatOfPoint2f(*dstPoints.toTypedArray())
//                println("src points : $srcMatOfPoint2f")
//                println("dst points : $dstMatOfPoint2f")

                if (srcMatOfPoint2f.empty() || dstMatOfPoint2f.empty()) {
                    println("Source or Destination points matrix is empty.")
                    return
                }

                //val intristic parameters

                val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

                val cameraIdList = cameraManager.cameraIdList
                val cameraId = 0
                val characteristics = cameraManager.getCameraCharacteristics(cameraId.toString())

                // Get focal lengths
                val focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)

                // Ensure focalLengths is not null and get the focal length values
                val focalLengthX = focalLengths?.getOrNull(0) ?: 0f
                val focalLengthY = if ((focalLengths?.size ?: 0) > 1) focalLengths?.get(1) else focalLengthX

                // Get sensor size
                val sensorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)

                // Principal point (assuming it's at the center of the sensor)
                val cx = sensorSize?.width?.div(2) ?: 0f
                val cy = sensorSize?.height?.div(2) ?: 0f





                // Find the Fundamental Matrix using the 8-point algorithm with RANSAC
                val mask = Mat()
                val fundamentalMatrix = Calib3d.findFundamentalMat(
                    srcMatOfPoint2f,
                    dstMatOfPoint2f,
                    Calib3d.FM_RANSAC,
                    3.0,  // Distance to epipolar line
                    0.99, // Confidence level
                    mask
                )

                // Construct intrinsic matrix K
                val K = arrayOf(
                    floatArrayOf(focalLengthX, 0f, cx),
                    focalLengthY?.let { floatArrayOf(0f, it, cy) },
                    floatArrayOf(0f, 0f, 1f)
                )

                // Transpose of intrinsic matrix K'
                val K_transpose = arrayOf(
                    floatArrayOf(focalLengthX, 0f, 0f),
                    focalLengthY?.let { floatArrayOf(0f, it, 0f) },
                    floatArrayOf(cx, cy, 1f)
                )

                // Check if the matrix is empty
                if (fundamentalMatrix.empty()) {
                    println("Fundamental matrix is empty.")



                    //println("intristic matrix $intristic")
                } else {
                    println("Fundamental Matrix: \n$fundamentalMatrix")
                    println("focal length X: $focalLengthX")
                    println("focal length Y: $focalLengthY")
                    println("principle point x: $cx")
                    println("principle point y: $cy")
                    println("matrix :$K")
                    println("matrix transpose: $K_transpose")

                    println("F dimensions: ${fundamentalMatrix.rows()} x ${fundamentalMatrix.cols()}")
                    println("K dimensions: ${K.size} x ${K[0]?.size}")
                    println("K_transpose dimensions: ${K_transpose.size} x ${K_transpose[0]?.size}")

                    // Usage
                    val F: Mat =fundamentalMatrix  // Your fundamental matrix
                    val fx = 3055.22
                    val fy = 3044.78
                    val cx = 1590.50
                    val cy = 2077.31

                    try {
                        val E = computeEssentialMatrix(F, fx, fy, cx, cy)
                        // Decompose essential matrix
                        val R1 = Mat()
                        val R2 = Mat()
                        val t = Mat()


                        println("Essential Matrix:")
                        println(E.dump())

                        Calib3d.decomposeEssentialMat(E, R1, R2, t)

                        println("\nRotation Matrix 1:")
                        println(R1.dump())

                        println("\nRotation Matrix 2:")
                        println(R2.dump())

                        println("\nTranslation Vector:")
                        println(t.dump())

                        // Choose the correct rotation matrix
                        val R = selectCorrectRotation(R1, R2, t, srcPoints, dstPoints)

                        println("correct rotation matrix:")
                        println(R.dump())

                        // Triangulate points
                        val points3D = triangulatePoints(R, Mat.eye(3, 3, CvType.CV_64F), t, srcPoints, dstPoints)

                        println("points 3d")
                        println(points3D)


                        // In your MainActivity or wherever you're processing the point cloud
                        val pointCloudProcessor = PointCloudProcessor()
                        val refinedPoints3D = pointCloudProcessor.refinePointCloud(points3D)

                        // In your activity
                        //val pointCloudView = PointCloudView(this)
                        //pointCloudView.points = refinedPoints3D
                        println("refined matrix :")
                        println(refinedPoints3D)
                        pointCloudView.points = refinedPoints3D

                        // Save the images to external storage
                        saveComparedFrames(previousFrame!!.toBitmap(), grayFrame.toBitmap(),refinedPoints3D, pointCloudView)

                        println(refinedPoints3D.size)










                        // Don't forget to release the matrices when done
                        F.release()
                        E.release()
                        R1.release()
                        R2.release()
                        t.release()
                        R.release()
                        // Use E...
                    } catch (e: CvException) {
                        Log.e("OpenCV Error", "Error in computeEssentialMatrix: ${e.message}")
                        // Handle the error...
                    }




                }


//


                // Draw matches
                val resultMat = Mat()
                Features2d.drawMatches(
                    previousFrame,
                    previousKeypoints,
                    grayFrame,
                    currentKeypoints,
                    MatOfDMatch().apply { fromList(goodMatches) },
                    resultMat
                )

                val resultBitmap = resultMat.toBitmap()

                runOnUiThread {
                    //resultImageView.setImageBitmap(resultBitmap)
//                    infoTextView.text = "Total keypoints: ${currentKeypoints.rows()}\n" +
//                            "Matched keypoints: ${goodMatches.size}\n" +
//                            "Best match distance: %.2f".format(goodMatches.firstOrNull()?.distance ?: 0f)


                }
            } else {
                runOnUiThread {
                    //resultImageView.setImageBitmap(currentFrame.toBitmap())
                    infoTextView.text = "Initializing... Keypoints: ${currentKeypoints.rows()}"
                }
            }

            // Update previous frame data
            previousFrame = grayFrame
            previousKeypoints = currentKeypoints
            previousDescriptors = currentDescriptors

            image.close()
        }

        private fun saveImageToExternalStorage(bitmap: Bitmap, fileName: String) {
            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
            }

            val resolver = applicationContext.contentResolver
            val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

            if (uri != null) {
                try {
                    val outputStream = resolver.openOutputStream(uri)
                    if (outputStream != null) {
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                    }
                    outputStream?.flush()
                    outputStream?.close()
                    Log.d("ImageSave", "Image saved to external storage: $fileName")
                } catch (e: IOException) {
                    Log.e("ImageSave", "Error saving image to external storage: ${e.message}")
                }
            } else {
                Log.e("ImageSave", "Error getting URI for external storage")
            }
        }

        private fun generateUniqueFileName(prefix: String): String {
            val timestamp = System.currentTimeMillis()
            return "$prefix$timestamp.jpg"
        }

        private fun isFileExists(fileName: String): Boolean {
            val resolver = applicationContext.contentResolver
            val projection = arrayOf(MediaStore.Images.Media._ID)
            val selection = "${MediaStore.Images.Media.DISPLAY_NAME} = ?"
            val selectionArgs = arrayOf(fileName)
            val cursor = resolver.query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, projection, selection, selectionArgs, null)
            val exists = (cursor?.count ?: 0) > 0
            cursor?.close()
            return exists
        }

        fun saveComparedFrames(previousFrame: Bitmap, currentFrame: Bitmap, refinedPoints3D: List<Point3>, pointCloudView: PointCloudView?) {
            val previousFrameFileName = generateUniqueFileName("previous_frame")
            val currentFrameFileName = generateUniqueFileName("current_frame")
            val pointsFrameFileName = generateUniqueFileName("3d_points_frame")
            val comparedFramesFileName = generateUniqueFileName("compared_frames")

            saveImageToExternalStorage(previousFrame, previousFrameFileName)
            saveImageToExternalStorage(currentFrame, currentFrameFileName)

            // Concatenate the previous frame and the current frame
            val concatedFrame = concatenateFrames(previousFrame, currentFrame)

            // Perform feature matching on the concatenated frame
            val resultMat = performFeatureMatching(previousFrame, concatedFrame)

            // Convert the result mat to a bitmap and save it to external storage
            val resultBitmap2 = resultMat.toBitmap()
            saveImageToExternalStorage(resultBitmap2, comparedFramesFileName)

            saveImageToExternalStorage(previousFrame, previousFrameFileName)
            saveImageToExternalStorage(currentFrame, currentFrameFileName)
            // Render the 3D points on the current frame
            val currentFrameCopy = currentFrame.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(currentFrameCopy)
            val paint = Paint().apply {
                color = Color.RED
                strokeWidth = 2f
                isAntiAlias = true
            }
//            paint.color = Color.RED
//            paint.strokeWidth = 2f
            for (point3D in refinedPoints3D) {
                val x = (point3D.x + 1) * currentFrameCopy.width / 2
                val y = (1 - point3D.y) * currentFrameCopy.height / 2
                canvas.drawCircle(x.toFloat(), y.toFloat(), 5f, paint)
            }
            saveImageToExternalStorage(currentFrameCopy, pointsFrameFileName)

            // Pass the original camera frame to the PointCloudView
            pointCloudView?.cameraFrame = currentFrame
            pointCloudView?.points = refinedPoints3D
        }

        private fun concatenateFrames(frame1: Bitmap, frame2: Bitmap): Bitmap {
            val width = frame1.width + frame2.width
            val height = maxOf(frame1.height, frame2.height)
            val concatedFrame = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(concatedFrame)

            canvas.drawBitmap(frame1, 0f, 0f, null)
            canvas.drawBitmap(frame2, frame1.width.toFloat(), 0f, null)

            return concatedFrame
        }

        private fun performFeatureMatching(frame1: Bitmap, frame2: Bitmap): Mat {
            val mat1 = frame1.toMat()
            val mat2 = frame2.toMat()

            val grayMat1 = Mat()
            val grayMat2 = Mat()
            Imgproc.cvtColor(mat1, grayMat1, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.cvtColor(mat2, grayMat2, Imgproc.COLOR_RGBA2GRAY)

            val sift = SIFT.create()
            val keypoints1 = MatOfKeyPoint()
            val keypoints2 = MatOfKeyPoint()
            val descriptors1 = Mat()
            val descriptors2 = Mat()
            sift.detectAndCompute(grayMat1, Mat(), keypoints1, descriptors1)
            sift.detectAndCompute(grayMat2, Mat(), keypoints2, descriptors2)

            val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE)
            val matches = MatOfDMatch()
            matcher.match(descriptors1, descriptors2, matches)

            val resultMat = Mat()
            Features2d.drawMatches(mat1, keypoints1, mat2, keypoints2, matches, resultMat)

            return resultMat
        }

        private fun triangulatePoints(R1: Mat, R2: Mat, t: Mat, points1: List<Point>, points2: List<Point>): List<Point3> {
            val K = Mat(3, 3, CvType.CV_64F)
            K.put(0, 0, 5.56, 0.0, 4.1285, 0.0, 5.56, 3.0965, 0.0, 0.0, 1.0)

            val P1 = Mat(3, 4, CvType.CV_64F)
            K.copyTo(P1.submat(0, 3, 0, 3))
            P1.put(0, 3, 0.0, 0.0, 0.0)

            val P2 = Mat(3, 4, CvType.CV_64F)

            // Check matrix dimensions before multiplication
            if (K.cols() != R1.rows() || K.cols() != t.rows()) {
                Log.e("Matrix Error", "Incompatible matrix dimensions in triangulatePoints")
                return emptyList() // Return an empty list or handle the error as appropriate
            }

            Core.gemm(K, R1, 1.0, Mat(), 0.0, P2.submat(0, 3, 0, 3))
            Core.gemm(K, t, 1.0, Mat(), 0.0, P2.submat(0, 3, 3, 4))

            val points1Mat = MatOfPoint2f()
            points1Mat.fromList(points1)

            val points2Mat = MatOfPoint2f()
            points2Mat.fromList(points2)

            val homogeneousPoints = Mat()
            Calib3d.triangulatePoints(P1, P2, points1Mat, points2Mat, homogeneousPoints)

            val points3D = mutableListOf<Point3>()
            for (i in 0 until homogeneousPoints.cols()) {
                val col = homogeneousPoints.col(i)
                val x = col.get(0, 0)[0] / col.get(3, 0)[0]
                val y = col.get(1, 0)[0] / col.get(3, 0)[0]
                val z = col.get(2, 0)[0] / col.get(3, 0)[0]
                points3D.add(Point3(x, y, z))
            }

            P1.release()
            P2.release()
            K.release()
            points1Mat.release()
            points2Mat.release()
            homogeneousPoints.release()

            return points3D
        }

        private fun selectCorrectRotation(R1: Mat, R2: Mat, t: Mat, points1: List<Point>, points2: List<Point>): Mat {
            val K = Mat(3, 3, CvType.CV_64F)
            K.put(0, 0, 5.56, 0.0, 4.1285, 0.0, 5.56, 3.0965, 0.0, 0.0, 1.0)

            val P1 = Mat(3, 4, CvType.CV_64F)
            K.copyTo(P1.submat(0, 3, 0, 3))
            P1.put(0, 3, 0.0, 0.0, 0.0)

            val rotations = listOf(R1, R2)
            var bestR: Mat? = null
            var maxPositive = 0

            for (R in rotations) {
                val P2 = Mat(3, 4, CvType.CV_64F)

                // Check matrix dimensions before multiplication
                if (K.cols() != R.rows() || K.cols() != t.rows()) {
                    Log.e("Matrix Error", "Incompatible matrix dimensions in selectCorrectRotation")
                    continue // Skip this iteration
                }

                Core.gemm(K, R, 1.0, Mat(), 0.0, P2.submat(0, 3, 0, 3))
                Core.gemm(K, t, 1.0, Mat(), 0.0, P2.submat(0, 3, 3, 4))

                val points1Mat = MatOfPoint2f()
                points1Mat.fromList(points1)

                val points2Mat = MatOfPoint2f()
                points2Mat.fromList(points2)

                val homogeneousPoints = Mat()
                Calib3d.triangulatePoints(P1, P2, points1Mat, points2Mat, homogeneousPoints)

                var positiveCount = 0

                for (i in 0 until homogeneousPoints.cols()) {
                    val col = homogeneousPoints.col(i)
                    val x = col.get(0, 0)[0]
                    val y = col.get(1, 0)[0]
                    val z = col.get(2, 0)[0]
                    val w = col.get(3, 0)[0]

                    if (w != 0.0) {
                        val point3D = Point3(x / w, y / w, z / w)

                        // Check if point is in front of both cameras
                        if (isInFrontOfCamera(point3D, Mat.eye(3, 3, CvType.CV_64F), Mat.zeros(3, 1, CvType.CV_64F)) &&
                            isInFrontOfCamera(point3D, R, t)) {
                            positiveCount++
                        }
                    }
                }

                if (positiveCount > maxPositive) {
                    maxPositive = positiveCount
                    bestR = R
                }

                P2.release()
                points1Mat.release()
                points2Mat.release()
                homogeneousPoints.release()
            }

            K.release()
            P1.release()

            return bestR ?: R1 // Return R1 as a fallback if no good solution is found
        }

        private fun isInFrontOfCamera(point3D: Point3, R: Mat, t: Mat): Boolean {
            val p = Mat(3, 1, CvType.CV_64F)
            p.put(0, 0, point3D.x, point3D.y, point3D.z)

            val transformed = Mat()
            Core.gemm(R, p, 1.0, t, 1.0, transformed)

            val result = transformed.get(2, 0)[0] > 0

            p.release()
            transformed.release()

            return result
        }

        fun computeEssentialMatrix(F: Mat, fx: Double, fy: Double, cx: Double, cy: Double): Mat {
            // Create the intrinsic matrix K
            val K = Mat(3, 3, CvType.CV_64FC1)
            K.put(0, 0, fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)

            // Compute K transpose
            val K_transpose = Mat()
            Core.transpose(K, K_transpose)

            // Check matrix dimensions before multiplication
            if (F.cols() != K.rows() || K_transpose.cols() != F.rows()) {
                Log.e("Matrix Error", "Incompatible matrix dimensions for Essential Matrix computation")
                return Mat() // Return an empty matrix or handle the error as appropriate
            }

            // Compute E = K' * F * K
            val temp = Mat()
            val E = Mat()
            Core.gemm(F, K, 1.0, Mat(), 0.0, temp)
            Core.gemm(K_transpose, temp, 1.0, Mat(), 0.0, E)

            // Release temporary matrices
            K.release()
            K_transpose.release()
            temp.release()

            return E
        }






        private fun ImageProxy.toBitmap(): Bitmap {
            val yBuffer = planes[0].buffer
            val uBuffer = planes[1].buffer
            val vBuffer = planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
            val imageBytes = out.toByteArray()
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        }




        private fun Bitmap.toMat(): Mat {
            val mat = Mat(height, width, CvType.CV_8UC4)
            Utils.bitmapToMat(this, mat)
            return mat
        }

        private fun Mat.toBitmap(): Bitmap {
            val resultBitmap = Bitmap.createBitmap(cols(), rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(this, resultBitmap)
            return resultBitmap
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}



class PointCloudProcessor {
    fun refinePointCloud(points3D: List<Point3>): List<Point3> {
        // Filter out points with negative z-values (behind the camera)
        val filteredPoints = points3D.filter { it.z > 0 }

        // Check if filteredPoints is empty
        if (filteredPoints.isEmpty()) {
            return emptyList()
        }

        // Remove outliers (you might need to adjust the threshold)
        val mean = calculateMean(filteredPoints)
        val stdDev = calculateStdDev(filteredPoints, mean)
        val threshold = 2.0 // Adjust this value as needed

        return filteredPoints.filter { point ->
            val distance = calculateDistance(point, mean)
            distance < threshold * stdDev
        }
    }

    private fun calculateMean(points: List<Point3>): Point3 {

        if (points.isEmpty()) {
            return Point3(0.0, 0.0, 0.0)
        }

        val sum = points.reduce { acc, point ->
            Point3(acc.x + point.x, acc.y + point.y, acc.z + point.z)
        }
        return Point3(sum.x / points.size, sum.y / points.size, sum.z / points.size)
    }

    private fun calculateStdDev(points: List<Point3>, mean: Point3): Double {

        if (points.isEmpty()) {
            return 0.0
        }

        val squaredDiffs = points.map { point ->
            val diff = Point3(point.x - mean.x, point.y - mean.y, point.z - mean.z)
            diff.x * diff.x + diff.y * diff.y + diff.z * diff.z
        }
        val variance = squaredDiffs.average()
        return Math.sqrt(variance)
    }

    private fun calculateDistance(p1: Point3, p2: Point3): Double {
        val dx = p1.x - p2.x
        val dy = p1.y - p2.y
        val dz = p1.z - p2.z
        return Math.sqrt(dx * dx + dy * dy + dz * dz)
    }
}
class PointCloudView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    var points: List<Point3> = emptyList()
        set(value) {
            field = value
            invalidate()
        }
    var cameraFrame: Bitmap? = null
        set(value) {
            field = value
            invalidate()
        }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        cameraFrame?.let { frame ->
            canvas.drawBitmap(frame, 0f, 0f, null)
        }

        if (points.isEmpty()) return

        val maxDistance = points.maxOf { sqrt(it.x * it.x + it.y * it.y + it.z * it.z) }
        val minZ = points.minOf { it.z }
        val maxZ = points.maxOf { it.z }

        points.forEach { point ->
            // Map 3D coordinates to 2D screen coordinates
            val x = (point.x + 1) * width / 2
            val y = (1 - point.y) * height / 2

            // Calculate distance from origin (0,0,0)
            val distance = sqrt(point.x * point.x + point.y * point.y + point.z * point.z)

            // Normalize distance to [0, 1] range
            val normalizedDistance = distance / maxDistance

            // Create a color gradient from blue (cold, far) to red (hot, close)
            val hue = (240 - normalizedDistance * 240).toFloat()
            val color = Color.HSVToColor(floatArrayOf(hue, 1f, 1f))

            // Set point size based on z-value (depth)
            val normalizedZ = (point.z - minZ) / (maxZ - minZ)
            val pointSize = 2f + normalizedZ * 8f  // Points will range from 2 to 10 in size

            paint.color = color
            canvas.drawCircle(x.toFloat(), y.toFloat(), pointSize.toFloat(), paint)
        }
    }
}


//package com.example.ivalkxyz
//
//import android.Manifest
//import android.annotation.SuppressLint
//import android.content.Context
//import android.content.pm.PackageManager
//import android.graphics.Bitmap
//import android.graphics.Rect
//import android.graphics.BitmapFactory
//import android.graphics.Canvas
//import android.graphics.Color
//import android.graphics.ImageFormat
//import android.graphics.Paint
//import android.graphics.YuvImage
//import android.hardware.camera2.CameraCharacteristics
//import android.hardware.camera2.CameraManager
//import android.os.Bundle
//import android.util.AttributeSet
//import android.util.Log
//import android.view.View
//import android.widget.Toast
//import androidx.appcompat.app.AppCompatActivity
//import androidx.camera.core.CameraSelector
//import androidx.camera.core.ImageAnalysis
//import androidx.camera.core.ImageProxy
//import androidx.camera.core.Preview
//import android.widget.ImageView
//import android.widget.TextView
//import androidx.camera.lifecycle.ProcessCameraProvider
//import androidx.camera.view.PreviewView
//import androidx.core.app.ActivityCompat
//import androidx.core.content.ContextCompat
//import org.opencv.android.NativeCameraView.TAG
//import org.opencv.android.OpenCVLoader
//import org.opencv.android.Utils
//import org.opencv.calib3d.Calib3d
//import org.opencv.core.*
//import org.opencv.features2d.SIFT
//import org.opencv.features2d.DescriptorMatcher
//import org.opencv.features2d.Features2d
//import org.opencv.imgproc.Imgproc
//import java.io.ByteArrayOutputStream
//import java.util.concurrent.ExecutorService
//import java.util.concurrent.Executors
//import kotlin.math.sqrt
//
//class MainActivity : AppCompatActivity() {
//    private lateinit var cameraExecutor: ExecutorService
//    private lateinit var viewFinder: PreviewView
//    //private lateinit var resultImageView: ImageView
//    private lateinit var infoTextView: TextView
//    //private lateinit var pointCloudView: PointCloudView
//
//    @SuppressLint("MissingInflatedId")
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_main)
//
//        viewFinder = findViewById(R.id.viewFinder)
//        //resultImageView = findViewById(R.id.resultImageView)
//        infoTextView = findViewById(R.id.infoTextView)
//        //pointCloudView = findViewById(R.id.pointCloudView)
//
//        if (!OpenCVLoader.initDebug()) {
//            Toast.makeText(this, "Unable to load OpenCV", Toast.LENGTH_LONG).show()
//            return
//        }
//
//        if (allPermissionsGranted()) {
//
//            startCamera()
//        } else {
//            ActivityCompat.requestPermissions(
//                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
//            )
//
//        }
//
//        cameraExecutor = Executors.newSingleThreadExecutor()
//    }
//
//    private fun startCamera() {
//        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
//
//        cameraProviderFuture.addListener({
//            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
//
//            val preview = Preview.Builder()
//                .build()
//                .also {
//                    it.setSurfaceProvider(viewFinder.surfaceProvider)
//                }
//
//
//
//
//
//            val imageAnalyzer = ImageAnalysis.Builder()
//                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                .build()
//                .also {
//                    it.setAnalyzer(cameraExecutor, FeatureMatchingAnalyzer())
//                }
//
//            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
//
//            try {
//                cameraProvider.unbindAll()
//                cameraProvider.bindToLifecycle(
//                    this, cameraSelector, preview, imageAnalyzer
//                )
//            } catch (exc: Exception) {
//                Log.e(TAG, "Use case binding failed", exc)
//            }
//
//        }, ContextCompat.getMainExecutor(this))
//    }
//
//    inner class FeatureMatchingAnalyzer : ImageAnalysis.Analyzer {
//        private val sift = SIFT.create()
//        private val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE)
//        private var previousFrame: Mat? = null
//        private var previousKeypoints: MatOfKeyPoint? = null
//        private var previousDescriptors: Mat? = null
//
//
//        @SuppressLint("SetTextI18n")
//        override fun analyze(image: ImageProxy) {
//            val currentFrame = image.toBitmap().toMat()
//            val grayFrame = Mat()
//
//            Imgproc.cvtColor(currentFrame, grayFrame, Imgproc.COLOR_RGBA2GRAY)
//
//            val currentKeypoints = MatOfKeyPoint()
//            //val previouskeypoints = MatOfKeyPoint()
//
//            val keypoints1 = MatOfKeyPoint()
//
//            val keypoints2 = MatOfKeyPoint()
//
//            val pointCloudView = findViewById<PointCloudView>(R.id.pointCloudView)
//
//            val currentDescriptors = Mat()
//            sift.detectAndCompute(grayFrame, Mat(), currentKeypoints, currentDescriptors)
//
//            if (previousFrame != null && previousKeypoints != null && previousDescriptors != null) {
//                // Match features
//                val matches = MatOfDMatch()
//                matcher.match(previousDescriptors, currentDescriptors, matches)
//
//                // Filter good matches
//                val goodMatches = matches.toArray().sortedBy { it.distance }.take(50)
//                println("mat")
//
//
//                // Extract matched keypoints
//                val listOfMatches = matches.toList()
//                println("list fo mathed points :$goodMatches")
//                println("list of previous points:${previousKeypoints!!.toList()}")
//                println("list of current points: ${currentKeypoints.toList()}")
//                val srcPoints = mutableListOf<Point>()
//                val dstPoints = mutableListOf<Point>()
//                for (match in goodMatches) {
//                    srcPoints.add(previousKeypoints!!.toList()[match.queryIdx].pt)
//                    dstPoints.add(currentKeypoints.toList()[match.trainIdx].pt)
//                }
//                println("points :")
//                println(srcPoints)
//                println(dstPoints)
//
//
//
//                // Convert points to MatOfPoint2f
//                val srcMatOfPoint2f = MatOfPoint2f(*srcPoints.toTypedArray())
//                val dstMatOfPoint2f = MatOfPoint2f(*dstPoints.toTypedArray())
//                println("src points : $srcMatOfPoint2f")
//                println("src points : $dstMatOfPoint2f")
//
//                if (srcMatOfPoint2f.empty() || dstMatOfPoint2f.empty()) {
//                    println("Source or Destination points matrix is empty.")
//                    return
//                }
//
//                //val intristic parameters
//
//                val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
//
//                val cameraIdList = cameraManager.cameraIdList
//                val cameraId = 0
//                val characteristics = cameraManager.getCameraCharacteristics(cameraId.toString())
//
//                // Get focal lengths
//                val focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
//
//                // Ensure focalLengths is not null and get the focal length values
//                val focalLengthX = focalLengths?.getOrNull(0) ?: 0f
//                val focalLengthY = if ((focalLengths?.size ?: 0) > 1) focalLengths?.get(1) else focalLengthX
//
//                // Get sensor size
//                val sensorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
//
//                // Principal point (assuming it's at the center of the sensor)
//                val cx = sensorSize?.width?.div(2) ?: 0f
//                val cy = sensorSize?.height?.div(2) ?: 0f
//
//
//
//
//
//                // Find the Fundamental Matrix using the 8-point algorithm with RANSAC
//                val mask = Mat()
//                val fundamentalMatrix = Calib3d.findFundamentalMat(
//                    srcMatOfPoint2f,
//                    dstMatOfPoint2f,
//                    Calib3d.FM_RANSAC,
//                    3.0,  // Distance to epipolar line
//                    0.99, // Confidence level
//                    mask
//                )
//
//                // Construct intrinsic matrix K
//                val K = arrayOf(
//                    floatArrayOf(focalLengthX, 0f, cx),
//                    focalLengthY?.let { floatArrayOf(0f, it, cy) },
//                    floatArrayOf(0f, 0f, 1f)
//                )
//
//                // Transpose of intrinsic matrix K'
//                val K_transpose = arrayOf(
//                    floatArrayOf(focalLengthX, 0f, 0f),
//                    focalLengthY?.let { floatArrayOf(0f, it, 0f) },
//                    floatArrayOf(cx, cy, 1f)
//                )
//
//                // Check if the matrix is empty
//                if (fundamentalMatrix.empty()) {
//                    println("Fundamental matrix is empty.")
//
//
//
//                    //println("intristic matrix $intristic")
//                } else {
//                    println("Fundamental Matrix: \n$fundamentalMatrix")
//                    println("focal length X: $focalLengthX")
//                    println("focal length Y: $focalLengthY")
//                    println("principle point x: $cx")
//                    println("principle point y: $cy")
//                    println("matrix :$K")
//                    println("matrix transpose: $K_transpose")
//
//                    println("F dimensions: ${fundamentalMatrix.rows()} x ${fundamentalMatrix.cols()}")
//                    println("K dimensions: ${K.size} x ${K[0]?.size}")
//                    println("K_transpose dimensions: ${K_transpose.size} x ${K_transpose[0]?.size}")
//
//                    // Usage
//                    val F: Mat =fundamentalMatrix  // Your fundamental matrix
//                    val fx = 5.56
//                    val fy = 5.56
//                    val cx = 4.1285
//                    val cy = 3.0965
//
//                    try {
//                        val E = computeEssentialMatrix(F, fx, fy, cx, cy)
//                        // Decompose essential matrix
//                        val R1 = Mat()
//                        val R2 = Mat()
//                        val t = Mat()
//
//
//                        println("Essential Matrix:")
//                        println(E.dump())
//
//                        Calib3d.decomposeEssentialMat(E, R1, R2, t)
//
//                        println("\nRotation Matrix 1:")
//                        println(R1.dump())
//
//                        println("\nRotation Matrix 2:")
//                        println(R2.dump())
//
//                        println("\nTranslation Vector:")
//                        println(t.dump())
//
//                        // Choose the correct rotation matrix
//                        val R = selectCorrectRotation(R1, R2, t, srcPoints, dstPoints)
//
//                        println("correct rotation matrix:")
//                        println(R.dump())
//
//                        // Triangulate points
//                        val points3D = triangulatePoints(R, Mat.eye(3, 3, CvType.CV_64F), t, srcPoints, dstPoints)
//
//                        println("points 3d")
//                        println(points3D)
//
//
//                        // In your MainActivity or wherever you're processing the point cloud
//                        val pointCloudProcessor = PointCloudProcessor()
//                        val refinedPoints3D = pointCloudProcessor.refinePointCloud(points3D)
//
//                        // In your activity
//                        //val pointCloudView = PointCloudView(this)
//                        //pointCloudView.points = refinedPoints3D
//                        println("refined matrix :")
//                        println(refinedPoints3D)
//                        pointCloudView.points = refinedPoints3D
//
//                        println(refinedPoints3D.size)
//
//
//
//
//
//
//
//
//
//
//                        // Don't forget to release the matrices when done
//                        F.release()
//                        E.release()
//                        R1.release()
//                        R2.release()
//                        t.release()
//                        R.release()
//                        // Use E...
//                    } catch (e: CvException) {
//                        Log.e("OpenCV Error", "Error in computeEssentialMatrix: ${e.message}")
//                        // Handle the error...
//                    }
//
//
//
//
//                }
//
//
////
//
//
//                // Draw matches
//                val resultMat = Mat()
//                Features2d.drawMatches(
//                    previousFrame,
//                    previousKeypoints,
//                    grayFrame,
//                    currentKeypoints,
//                    MatOfDMatch().apply { fromList(goodMatches) },
//                    resultMat
//                )
//
//                val resultBitmap = resultMat.toBitmap()
//
//                runOnUiThread {
//                    //resultImageView.setImageBitmap(resultBitmap)
////                    infoTextView.text = "Total keypoints: ${currentKeypoints.rows()}\n" +
////                            "Matched keypoints: ${goodMatches.size}\n" +
////                            "Best match distance: %.2f".format(goodMatches.firstOrNull()?.distance ?: 0f)
//
//
//                }
//            } else {
//                runOnUiThread {
//                    //resultImageView.setImageBitmap(currentFrame.toBitmap())
//                    infoTextView.text = "Initializing... Keypoints: ${currentKeypoints.rows()}"
//                }
//            }
//
//            // Update previous frame data
//            previousFrame = grayFrame
//            previousKeypoints = currentKeypoints
//            previousDescriptors = currentDescriptors
//
//            image.close()
//        }
//
//        private fun triangulatePoints(R1: Mat, R2: Mat, t: Mat, points1: List<Point>, points2: List<Point>): List<Point3> {
//            val K = Mat(3, 3, CvType.CV_64F)
//            K.put(0, 0, 5.56, 0.0, 4.1285, 0.0, 5.56, 3.0965, 0.0, 0.0, 1.0)
//
//            val P1 = Mat(3, 4, CvType.CV_64F)
//            K.copyTo(P1.submat(0, 3, 0, 3))
//            P1.put(0, 3, 0.0, 0.0, 0.0)
//
//            val P2 = Mat(3, 4, CvType.CV_64F)
//
//            // Check matrix dimensions before multiplication
//            if (K.cols() != R1.rows() || K.cols() != t.rows()) {
//                Log.e("Matrix Error", "Incompatible matrix dimensions in triangulatePoints")
//                return emptyList() // Return an empty list or handle the error as appropriate
//            }
//
//            Core.gemm(K, R1, 1.0, Mat(), 0.0, P2.submat(0, 3, 0, 3))
//            Core.gemm(K, t, 1.0, Mat(), 0.0, P2.submat(0, 3, 3, 4))
//
//            val points1Mat = MatOfPoint2f()
//            points1Mat.fromList(points1)
//
//            val points2Mat = MatOfPoint2f()
//            points2Mat.fromList(points2)
//
//            val homogeneousPoints = Mat()
//            Calib3d.triangulatePoints(P1, P2, points1Mat, points2Mat, homogeneousPoints)
//
//            val points3D = mutableListOf<Point3>()
//            for (i in 0 until homogeneousPoints.cols()) {
//                val col = homogeneousPoints.col(i)
//                val x = col.get(0, 0)[0] / col.get(3, 0)[0]
//                val y = col.get(1, 0)[0] / col.get(3, 0)[0]
//                val z = col.get(2, 0)[0] / col.get(3, 0)[0]
//                points3D.add(Point3(x, y, z))
//            }
//
//            P1.release()
//            P2.release()
//            K.release()
//            points1Mat.release()
//            points2Mat.release()
//            homogeneousPoints.release()
//
//            return points3D
//        }
//
//        private fun selectCorrectRotation(R1: Mat, R2: Mat, t: Mat, points1: List<Point>, points2: List<Point>): Mat {
//            val K = Mat(3, 3, CvType.CV_64F)
//            K.put(0, 0, 5.56, 0.0, 4.1285, 0.0, 5.56, 3.0965, 0.0, 0.0, 1.0)
//
//            val P1 = Mat(3, 4, CvType.CV_64F)
//            K.copyTo(P1.submat(0, 3, 0, 3))
//            P1.put(0, 3, 0.0, 0.0, 0.0)
//
//            val rotations = listOf(R1, R2)
//            var bestR: Mat? = null
//            var maxPositive = 0
//
//            for (R in rotations) {
//                val P2 = Mat(3, 4, CvType.CV_64F)
//
//                // Check matrix dimensions before multiplication
//                if (K.cols() != R.rows() || K.cols() != t.rows()) {
//                    Log.e("Matrix Error", "Incompatible matrix dimensions in selectCorrectRotation")
//                    continue // Skip this iteration
//                }
//
//                Core.gemm(K, R, 1.0, Mat(), 0.0, P2.submat(0, 3, 0, 3))
//                Core.gemm(K, t, 1.0, Mat(), 0.0, P2.submat(0, 3, 3, 4))
//
//                val points1Mat = MatOfPoint2f()
//                points1Mat.fromList(points1)
//
//                val points2Mat = MatOfPoint2f()
//                points2Mat.fromList(points2)
//
//                val homogeneousPoints = Mat()
//                Calib3d.triangulatePoints(P1, P2, points1Mat, points2Mat, homogeneousPoints)
//
//                var positiveCount = 0
//
//                for (i in 0 until homogeneousPoints.cols()) {
//                    val col = homogeneousPoints.col(i)
//                    val x = col.get(0, 0)[0]
//                    val y = col.get(1, 0)[0]
//                    val z = col.get(2, 0)[0]
//                    val w = col.get(3, 0)[0]
//
//                    if (w != 0.0) {
//                        val point3D = Point3(x / w, y / w, z / w)
//
//                        // Check if point is in front of both cameras
//                        if (isInFrontOfCamera(point3D, Mat.eye(3, 3, CvType.CV_64F), Mat.zeros(3, 1, CvType.CV_64F)) &&
//                            isInFrontOfCamera(point3D, R, t)) {
//                            positiveCount++
//                        }
//                    }
//                }
//
//                if (positiveCount > maxPositive) {
//                    maxPositive = positiveCount
//                    bestR = R
//                }
//
//                P2.release()
//                points1Mat.release()
//                points2Mat.release()
//                homogeneousPoints.release()
//            }
//
//            K.release()
//            P1.release()
//
//            return bestR ?: R1 // Return R1 as a fallback if no good solution is found
//        }
//
//        private fun isInFrontOfCamera(point3D: Point3, R: Mat, t: Mat): Boolean {
//            val p = Mat(3, 1, CvType.CV_64F)
//            p.put(0, 0, point3D.x, point3D.y, point3D.z)
//
//            val transformed = Mat()
//            Core.gemm(R, p, 1.0, t, 1.0, transformed)
//
//            val result = transformed.get(2, 0)[0] > 0
//
//            p.release()
//            transformed.release()
//
//            return result
//        }
//
//        fun computeEssentialMatrix(F: Mat, fx: Double, fy: Double, cx: Double, cy: Double): Mat {
//            // Create the intrinsic matrix K
//            val K = Mat(3, 3, CvType.CV_64FC1)
//            K.put(0, 0, fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
//
//            // Compute K transpose
//            val K_transpose = Mat()
//            Core.transpose(K, K_transpose)
//
//            // Check matrix dimensions before multiplication
//            if (F.cols() != K.rows() || K_transpose.cols() != F.rows()) {
//                Log.e("Matrix Error", "Incompatible matrix dimensions for Essential Matrix computation")
//                return Mat() // Return an empty matrix or handle the error as appropriate
//            }
//
//            // Compute E = K' * F * K
//            val temp = Mat()
//            val E = Mat()
//            Core.gemm(F, K, 1.0, Mat(), 0.0, temp)
//            Core.gemm(K_transpose, temp, 1.0, Mat(), 0.0, E)
//
//            // Release temporary matrices
//            K.release()
//            K_transpose.release()
//            temp.release()
//
//            return E
//        }
//
//
//
//
//
//
//        private fun ImageProxy.toBitmap(): Bitmap {
//            val yBuffer = planes[0].buffer
//            val uBuffer = planes[1].buffer
//            val vBuffer = planes[2].buffer
//
//            val ySize = yBuffer.remaining()
//            val uSize = uBuffer.remaining()
//            val vSize = vBuffer.remaining()
//
//            val nv21 = ByteArray(ySize + uSize + vSize)
//
//            yBuffer.get(nv21, 0, ySize)
//            vBuffer.get(nv21, ySize, vSize)
//            uBuffer.get(nv21, ySize + vSize, uSize)
//
//            val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
//            val out = ByteArrayOutputStream()
//            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
//            val imageBytes = out.toByteArray()
//            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
//        }
//
//
//
//
//        private fun Bitmap.toMat(): Mat {
//            val mat = Mat(height, width, CvType.CV_8UC4)
//            Utils.bitmapToMat(this, mat)
//            return mat
//        }
//
//        private fun Mat.toBitmap(): Bitmap {
//            val resultBitmap = Bitmap.createBitmap(cols(), rows(), Bitmap.Config.ARGB_8888)
//            Utils.matToBitmap(this, resultBitmap)
//            return resultBitmap
//        }
//    }
//
//    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
//        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
//    }
//
//    override fun onRequestPermissionsResult(
//        requestCode: Int, permissions: Array<String>, grantResults: IntArray
//    ) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
//        if (requestCode == REQUEST_CODE_PERMISSIONS) {
//            if (allPermissionsGranted()) {
//                startCamera()
//            } else {
//                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
//                finish()
//            }
//        }
//    }
//
//    override fun onDestroy() {
//        super.onDestroy()
//        cameraExecutor.shutdown()
//    }
//
//    companion object {
//        private const val REQUEST_CODE_PERMISSIONS = 10
//        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
//    }
//}
//
//
//
//class PointCloudProcessor {
//    fun refinePointCloud(points3D: List<Point3>): List<Point3> {
//        // Filter out points with negative z-values (behind the camera)
//        val filteredPoints = points3D.filter { it.z > 0 }
//
//        // Check if filteredPoints is empty
//        if (filteredPoints.isEmpty()) {
//            return emptyList()
//        }
//
//        // Remove outliers (you might need to adjust the threshold)
//        val mean = calculateMean(filteredPoints)
//        val stdDev = calculateStdDev(filteredPoints, mean)
//        val threshold = 2.0 // Adjust this value as needed
//
//        return filteredPoints.filter { point ->
//            val distance = calculateDistance(point, mean)
//            distance < threshold * stdDev
//        }
//    }
//
//    private fun calculateMean(points: List<Point3>): Point3 {
//
//        if (points.isEmpty()) {
//            return Point3(0.0, 0.0, 0.0)
//        }
//
//        val sum = points.reduce { acc, point ->
//            Point3(acc.x + point.x, acc.y + point.y, acc.z + point.z)
//        }
//        return Point3(sum.x / points.size, sum.y / points.size, sum.z / points.size)
//    }
//
//    private fun calculateStdDev(points: List<Point3>, mean: Point3): Double {
//
//        if (points.isEmpty()) {
//            return 0.0
//        }
//
//        val squaredDiffs = points.map { point ->
//            val diff = Point3(point.x - mean.x, point.y - mean.y, point.z - mean.z)
//            diff.x * diff.x + diff.y * diff.y + diff.z * diff.z
//        }
//        val variance = squaredDiffs.average()
//        return Math.sqrt(variance)
//    }
//
//    private fun calculateDistance(p1: Point3, p2: Point3): Double {
//        val dx = p1.x - p2.x
//        val dy = p1.y - p2.y
//        val dz = p1.z - p2.z
//        return Math.sqrt(dx * dx + dy * dy + dz * dz)
//    }
//}
//class PointCloudView @JvmOverloads constructor(
//    context: Context,
//    attrs: AttributeSet? = null,
//    defStyleAttr: Int = 0
//) : View(context, attrs, defStyleAttr) {
//
//    private val paint = Paint().apply {
//        style = Paint.Style.FILL
//        isAntiAlias = true
//    }
//
//    var points: List<Point3> = emptyList()
//        set(value) {
//            field = value
//            invalidate()
//        }
//
//    override fun onDraw(canvas: Canvas) {
//        super.onDraw(canvas)
//
//        if (points.isEmpty()) return
//
//        val maxDistance = points.maxOf { sqrt(it.x * it.x + it.y * it.y + it.z * it.z) }
//        val minZ = points.minOf { it.z }
//        val maxZ = points.maxOf { it.z }
//
//        points.forEach { point ->
//            // Map 3D coordinates to 2D screen coordinates
//            val x = (point.x + 1) * width / 2
//            val y = (1 - point.y) * height / 2
//
//            // Calculate distance from origin (0,0,0)
//            val distance = sqrt(point.x * point.x + point.y * point.y + point.z * point.z)
//
//            // Normalize distance to [0, 1] range
//            val normalizedDistance = distance / maxDistance
//
//            // Create a color gradient from blue (cold, far) to red (hot, close)
//            val hue = (240 - normalizedDistance * 240).toFloat()
//            val color = Color.HSVToColor(floatArrayOf(hue, 1f, 1f))
//
//            // Set point size based on z-value (depth)
//            val normalizedZ = (point.z - minZ) / (maxZ - minZ)
//            val pointSize = 2f + normalizedZ * 8f  // Points will range from 2 to 10 in size
//
//            paint.color = color
//            canvas.drawCircle(x.toFloat(), y.toFloat(), pointSize.toFloat(), paint)
//        }
//    }
//}


//package com.example.ivalkxyz
//
//import android.Manifest
//import android.annotation.SuppressLint
//import android.content.Context
//import android.content.pm.PackageManager
//import android.graphics.Bitmap
//import android.graphics.Rect
//import android.graphics.BitmapFactory
//import android.graphics.ImageFormat
//import android.graphics.YuvImage
//import android.hardware.camera2.CameraCharacteristics
//import android.hardware.camera2.CameraManager
//import android.os.Bundle
//import android.util.Log
//import android.widget.Toast
//import androidx.appcompat.app.AppCompatActivity
//import androidx.camera.core.CameraSelector
//import androidx.camera.core.ImageAnalysis
//import androidx.camera.core.ImageProxy
//import androidx.camera.core.Preview
//import android.widget.ImageView
//import android.widget.TextView
//import androidx.camera.lifecycle.ProcessCameraProvider
//import androidx.camera.view.PreviewView
//import androidx.core.app.ActivityCompat
//import androidx.core.content.ContextCompat
//import org.opencv.android.NativeCameraView.TAG
//import org.opencv.android.OpenCVLoader
//import org.opencv.android.Utils
//import org.opencv.calib3d.Calib3d
//import org.opencv.core.*
//import org.opencv.features2d.SIFT
//import org.opencv.features2d.DescriptorMatcher
//import org.opencv.features2d.Features2d
//import org.opencv.imgproc.Imgproc
//import java.io.ByteArrayOutputStream
//import java.util.concurrent.ExecutorService
//import java.util.concurrent.Executors
//
//class MainActivity : AppCompatActivity() {
//    private lateinit var cameraExecutor: ExecutorService
//    private lateinit var viewFinder: PreviewView
//    private lateinit var resultImageView: ImageView
//    private lateinit var infoTextView: TextView
//    //private lateinit var pointCloudView: PointCloudView
//
//    @SuppressLint("MissingInflatedId")
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_main)
//
//        viewFinder = findViewById(R.id.viewFinder)
//        resultImageView = findViewById(R.id.resultImageView)
//        infoTextView = findViewById(R.id.infoTextView)
//        //pointCloudView = findViewById(R.id.pointCloudView)
//
//        if (!OpenCVLoader.initDebug()) {
//            Toast.makeText(this, "Unable to load OpenCV", Toast.LENGTH_LONG).show()
//            return
//        }
//
//        if (allPermissionsGranted()) {
//
//            startCamera()
//        } else {
//            ActivityCompat.requestPermissions(
//                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
//            )
//
//        }
//
//        cameraExecutor = Executors.newSingleThreadExecutor()
//    }
//
//    private fun startCamera() {
//        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
//
//        cameraProviderFuture.addListener({
//            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
//
//            val preview = Preview.Builder()
//                .build()
//                .also {
//                    it.setSurfaceProvider(viewFinder.surfaceProvider)
//                }
//
//            val imageAnalyzer = ImageAnalysis.Builder()
//                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                .build()
//                .also {
//                    it.setAnalyzer(cameraExecutor, FeatureMatchingAnalyzer())
//                }
//
//            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
//
//            try {
//                cameraProvider.unbindAll()
//                cameraProvider.bindToLifecycle(
//                    this, cameraSelector, preview, imageAnalyzer
//                )
//            } catch (exc: Exception) {
//                Log.e(TAG, "Use case binding failed", exc)
//            }
//
//        }, ContextCompat.getMainExecutor(this))
//    }
//
//    inner class FeatureMatchingAnalyzer : ImageAnalysis.Analyzer {
//        private val sift = SIFT.create()
//        private val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE)
//        private var previousFrame: Mat? = null
//        private var previousKeypoints: MatOfKeyPoint? = null
//        private var previousDescriptors: Mat? = null
//
//
//        @SuppressLint("SetTextI18n")
//        override fun analyze(image: ImageProxy) {
//            val currentFrame = image.toBitmap().toMat()
//            val grayFrame = Mat()
//
//            Imgproc.cvtColor(currentFrame, grayFrame, Imgproc.COLOR_RGBA2GRAY)
//
//            val currentKeypoints = MatOfKeyPoint()
//            //val previouskeypoints = MatOfKeyPoint()
//
//            val keypoints1 = MatOfKeyPoint()
//
//            val keypoints2 = MatOfKeyPoint()
//
//            val currentDescriptors = Mat()
//            sift.detectAndCompute(grayFrame, Mat(), currentKeypoints, currentDescriptors)
//
//            if (previousFrame != null && previousKeypoints != null && previousDescriptors != null) {
//                // Match features
//                val matches = MatOfDMatch()
//                matcher.match(previousDescriptors, currentDescriptors, matches)
//
//                // Filter good matches
//                val goodMatches = matches.toArray().sortedBy { it.distance }.take(50)
//
//
//                // Extract matched keypoints
//                val listOfMatches = matches.toList()
//                println("list fo mathed points :$goodMatches")
//                println("list of previous points:${previousKeypoints!!.toList()}")
//                println("list of current points: ${currentKeypoints.toList()}")
//                val srcPoints = mutableListOf<Point>()
//                val dstPoints = mutableListOf<Point>()
//                for (match in goodMatches) {
//                    srcPoints.add(previousKeypoints!!.toList()[match.queryIdx].pt)
//                    dstPoints.add(currentKeypoints.toList()[match.trainIdx].pt)
//                }
//                println("points :")
//                println(srcPoints)
//                println(dstPoints)
//
//
//
//                // Convert points to MatOfPoint2f
//                val srcMatOfPoint2f = MatOfPoint2f(*srcPoints.toTypedArray())
//                val dstMatOfPoint2f = MatOfPoint2f(*dstPoints.toTypedArray())
//                println("src points : $srcMatOfPoint2f")
//                println("src points : $dstMatOfPoint2f")
//
//                if (srcMatOfPoint2f.empty() || dstMatOfPoint2f.empty()) {
//                    println("Source or Destination points matrix is empty.")
//                    return
//                }
//
//                //val intristic parameters
//
//                val cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
//
//                val cameraIdList = cameraManager.cameraIdList
//                val cameraId = 0
//                val characteristics = cameraManager.getCameraCharacteristics(cameraId.toString())
//
//                // Get focal lengths
//                val focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
//
//                // Ensure focalLengths is not null and get the focal length values
//                val focalLengthX = focalLengths?.getOrNull(0) ?: 0f
//                val focalLengthY = if ((focalLengths?.size ?: 0) > 1) focalLengths?.get(1) else focalLengthX
//
//                // Get sensor size
//                val sensorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
//
//                // Principal point (assuming it's at the center of the sensor)
//                val cx = sensorSize?.width?.div(2) ?: 0f
//                val cy = sensorSize?.height?.div(2) ?: 0f
//
//
//
//
//
//                // Find the Fundamental Matrix using the 8-point algorithm with RANSAC
//                val mask = Mat()
//                val fundamentalMatrix = Calib3d.findFundamentalMat(
//                    srcMatOfPoint2f,
//                    dstMatOfPoint2f,
//                    Calib3d.FM_RANSAC,
//                    3.0,  // Distance to epipolar line
//                    0.99, // Confidence level
//                    mask
//                )
//
//                // Construct intrinsic matrix K
//                val K = arrayOf(
//                    floatArrayOf(focalLengthX, 0f, cx),
//                    focalLengthY?.let { floatArrayOf(0f, it, cy) },
//                    floatArrayOf(0f, 0f, 1f)
//                )
//
//                // Transpose of intrinsic matrix K'
//                val K_transpose = arrayOf(
//                    floatArrayOf(focalLengthX, 0f, 0f),
//                    focalLengthY?.let { floatArrayOf(0f, it, 0f) },
//                    floatArrayOf(cx, cy, 1f)
//                )
//
//                // Check if the matrix is empty
//                if (fundamentalMatrix.empty()) {
//                    println("Fundamental matrix is empty.")
//
//
//
//                    //println("intristic matrix $intristic")
//                } else {
//                    println("Fundamental Matrix: \n$fundamentalMatrix")
//                    println("focal length X: $focalLengthX")
//                    println("focal length Y: $focalLengthY")
//                    println("principle point x: $cx")
//                    println("principle point y: $cy")
//                    println("matrix :$K")
//                    println("matrix transpose: $K_transpose")
//
//                    println("F dimensions: ${fundamentalMatrix.rows()} x ${fundamentalMatrix.cols()}")
//                    println("K dimensions: ${K.size} x ${K[0]?.size}")
//                    println("K_transpose dimensions: ${K_transpose.size} x ${K_transpose[0]?.size}")
//
//                    // Usage
//                    val F: Mat =fundamentalMatrix  // Your fundamental matrix
//                    val fx = 5.56
//                    val fy = 5.56
//                    val cx = 4.1285
//                    val cy = 3.0965
//
//                    try {
//                        val E = computeEssentialMatrix(F, fx, fy, cx, cy)
//                        // Decompose essential matrix
//                        val R1 = Mat()
//                        val R2 = Mat()
//                        val t = Mat()
//
//
//                        println("Essential Matrix:")
//                        println(E.dump())
//
//                        Calib3d.decomposeEssentialMat(E, R1, R2, t)
//
//                        println("\nRotation Matrix 1:")
//                        println(R1.dump())
//
//                        println("\nRotation Matrix 2:")
//                        println(R2.dump())
//
//                        println("\nTranslation Vector:")
//                        println(t.dump())
//
//                        // Choose the correct rotation matrix
//                        val R = selectCorrectRotation(R1, R2, t, srcPoints, dstPoints)
//
//                        println("correct rotation matrix:")
//                        println(R.dump())
//
//                        // Triangulate points
//                        val points3D = triangulatePoints(R, Mat.eye(3, 3, CvType.CV_64F), t, srcPoints, dstPoints)
//
//                        println("points 3d")
//                        println(points3D)
//
//                        // In your MainActivity or wherever you're processing the point cloud
//                        val pointCloudProcessor = PointCloudProcessor()
//                        val refinedPoints3D = pointCloudProcessor.refinePointCloud(points3D)
//
//                        // In your activity
//                        //val pointCloudView = PointCloudView(this)
//                        //pointCloudView.points = refinedPoints3D
//                        println("refined matrix :")
//                        println(refinedPoints3D)
//                        println(refinedPoints3D.size)
//
//
//
//
//
//
//
//
//
//
//                        // Don't forget to release the matrices when done
//                        F.release()
//                        E.release()
//                        R1.release()
//                        R2.release()
//                        t.release()
//                        R.release()
//                        // Use E...
//                    } catch (e: CvException) {
//                        Log.e("OpenCV Error", "Error in computeEssentialMatrix: ${e.message}")
//                        // Handle the error...
//                    }
//
//
//
//
//                }
//
//
////
//
//
//                // Draw matches
//                val resultMat = Mat()
//                Features2d.drawMatches(
//                    previousFrame,
//                    previousKeypoints,
//                    grayFrame,
//                    currentKeypoints,
//                    MatOfDMatch().apply { fromList(goodMatches) },
//                    resultMat
//                )
//
//                val resultBitmap = resultMat.toBitmap()
//
//                runOnUiThread {
//                    resultImageView.setImageBitmap(resultBitmap)
//                    infoTextView.text = "Total keypoints: ${currentKeypoints.rows()}\n" +
//                            "Matched keypoints: ${goodMatches.size}\n" +
//                            "Best match distance: %.2f".format(goodMatches.firstOrNull()?.distance ?: 0f)
//
//
//                }
//            } else {
//                runOnUiThread {
//                    resultImageView.setImageBitmap(currentFrame.toBitmap())
//                    infoTextView.text = "Initializing... Keypoints: ${currentKeypoints.rows()}"
//                }
//            }
//
//            // Update previous frame data
//            previousFrame = grayFrame
//            previousKeypoints = currentKeypoints
//            previousDescriptors = currentDescriptors
//
//            image.close()
//        }
//
//        private fun triangulatePoints(R1: Mat, R2: Mat, t: Mat, points1: List<Point>, points2: List<Point>): List<Point3> {
//            val K = Mat(3, 3, CvType.CV_64F)
//            K.put(0, 0, 5.56, 0.0, 4.1285, 0.0, 5.56, 3.0965, 0.0, 0.0, 1.0)
//
//            val P1 = Mat(3, 4, CvType.CV_64F)
//            K.copyTo(P1.submat(0, 3, 0, 3))
//            P1.put(0, 3, 0.0, 0.0, 0.0)
//
//            val P2 = Mat(3, 4, CvType.CV_64F)
//
//            // Check matrix dimensions before multiplication
//            if (K.cols() != R1.rows() || K.cols() != t.rows()) {
//                Log.e("Matrix Error", "Incompatible matrix dimensions in triangulatePoints")
//                return emptyList() // Return an empty list or handle the error as appropriate
//            }
//
//            Core.gemm(K, R1, 1.0, Mat(), 0.0, P2.submat(0, 3, 0, 3))
//            Core.gemm(K, t, 1.0, Mat(), 0.0, P2.submat(0, 3, 3, 4))
//
//            val points1Mat = MatOfPoint2f()
//            points1Mat.fromList(points1)
//
//            val points2Mat = MatOfPoint2f()
//            points2Mat.fromList(points2)
//
//            val homogeneousPoints = Mat()
//            Calib3d.triangulatePoints(P1, P2, points1Mat, points2Mat, homogeneousPoints)
//
//            val points3D = mutableListOf<Point3>()
//            for (i in 0 until homogeneousPoints.cols()) {
//                val col = homogeneousPoints.col(i)
//                val x = col.get(0, 0)[0] / col.get(3, 0)[0]
//                val y = col.get(1, 0)[0] / col.get(3, 0)[0]
//                val z = col.get(2, 0)[0] / col.get(3, 0)[0]
//                points3D.add(Point3(x, y, z))
//            }
//
//            P1.release()
//            P2.release()
//            K.release()
//            points1Mat.release()
//            points2Mat.release()
//            homogeneousPoints.release()
//
//            return points3D
//        }
//
//        private fun selectCorrectRotation(R1: Mat, R2: Mat, t: Mat, points1: List<Point>, points2: List<Point>): Mat {
//            val K = Mat(3, 3, CvType.CV_64F)
//            K.put(0, 0, 5.56, 0.0, 4.1285, 0.0, 5.56, 3.0965, 0.0, 0.0, 1.0)
//
//            val P1 = Mat(3, 4, CvType.CV_64F)
//            K.copyTo(P1.submat(0, 3, 0, 3))
//            P1.put(0, 3, 0.0, 0.0, 0.0)
//
//            val rotations = listOf(R1, R2)
//            var bestR: Mat? = null
//            var maxPositive = 0
//
//            for (R in rotations) {
//                val P2 = Mat(3, 4, CvType.CV_64F)
//
//                // Check matrix dimensions before multiplication
//                if (K.cols() != R.rows() || K.cols() != t.rows()) {
//                    Log.e("Matrix Error", "Incompatible matrix dimensions in selectCorrectRotation")
//                    continue // Skip this iteration
//                }
//
//                Core.gemm(K, R, 1.0, Mat(), 0.0, P2.submat(0, 3, 0, 3))
//                Core.gemm(K, t, 1.0, Mat(), 0.0, P2.submat(0, 3, 3, 4))
//
//                val points1Mat = MatOfPoint2f()
//                points1Mat.fromList(points1)
//
//                val points2Mat = MatOfPoint2f()
//                points2Mat.fromList(points2)
//
//                val homogeneousPoints = Mat()
//                Calib3d.triangulatePoints(P1, P2, points1Mat, points2Mat, homogeneousPoints)
//
//                var positiveCount = 0
//
//                for (i in 0 until homogeneousPoints.cols()) {
//                    val col = homogeneousPoints.col(i)
//                    val x = col.get(0, 0)[0]
//                    val y = col.get(1, 0)[0]
//                    val z = col.get(2, 0)[0]
//                    val w = col.get(3, 0)[0]
//
//                    if (w != 0.0) {
//                        val point3D = Point3(x / w, y / w, z / w)
//
//                        // Check if point is in front of both cameras
//                        if (isInFrontOfCamera(point3D, Mat.eye(3, 3, CvType.CV_64F), Mat.zeros(3, 1, CvType.CV_64F)) &&
//                            isInFrontOfCamera(point3D, R, t)) {
//                            positiveCount++
//                        }
//                    }
//                }
//
//                if (positiveCount > maxPositive) {
//                    maxPositive = positiveCount
//                    bestR = R
//                }
//
//                P2.release()
//                points1Mat.release()
//                points2Mat.release()
//                homogeneousPoints.release()
//            }
//
//            K.release()
//            P1.release()
//
//            return bestR ?: R1 // Return R1 as a fallback if no good solution is found
//        }
//
//        private fun isInFrontOfCamera(point3D: Point3, R: Mat, t: Mat): Boolean {
//            val p = Mat(3, 1, CvType.CV_64F)
//            p.put(0, 0, point3D.x, point3D.y, point3D.z)
//
//            val transformed = Mat()
//            Core.gemm(R, p, 1.0, t, 1.0, transformed)
//
//            val result = transformed.get(2, 0)[0] > 0
//
//            p.release()
//            transformed.release()
//
//            return result
//        }
//
//        fun computeEssentialMatrix(F: Mat, fx: Double, fy: Double, cx: Double, cy: Double): Mat {
//            // Create the intrinsic matrix K
//            val K = Mat(3, 3, CvType.CV_64FC1)
//            K.put(0, 0, fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
//
//            // Compute K transpose
//            val K_transpose = Mat()
//            Core.transpose(K, K_transpose)
//
//            // Check matrix dimensions before multiplication
//            if (F.cols() != K.rows() || K_transpose.cols() != F.rows()) {
//                Log.e("Matrix Error", "Incompatible matrix dimensions for Essential Matrix computation")
//                return Mat() // Return an empty matrix or handle the error as appropriate
//            }
//
//            // Compute E = K' * F * K
//            val temp = Mat()
//            val E = Mat()
//            Core.gemm(F, K, 1.0, Mat(), 0.0, temp)
//            Core.gemm(K_transpose, temp, 1.0, Mat(), 0.0, E)
//
//            // Release temporary matrices
//            K.release()
//            K_transpose.release()
//            temp.release()
//
//            return E
//        }
//
//
//
//
//
//
//        private fun ImageProxy.toBitmap(): Bitmap {
//            val yBuffer = planes[0].buffer
//            val uBuffer = planes[1].buffer
//            val vBuffer = planes[2].buffer
//
//            val ySize = yBuffer.remaining()
//            val uSize = uBuffer.remaining()
//            val vSize = vBuffer.remaining()
//
//            val nv21 = ByteArray(ySize + uSize + vSize)
//
//            yBuffer.get(nv21, 0, ySize)
//            vBuffer.get(nv21, ySize, vSize)
//            uBuffer.get(nv21, ySize + vSize, uSize)
//
//            val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
//            val out = ByteArrayOutputStream()
//            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
//            val imageBytes = out.toByteArray()
//            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
//        }
//
//
//
//
//        private fun Bitmap.toMat(): Mat {
//            val mat = Mat(height, width, CvType.CV_8UC4)
//            Utils.bitmapToMat(this, mat)
//            return mat
//        }
//
//        private fun Mat.toBitmap(): Bitmap {
//            val resultBitmap = Bitmap.createBitmap(cols(), rows(), Bitmap.Config.ARGB_8888)
//            Utils.matToBitmap(this, resultBitmap)
//            return resultBitmap
//        }
//    }
//
//    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
//        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
//    }
//
//    override fun onRequestPermissionsResult(
//        requestCode: Int, permissions: Array<String>, grantResults: IntArray
//    ) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
//        if (requestCode == REQUEST_CODE_PERMISSIONS) {
//            if (allPermissionsGranted()) {
//                startCamera()
//            } else {
//                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
//                finish()
//            }
//        }
//    }
//
//    override fun onDestroy() {
//        super.onDestroy()
//        cameraExecutor.shutdown()
//    }
//
//    companion object {
//        private const val REQUEST_CODE_PERMISSIONS = 10
//        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
//    }
//}
//
//
//
//class PointCloudProcessor {
//    fun refinePointCloud(points3D: List<Point3>): List<Point3> {
//        // Filter out points with negative z-values (behind the camera)
//        val filteredPoints = points3D.filter { it.z > 0 }
//
//        // Check if filteredPoints is empty
//        if (filteredPoints.isEmpty()) {
//            return emptyList()
//        }
//
//        // Remove outliers (you might need to adjust the threshold)
//        val mean = calculateMean(filteredPoints)
//        val stdDev = calculateStdDev(filteredPoints, mean)
//        val threshold = 2.0 // Adjust this value as needed
//
//        return filteredPoints.filter { point ->
//            val distance = calculateDistance(point, mean)
//            distance < threshold * stdDev
//        }
//    }
//
//    private fun calculateMean(points: List<Point3>): Point3 {
//
//        if (points.isEmpty()) {
//            return Point3(0.0, 0.0, 0.0)
//        }
//
//        val sum = points.reduce { acc, point ->
//            Point3(acc.x + point.x, acc.y + point.y, acc.z + point.z)
//        }
//        return Point3(sum.x / points.size, sum.y / points.size, sum.z / points.size)
//    }
//
//    private fun calculateStdDev(points: List<Point3>, mean: Point3): Double {
//
//        if (points.isEmpty()) {
//            return 0.0
//        }
//
//        val squaredDiffs = points.map { point ->
//            val diff = Point3(point.x - mean.x, point.y - mean.y, point.z - mean.z)
//            diff.x * diff.x + diff.y * diff.y + diff.z * diff.z
//        }
//        val variance = squaredDiffs.average()
//        return Math.sqrt(variance)
//    }
//
//    private fun calculateDistance(p1: Point3, p2: Point3): Double {
//        val dx = p1.x - p2.x
//        val dy = p1.y - p2.y
//        val dz = p1.z - p2.z
//        return Math.sqrt(dx * dx + dy * dy + dz * dz)
//    }
//}



// Visualization (basic example using Android's Canvas)
//class PointCloudView(context: Context) : View(context) {
//    var points: List<Point3> = emptyList()
//
//    override fun onDraw(canvas: Canvas) {
//        super.onDraw(canvas)
//        val paint = Paint().apply {
//            color = Color.BLUE
//            style = Paint.Style.FILL
//        }
//
//        points.forEach { point ->
//            // Simple projection to 2D (you might want to implement a proper camera model)
//            val x = (point.x * 100 + width / 2).toFloat()
//            val y = (point.y * 100 + height / 2).toFloat()
//            canvas.drawCircle(x, y, 5f, paint)
//        }
//    }
//}


// Add pointCloudView to your layout

//package com.example.ivalkxyz
//
//import android.Manifest
//import android.annotation.SuppressLint
//import android.content.pm.PackageManager
//import android.graphics.Bitmap
//import android.graphics.Rect
//import android.graphics.BitmapFactory
//import android.graphics.ImageFormat
//import android.graphics.YuvImage
//import android.os.Bundle
//import android.util.Log
//import android.widget.Toast
//import androidx.appcompat.app.AppCompatActivity
//import androidx.camera.core.CameraSelector
//import androidx.camera.core.ImageAnalysis
//import androidx.camera.core.ImageProxy
//import androidx.camera.core.Preview
//import android.widget.ImageView
//import android.widget.TextView
//import androidx.camera.lifecycle.ProcessCameraProvider
//import androidx.camera.view.PreviewView
//import androidx.core.app.ActivityCompat
//import androidx.core.content.ContextCompat
//import org.opencv.android.NativeCameraView.TAG
//import org.opencv.android.OpenCVLoader
//import org.opencv.android.Utils
//import org.opencv.calib3d.Calib3d
//import org.opencv.core.*
//import org.opencv.features2d.SIFT
//import org.opencv.features2d.DescriptorMatcher
//import org.opencv.features2d.Features2d
//import org.opencv.imgproc.Imgproc
//import java.io.ByteArrayOutputStream
//import java.util.concurrent.ExecutorService
//import java.util.concurrent.Executors
//
//class MainActivity : AppCompatActivity() {
//    private lateinit var cameraExecutor: ExecutorService
//    private lateinit var viewFinder: PreviewView
//    private lateinit var resultImageView: ImageView
//    private lateinit var infoTextView: TextView
//
//    @SuppressLint("MissingInflatedId")
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_main)
//
//        viewFinder = findViewById(R.id.viewFinder)
//        resultImageView = findViewById(R.id.resultImageView)
//        infoTextView = findViewById(R.id.infoTextView)
//
//        if (!OpenCVLoader.initDebug()) {
//            Toast.makeText(this, "Unable to load OpenCV", Toast.LENGTH_LONG).show()
//            return
//        }
//
//        if (allPermissionsGranted()) {
//            startCamera()
//        } else {
//            ActivityCompat.requestPermissions(
//                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
//            )
//        }
//
//        cameraExecutor = Executors.newSingleThreadExecutor()
//    }
//
//    private fun startCamera() {
//        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
//
//        cameraProviderFuture.addListener({
//            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
//
//            val preview = Preview.Builder()
//                .build()
//                .also {
//                    it.setSurfaceProvider(viewFinder.surfaceProvider)
//                }
//
//            val imageAnalyzer = ImageAnalysis.Builder()
//                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                .build()
//                .also {
//                    it.setAnalyzer(cameraExecutor, FeatureMatchingAnalyzer())
//                }
//
//            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
//
//            try {
//                cameraProvider.unbindAll()
//                cameraProvider.bindToLifecycle(
//                    this, cameraSelector, preview, imageAnalyzer
//                )
//            } catch (exc: Exception) {
//                Log.e(TAG, "Use case binding failed", exc)
//            }
//
//        }, ContextCompat.getMainExecutor(this))
//    }
//
//    private inner class FeatureMatchingAnalyzer : ImageAnalysis.Analyzer {
//        private val sift = SIFT.create()
//        private val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE)
//        private var previousFrame: Mat? = null
//        private var previousKeypoints: MatOfKeyPoint? = null
//        private var previousDescriptors: Mat? = null
//
//        @SuppressLint("SetTextI18n")
//        override fun analyze(image: ImageProxy) {
//            val currentFrame = image.toBitmap().toMat()
//            val grayFrame = Mat()
//            Imgproc.cvtColor(currentFrame, grayFrame, Imgproc.COLOR_RGBA2GRAY)
//
//            val currentKeypoints = MatOfKeyPoint()
//            val currentDescriptors = Mat()
//            sift.detectAndCompute(grayFrame, Mat(), currentKeypoints, currentDescriptors)
//
//            if (previousFrame != null && previousKeypoints != null && previousDescriptors != null) {
//                // Match features
//                val matches = MatOfDMatch()
//                matcher.match(previousDescriptors, currentDescriptors, matches)
//
//                // Filter good matches
//                val goodMatches = matches.toArray().sortedBy { it.distance }.take(50)
//
//                // Draw matches
//                val resultMat = Mat()
//                Features2d.drawMatches(
//                    previousFrame,
//                    previousKeypoints,
//                    grayFrame,
//                    currentKeypoints,
//                    MatOfDMatch().apply { fromList(goodMatches) },
//                    resultMat
//                )
//
//                val resultBitmap = resultMat.toBitmap()
//
//                runOnUiThread {
//                    resultImageView.setImageBitmap(resultBitmap)
//                    infoTextView.text = "Total keypoints: ${currentKeypoints.rows()}\n" +
//                            "Matched keypoints: ${goodMatches.size}\n" +
//                            "Best match distance: %.2f".format(goodMatches.firstOrNull()?.distance ?: 0f)
//                }
//            } else {
//                runOnUiThread {
//                    resultImageView.setImageBitmap(currentFrame.toBitmap())
//                    infoTextView.text = "Initializing... Keypoints: ${currentKeypoints.rows()}"
//                }
//            }
//
//            // Update previous frame data
//            previousFrame = grayFrame
//            previousKeypoints = currentKeypoints
//            previousDescriptors = currentDescriptors
//
//            image.close()
//        }
//
//        private fun ImageProxy.toBitmap(): Bitmap {
//            val yBuffer = planes[0].buffer
//            val uBuffer = planes[1].buffer
//            val vBuffer = planes[2].buffer
//
//            val ySize = yBuffer.remaining()
//            val uSize = uBuffer.remaining()
//            val vSize = vBuffer.remaining()
//
//            val nv21 = ByteArray(ySize + uSize + vSize)
//
//            yBuffer.get(nv21, 0, ySize)
//            vBuffer.get(nv21, ySize, vSize)
//            uBuffer.get(nv21, ySize + vSize, uSize)
//
//            val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
//            val out = ByteArrayOutputStream()
//            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
//            val imageBytes = out.toByteArray()
//            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
//        }
//
//        private fun estimateMotion(matches: List<DMatch>, keypoints1: MatOfKeyPoint, keypoints2: MatOfKeyPoint): Mat {
//            val points1 = MatOfPoint2f(*matches.map { keypoints1.toArray()[it.queryIdx].pt }.toTypedArray())
//            val points2 = MatOfPoint2f(*matches.map { keypoints2.toArray()[it.trainIdx].pt }.toTypedArray())
//
//            val fundamentalMatrix = Calib3d.findFundamentalMat(points1, points2, Calib3d.FM_RANSAC)
//
//            return fundamentalMatrix
//        }
//
//        private fun triangulatePoints(projMatrix1: Mat, projMatrix2: Mat, points1: MatOfPoint2f, points2: MatOfPoint2f): Mat {
//            val points4D = Mat()
//            Calib3d.triangulatePoints(projMatrix1, projMatrix2, points1, points2, points4D)
//            return points4D
//        }
//
//
//
//        private fun Bitmap.toMat(): Mat {
//            val mat = Mat(height, width, CvType.CV_8UC4)
//            Utils.bitmapToMat(this, mat)
//            return mat
//        }
//
//        private fun Mat.toBitmap(): Bitmap {
//            val resultBitmap = Bitmap.createBitmap(cols(), rows(), Bitmap.Config.ARGB_8888)
//            Utils.matToBitmap(this, resultBitmap)
//            return resultBitmap
//        }
//    }
//
//    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
//        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
//    }
//
//    override fun onRequestPermissionsResult(
//        requestCode: Int, permissions: Array<String>, grantResults: IntArray
//    ) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
//        if (requestCode == REQUEST_CODE_PERMISSIONS) {
//            if (allPermissionsGranted()) {
//                startCamera()
//            } else {
//                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
//                finish()
//            }
//        }
//    }
//
//    override fun onDestroy() {
//        super.onDestroy()
//        cameraExecutor.shutdown()
//    }
//
//    companion object {
//        private const val REQUEST_CODE_PERMISSIONS = 10
//        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
//    }
//}

//package com.example.ivalkxyz
//
//import android.Manifest
//import android.annotation.SuppressLint
//import android.content.pm.PackageManager
//import android.graphics.Bitmap
//import android.graphics.Rect
//import android.graphics.BitmapFactory
//import android.graphics.ImageFormat
//import android.graphics.YuvImage
//import android.os.Bundle
//import android.util.Log
//import android.widget.Toast
//import androidx.appcompat.app.AppCompatActivity
//import androidx.camera.core.CameraSelector
//import androidx.camera.core.ImageAnalysis
//import androidx.camera.core.ImageProxy
//import androidx.camera.core.Preview
//import android.widget.ImageView
//import android.widget.TextView
//import androidx.camera.lifecycle.ProcessCameraProvider
//import androidx.camera.view.PreviewView
//import androidx.core.app.ActivityCompat
//import androidx.core.content.ContextCompat
//import org.opencv.android.NativeCameraView.TAG
//import org.opencv.android.OpenCVLoader
//import org.opencv.android.Utils
//import org.opencv.core.*
//import org.opencv.features2d.SIFT
//import org.opencv.features2d.DescriptorMatcher
//import org.opencv.features2d.Features2d
//import org.opencv.imgproc.Imgproc
//import java.io.ByteArrayOutputStream
//import java.util.concurrent.ExecutorService
//import java.util.concurrent.Executors
//
//
//import android.content.Context
//import android.hardware.Sensor
//import android.hardware.SensorEvent
//import android.hardware.SensorEventListener
//import android.hardware.SensorManager
//
//import org.ejml.simple.SimpleMatrix
//import org.apache.commons.math3.linear.ArrayRealVector
//import org.apache.commons.math3.linear.DecompositionSolver
//import org.apache.commons.math3.linear.QRDecomposition
//import org.apache.commons.math3.linear.RealMatrix
//import org.apache.commons.math3.linear.RealVector
//import org.opencv.calib3d.Calib3d
//import org.opencv.core.*
//import kotlin.math.sqrt
//
//class MainActivity : AppCompatActivity() {
//    private lateinit var cameraExecutor: ExecutorService
//    private lateinit var viewFinder: PreviewView
//    private lateinit var resultImageView: ImageView
//    private lateinit var infoTextView: TextView
//
//    @SuppressLint("MissingInflatedId")
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_main)
//
//        viewFinder = findViewById(R.id.viewFinder)
//        resultImageView = findViewById(R.id.resultImageView)
//        infoTextView = findViewById(R.id.infoTextView)
//
//        if (!OpenCVLoader.initDebug()) {
//            Toast.makeText(this, "Unable to load OpenCV", Toast.LENGTH_LONG).show()
//            return
//        }
//
//        if (allPermissionsGranted()) {
//            startCamera()
//        } else {
//            ActivityCompat.requestPermissions(
//                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
//            )
//        }
//
//        cameraExecutor = Executors.newSingleThreadExecutor()
//    }
//
//    private fun startCamera() {
//        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
//
//        cameraProviderFuture.addListener({
//            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
//
//            val preview = Preview.Builder()
//                .build()
//                .also {
//                    it.setSurfaceProvider(viewFinder.surfaceProvider)
//                }
//
//            val imageAnalyzer = ImageAnalysis.Builder()
//                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                .build()
//                .also {
//                    it.setAnalyzer(cameraExecutor, FeatureMatchingAnalyzer())
//                }
//
//            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
//
//            try {
//                cameraProvider.unbindAll()
//                cameraProvider.bindToLifecycle(
//                    this, cameraSelector, preview, imageAnalyzer
//                )
//            } catch (exc: Exception) {
//                Log.e(TAG, "Use case binding failed", exc)
//            }
//
//        }, ContextCompat.getMainExecutor(this))
//    }
//
//    private inner class FeatureMatchingAnalyzer : ImageAnalysis.Analyzer {
//
//
//        private val sift = SIFT.create()
//        private val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE)
//        private var previousFrame: Mat? = null
//        private var previousKeypoints: MatOfKeyPoint? = null
//        private var previousDescriptors: Mat? = null
//
//        private val sensorDataCollector = SensorDataCollector(this@MainActivity)
//
//        init {
//            sensorDataCollector.start()
//        }
//
//        @SuppressLint("SetTextI18n")
//        override fun analyze(image: ImageProxy) {
//            val currentFrame = image.toBitmap().toMat()
//            val grayFrame = Mat()
//            Imgproc.cvtColor(currentFrame, grayFrame, Imgproc.COLOR_RGBA2GRAY)
//
//            val currentKeypoints = MatOfKeyPoint()
//            val currentDescriptors = Mat()
//            sift.detectAndCompute(grayFrame, Mat(), currentKeypoints, currentDescriptors)
//
//            if (previousFrame != null && previousKeypoints != null && previousDescriptors != null) {
//                // Match features
//                val matches = MatOfDMatch()
//                matcher.match(previousDescriptors, currentDescriptors, matches)
//
//                // Filter good matches
//                val goodMatches = matches.toArray().sortedBy { it.distance }.take(50)
//
//                // Draw matches
//                val resultMat = Mat()
//                Features2d.drawMatches(
//                    previousFrame,
//                    previousKeypoints,
//                    grayFrame,
//                    currentKeypoints,
//                    MatOfDMatch().apply { fromList(goodMatches) },
//                    resultMat
//                )
//
//                // Collect matching points
//                val previousMatchedPoints = MatOfPoint2f()
//                val currentMatchedPoints = MatOfPoint2f()
//                for (match in goodMatches) {
//                    previousMatchedPoints.push_back(MatOfPoint2f(previousKeypoints!!.toArray()[match.queryIdx].pt))
//                    currentMatchedPoints.push_back(MatOfPoint2f(currentKeypoints.toArray()[match.trainIdx].pt))
//                }
//
//                // Integrate IMU data
//                val accel = sensorDataCollector.acceleration
//                val gyro = sensorDataCollector.angularVelocity
//
//                // Construct the observation matrix
//                buildObservationMatrix(previousMatchedPoints, currentMatchedPoints, currentDescriptors, accel, gyro)
//
//                // Draw matches
//                //val resultMat = Mat()
//                Features2d.drawMatches(
//                    previousFrame,
//                    previousKeypoints,
//                    grayFrame,
//                    currentKeypoints,
//                    MatOfDMatch().apply { fromList(goodMatches) },
//                    resultMat
//                )
//
//                val resultBitmap = resultMat.toBitmap()
//
//                runOnUiThread {
//                    resultImageView.setImageBitmap(resultBitmap)
//                    infoTextView.text = "Total keypoints: ${currentKeypoints.rows()}\n" +
//                            "Matched keypoints: ${goodMatches.size}\n" +
//                            "Best match distance: %.2f".format(goodMatches.firstOrNull()?.distance ?: 0f)
//                }
//            } else {
//                runOnUiThread {
//                    resultImageView.setImageBitmap(currentFrame.toBitmap())
//                    infoTextView.text = "Initializing... Keypoints: ${currentKeypoints.rows()}"
//                }
//            }
//
//            // Update previous frame data
//            previousFrame = grayFrame
//            previousKeypoints = currentKeypoints
//            previousDescriptors = currentDescriptors
//
//            image.close()
//        }
//
//        private fun buildObservationMatrix(
//            previousPoints: MatOfPoint2f,
//            currentPoints: MatOfPoint2f,
//            currentDescriptors: Mat,
//            accel: FloatArray,
//            gyro: FloatArray
//        ) {
//            // Construct an observation matrix
//            val observationMatrix = mutableListOf<FloatArray>()
//
//            for (i in 0 until currentPoints.rows()) {
//                val u = currentPoints[i, 0][0].toFloat()
//                val v = currentPoints[i, 0][1].toFloat()
//
//                // Extract the descriptor vector for the current point
//                val descriptorVector = FloatArray(currentDescriptors.cols())
//                currentDescriptors.row(i).get(0, 0, descriptorVector)
//
//                // Example observation row including descriptor vector
//                val observationRow = FloatArray(2 + descriptorVector.size + 6)
//                observationRow[0] = u
//                observationRow[1] = v
//                System.arraycopy(descriptorVector, 0, observationRow, 2, descriptorVector.size)
//                observationRow[2 + descriptorVector.size] = accel[0]
//                observationRow[3 + descriptorVector.size] = accel[1]
//                observationRow[4 + descriptorVector.size] = accel[2]
//                observationRow[5 + descriptorVector.size] = gyro[0]
//                observationRow[6 + descriptorVector.size] = gyro[1]
//                observationRow[7 + descriptorVector.size] = gyro[2]
//
//                observationMatrix.add(observationRow)
//                print(observationMatrix)
//            }
//
//
//            // Use the observation matrix for further processing like pose estimation or bundle adjustment
//        }
//
//
//        private fun ImageProxy.toBitmap(): Bitmap {
//            val yBuffer = planes[0].buffer
//            val uBuffer = planes[1].buffer
//            val vBuffer = planes[2].buffer
//
//            val ySize = yBuffer.remaining()
//            val uSize = uBuffer.remaining()
//            val vSize = vBuffer.remaining()
//
//            val nv21 = ByteArray(ySize + uSize + vSize)
//
//            yBuffer.get(nv21, 0, ySize)
//            vBuffer.get(nv21, ySize, vSize)
//            uBuffer.get(nv21, ySize + vSize, uSize)
//
//            val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
//            val out = ByteArrayOutputStream()
//            yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
//            val imageBytes = out.toByteArray()
//            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
//        }
//
//        private fun Bitmap.toMat(): Mat {
//            val mat = Mat(height, width, CvType.CV_8UC4)
//            Utils.bitmapToMat(this, mat)
//            return mat
//        }
//
//        private fun Mat.toBitmap(): Bitmap {
//            val resultBitmap = Bitmap.createBitmap(cols(), rows(), Bitmap.Config.ARGB_8888)
//            Utils.matToBitmap(this, resultBitmap)
//            return resultBitmap
//        }
//    }
//
//    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
//        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
//    }
//
//    override fun onRequestPermissionsResult(
//        requestCode: Int, permissions: Array<String>, grantResults: IntArray
//    ) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
//        if (requestCode == REQUEST_CODE_PERMISSIONS) {
//            if (allPermissionsGranted()) {
//                startCamera()
//            } else {
//                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
//                finish()
//            }
//        }
//    }
//
//    override fun onDestroy() {
//        super.onDestroy()
//        cameraExecutor.shutdown()
//    }
//
//    companion object {
//        private const val REQUEST_CODE_PERMISSIONS = 10
//        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
//    }
//}
//
//class SensorDataCollector(context: Context) : SensorEventListener {
//    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
//    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
//    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
//
//    var acceleration = FloatArray(3)
//    var angularVelocity = FloatArray(3)
//
//    fun start() {
//        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST)
//        sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_FASTEST)
//    }
//
//    fun stop() {
//        sensorManager.unregisterListener(this)
//    }
//
//    override fun onSensorChanged(event: SensorEvent?) {
//        event?.let {
//            when (event.sensor.type) {
//                Sensor.TYPE_ACCELEROMETER -> {
//                    System.arraycopy(event.values, 0, acceleration, 0, acceleration.size)
//                }
//                Sensor.TYPE_GYROSCOPE -> {
//                    System.arraycopy(event.values, 0, angularVelocity, 0, angularVelocity.size)
//                }
//            }
//        }
//    }
//
//    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
//        // Handle sensor accuracy changes if needed
//    }
//}



//package com.example.ivalkxyz
//
//import android.Manifest
//import android.content.pm.PackageManager
//import android.graphics.Bitmap
//import android.os.Bundle
//import android.widget.Toast
//import androidx.appcompat.app.AppCompatActivity
//import androidx.camera.core.CameraSelector
//import androidx.camera.core.ImageAnalysis
//import androidx.camera.core.ImageProxy
//import androidx.camera.core.Preview
//import android.widget.ImageView
//import androidx.camera.lifecycle.ProcessCameraProvider
//import androidx.camera.view.PreviewView
//import androidx.core.app.ActivityCompat
//import androidx.core.content.ContextCompat
//import org.opencv.android.OpenCVLoader
//import org.opencv.android.Utils
//import org.opencv.core.Mat
//import org.opencv.core.MatOfKeyPoint
//import org.opencv.features2d.SIFT
//import org.opencv.features2d.Features2d
//import org.opencv.imgproc.Imgproc
//import java.util.concurrent.ExecutorService
//import java.util.concurrent.Executors
//
//class MainActivity : AppCompatActivity() {
//    private lateinit var cameraExecutor: ExecutorService
//    private lateinit var viewFinder: PreviewView
//    private lateinit var resultImageView: ImageView
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_main)
//
//        viewFinder = findViewById(R.id.viewFinder)
//        resultImageView = findViewById(R.id.resultImageView)
//
//        // Initialize OpenCV
//        if (!OpenCVLoader.initDebug()) {
//            Toast.makeText(this, "Unable to load OpenCV", Toast.LENGTH_LONG).show()
//            return
//        }
//
//        // Request camera permissions
//        if (allPermissionsGranted()) {
//            startCamera()
//        } else {
//            ActivityCompat.requestPermissions(
//                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
//            )
//        }
//
//        cameraExecutor = Executors.newSingleThreadExecutor()
//    }
//
//    private fun startCamera() {
//        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
//
//        cameraProviderFuture.addListener({
//            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
//
//            val preview = Preview.Builder()
//                .build()
//                .also {
//                    it.setSurfaceProvider(viewFinder.surfaceProvider)
//                }
//
//            val imageAnalyzer = ImageAnalysis.Builder()
//                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                .build()
//                .also {
//                    it.setAnalyzer(cameraExecutor, SIFTAnalyzer())
//                }
//
//            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
//
//            try {
//                cameraProvider.unbindAll()
//                cameraProvider.bindToLifecycle(
//                    this, cameraSelector, preview, imageAnalyzer
//                )
//            } catch (exc: Exception) {
//                // Handle exceptions
//            }
//
//        }, ContextCompat.getMainExecutor(this))
//    }
//
//    private inner class SIFTAnalyzer : ImageAnalysis.Analyzer {
//        private val sift = SIFT.create()
//
//        override fun analyze(image: ImageProxy) {
//            runOnUiThread {
//                val bitmap = viewFinder.bitmap ?: return@runOnUiThread
//                val mat = Mat()
//                Utils.bitmapToMat(bitmap, mat)
//                Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY)
//
//                val keypoints = MatOfKeyPoint()
//                sift.detect(mat, keypoints)
//
//                Features2d.drawKeypoints(mat, keypoints, mat)
//
//                val resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
//                Utils.matToBitmap(mat, resultBitmap)
//
//                resultImageView.setImageBitmap(resultBitmap)
//                image.close()
//            }
//        }
//    }
//
//    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
//        ContextCompat.checkSelfPermission(
//            baseContext, it
//        ) == PackageManager.PERMISSION_GRANTED
//    }
//
//    override fun onRequestPermissionsResult(
//        requestCode: Int, permissions: Array<String>, grantResults: IntArray
//    ) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
//        if (requestCode == REQUEST_CODE_PERMISSIONS) {
//            if (allPermissionsGranted()) {
//                startCamera()
//            } else {
//                Toast.makeText(
//                    this,
//                    "Permissions not granted by the user.",
//                    Toast.LENGTH_SHORT
//                ).show()
//                finish()
//            }
//        }
//    }
//
//    override fun onDestroy() {
//        super.onDestroy()
//        cameraExecutor.shutdown()
//    }
//
//    companion object {
//        private const val REQUEST_CODE_PERMISSIONS = 10
//        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
//    }
//}
