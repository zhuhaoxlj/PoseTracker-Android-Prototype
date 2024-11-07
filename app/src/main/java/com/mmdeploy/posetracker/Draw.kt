package com.mmdeploy.posetracker

import mmdeploy.PoseTracker
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max

/**
 * @description: this is a util class for java demo.
 */
object Draw {
    /**
     * This function changes cvMat to Mat.
     *
     * @param frame:   the image with opencv Mat format.
     * @param results: the results array of PoseTracker.
     */
    fun drawPoseTrackerResult(frame: Mat, results: Array<PoseTracker.Result>) {
        var frame = frame
        val skeleton: Array<IntArray?>? = arrayOf<IntArray?>(
            intArrayOf(15, 13),
            intArrayOf(13, 11),
            intArrayOf(16, 14),
            intArrayOf(14, 12),
            intArrayOf(11, 12),
            intArrayOf(5, 11),
            intArrayOf(6, 12),
            intArrayOf(5, 6),
            intArrayOf(5, 7),
            intArrayOf(6, 8),
            intArrayOf(7, 9),
            intArrayOf(8, 10),
            intArrayOf(1, 2),
            intArrayOf(0, 1),
            intArrayOf(0, 2),
            intArrayOf(1, 3),
            intArrayOf(2, 4),
            intArrayOf(3, 5),
            intArrayOf(4, 6)
        )
        val fishSkeleton: Array<IntArray?>? = arrayOf<IntArray?>(
            intArrayOf(5, 6), intArrayOf(11, 12), intArrayOf(14, 13)
        )

        val middleLine: Array<IntArray?>? = arrayOf<IntArray?>(
            intArrayOf(5, 6), intArrayOf(12, 11)
        )

        val palette: Array<Scalar?>? = arrayOf<Scalar?>(
            Scalar(255.0, 128.0, 0.0), Scalar(255.0, 153.0, 51.0), Scalar(255.0, 178.0, 102.0),
            Scalar(230.0, 230.0, 0.0), Scalar(255.0, 153.0, 255.0), Scalar(153.0, 204.0, 255.0),
            Scalar(255.0, 102.0, 255.0), Scalar(255.0, 51.0, 255.0), Scalar(102.0, 178.0, 255.0),
            Scalar(51.0, 153.0, 255.0), Scalar(255.0, 153.0, 153.0), Scalar(255.0, 102.0, 102.0),
            Scalar(255.0, 51.0, 51.0), Scalar(153.0, 255.0, 153.0), Scalar(102.0, 255.0, 102.0),
            Scalar(51.0, 255.0, 51.0), Scalar(0.0, 255.0, 0.0), Scalar(0.0, 0.0, 255.0),
            Scalar(255.0, 0.0, 0.0), Scalar(255.0, 255.0, 255.0)
        )
        val linkColor: IntArray? = intArrayOf(
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        )
        val pointColor: IntArray? =
            intArrayOf(16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0)

        val scale = 1280 / max(frame.cols().toDouble(), frame.rows().toDouble()).toFloat()
        if (scale != 1f) {
            Imgproc.resize(frame, frame, Size(), scale.toDouble(), scale.toDouble())
        } else {
            frame = frame.clone()
        }
        for (i in results.indices) {
            val pt = results[i]
            for (j in pt.keypoints.indices) {
                val p = pt.keypoints[j]
                p.x *= scale
                p.y *= scale
                pt.keypoints[j] = p
            }
            val scoreThr = 0.4f
            val used: IntArray? = IntArray(pt.keypoints.size * 2)
            for (j in skeleton!!.indices) {
                val u = skeleton[j]!![0]
                val v = skeleton[j]!![1]
                if (pt.scores[u] > scoreThr && pt.scores[v] > scoreThr) {
                    used!![v] = 1
                    used[u] = used[v]
                    val pointU = Point(pt.keypoints[u].x.toDouble(), pt.keypoints[u].y.toDouble())
                    val pointV = Point(pt.keypoints[v].x.toDouble(), pt.keypoints[v].y.toDouble())
                    Imgproc.line(frame, pointU, pointV, palette!![linkColor!![j]], 4)
                }
            }

            // 画鱼骨线
            for (j in fishSkeleton!!.indices) {
                val u = fishSkeleton[j]!![0]
                val v = fishSkeleton[j]!![1]
                if (pt.scores[u] > scoreThr && pt.scores[v] > scoreThr) {
                    used!![v] = 1
                    used[u] = used[v]
                    val pointU = Point(pt.keypoints[u].x.toDouble(), pt.keypoints[u].y.toDouble())
                    val pointV = Point(pt.keypoints[v].x.toDouble(), pt.keypoints[v].y.toDouble())
                    Imgproc.drawExtendedLine(
                        frame,
                        pointU,
                        pointV,
                        100.0,
                        Scalar(255.0, 255.0, 255.0),
                        4
                    )
                }
            }

            // 中点连线
            run {
                var j = 0
                while (j < middleLine!!.size) {
                    val u1 = middleLine[j]!![0]
                    val v1 = middleLine[j]!![1]
                    val u2 = middleLine[j + 1]!![0]
                    val v2 = middleLine[j + 1]!![1]
                    if (pt.scores[u1] > scoreThr && pt.scores[v1] > scoreThr && pt.scores[u2] > scoreThr && pt.scores[v2] > scoreThr) {
                        val pointU1 =
                            Point(pt.keypoints[u1].x.toDouble(), pt.keypoints[u1].y.toDouble())
                        val pointV1 =
                            Point(pt.keypoints[v1].x.toDouble(), pt.keypoints[v1].y.toDouble())
                        val pointU2 =
                            Point(pt.keypoints[u2].x.toDouble(), pt.keypoints[u2].y.toDouble())
                        val pointV2 =
                            Point(pt.keypoints[v2].x.toDouble(), pt.keypoints[v2].y.toDouble())
                        Imgproc.drawExtendedLine(
                            frame,
                            getMidPoint(pointV1, pointU1),
                            getMidPoint(pointV2, pointU2),
                            300.0,
                            Scalar(255.0, 255.0, 255.0),
                            4
                        )
                    }
                    j += 2
                }
            }


            for (j in pt.keypoints.indices) {
                if (used!![j] == 1) {
                    val p = Point(pt.keypoints[j].x.toDouble(), pt.keypoints[j].y.toDouble())
                    Imgproc.circle(frame, p, 1, palette!![pointColor!![j]], 2)
                }
            }
            val bbox: FloatArray? =
                floatArrayOf(pt.bbox.left, pt.bbox.top, pt.bbox.right, pt.bbox.bottom)
            for (j in 0..3) {
                bbox!![j] *= scale
            }
            Imgproc.rectangle(
                frame, Point(bbox!![0].toDouble(), bbox[1].toDouble()),
                Point(bbox[2].toDouble(), bbox[3].toDouble()), Scalar(0.0, 255.0, 0.0)
            )
        }
    }

    /**
     * 计算两点之间的中点坐标
     *
     * @param pt1 第一个点
     * @param pt2 第二个点
     * @return 中点坐标
     */
    fun getMidPoint(pt1: Point, pt2: Point): Point {
        return Point(
            (pt1.x + pt2.x) / 2.0,  // x坐标取平均值
            (pt1.y + pt2.y) / 2.0 // y坐标取平均值
        )
    }
}
