package com.mmdeploy.posetracker

import mmdeploy.DataType
import mmdeploy.Mat
import mmdeploy.PixelFormat

/**
 * @description: this is a util class for java demo.
 */
object MatUtils {
    /**
     * This function changes cvMat to Mat.
     *
     * @param cvMat: the image with opencv Mat format.
     * @return: the image with Mat format.
     */
    fun cvMatToMat(cvMat: org.opencv.core.Mat): Mat {
        val dataPointer =
            ByteArray(cvMat.rows() * cvMat.cols() * cvMat.channels() * cvMat.elemSize().toInt())
        cvMat.get(0, 0, dataPointer)
        return Mat(
            cvMat.rows(), cvMat.cols(), cvMat.channels(),
            PixelFormat.BGR, DataType.INT8, dataPointer
        )
    }
}
