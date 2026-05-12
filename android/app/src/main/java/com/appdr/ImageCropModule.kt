package com.appdr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import java.io.File
import java.io.FileOutputStream

class ImageCropModule(private val reactContext: ReactApplicationContext) :
  ReactContextBaseJavaModule(reactContext) {

  override fun getName(): String = "DRImageCropper"

  @ReactMethod
  fun cropCenterSquare(filePath: String, cropScale: Double, promise: Promise) {
    try {
      val source = getSourceFile(filePath)

      if (!source.exists()) {
        promise.reject("SOURCE_NOT_FOUND", "Captured photo file was not found.")
        return
      }

      val decoded = BitmapFactory.decodeFile(source.absolutePath)

      if (decoded == null) {
        promise.reject("DECODE_FAILED", "Captured photo could not be decoded.")
        return
      }

      val oriented = applyExifOrientation(decoded, source.absolutePath)
      val safeScale = cropScale.coerceIn(0.1, 1.0)
      val cropSize = (minOf(oriented.width, oriented.height) * safeScale).toInt()
        .coerceAtLeast(1)
      val left = ((oriented.width - cropSize) / 2).coerceAtLeast(0)
      val top = ((oriented.height - cropSize) / 2).coerceAtLeast(0)
      val cropped = Bitmap.createBitmap(oriented, left, top, cropSize, cropSize)
      val output = File(
        reactContext.cacheDir,
        "dr_analysis_${System.currentTimeMillis()}.jpg",
      )

      FileOutputStream(output).use { stream ->
        cropped.compress(Bitmap.CompressFormat.JPEG, 94, stream)
      }

      if (cropped != oriented) {
        cropped.recycle()
      }
      if (oriented != decoded) {
        oriented.recycle()
      }
      decoded.recycle()

      promise.resolve(output.absolutePath)
    } catch (error: Exception) {
      promise.reject("CROP_FAILED", error.message, error)
    }
  }

  private fun getSourceFile(filePath: String): File {
    return if (filePath.startsWith("file://")) {
      File(Uri.parse(filePath).path ?: "")
    } else {
      File(filePath)
    }
  }

  private fun applyExifOrientation(bitmap: Bitmap, filePath: String): Bitmap {
    val orientation = ExifInterface(filePath).getAttributeInt(
      ExifInterface.TAG_ORIENTATION,
      ExifInterface.ORIENTATION_NORMAL,
    )
    val rotation = when (orientation) {
      ExifInterface.ORIENTATION_ROTATE_90 -> 90f
      ExifInterface.ORIENTATION_ROTATE_180 -> 180f
      ExifInterface.ORIENTATION_ROTATE_270 -> 270f
      else -> 0f
    }

    if (rotation == 0f) {
      return bitmap
    }

    val matrix = Matrix().apply { postRotate(rotation) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
  }
}
