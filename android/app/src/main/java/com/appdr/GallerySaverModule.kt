package com.appdr

import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

class GallerySaverModule(private val reactContext: ReactApplicationContext) :
  ReactContextBaseJavaModule(reactContext) {

  override fun getName(): String = "DRGallerySaver"

  @ReactMethod
  fun saveImage(filePath: String, albumName: String, promise: Promise) {
    try {
      val source = getSourceFile(filePath)

      if (!source.exists()) {
        promise.reject("SOURCE_NOT_FOUND", "Captured photo file was not found.")
        return
      }

      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        saveWithMediaStore(source, albumName, promise)
      } else {
        saveLegacy(source, albumName, promise)
      }
    } catch (error: Exception) {
      promise.reject("SAVE_FAILED", error.message, error)
    }
  }

  private fun getSourceFile(filePath: String): File {
    return if (filePath.startsWith("file://")) {
      File(Uri.parse(filePath).path ?: "")
    } else {
      File(filePath)
    }
  }

  private fun saveWithMediaStore(source: File, albumName: String, promise: Promise) {
    val resolver = reactContext.contentResolver
    val values = android.content.ContentValues().apply {
      put(MediaStore.Images.Media.DISPLAY_NAME, source.name)
      put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
      put(
        MediaStore.Images.Media.RELATIVE_PATH,
        "${Environment.DIRECTORY_DCIM}/$albumName",
      )
      put(MediaStore.Images.Media.IS_PENDING, 1)
    }

    val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)

    if (uri == null) {
      promise.reject("SAVE_FAILED", "Could not create gallery entry.")
      return
    }

    resolver.openOutputStream(uri).use { output ->
      FileInputStream(source).use { input ->
        if (output == null) {
          promise.reject("SAVE_FAILED", "Could not open gallery output stream.")
          return
        }

        input.copyTo(output)
      }
    }

    values.clear()
    values.put(MediaStore.Images.Media.IS_PENDING, 0)
    resolver.update(uri, values, null, null)
    promise.resolve(uri.toString())
  }

  private fun saveLegacy(source: File, albumName: String, promise: Promise) {
    val picturesDir = Environment.getExternalStoragePublicDirectory(
      Environment.DIRECTORY_PICTURES,
    )
    val albumDir = File(picturesDir, albumName)

    if (!albumDir.exists() && !albumDir.mkdirs()) {
      promise.reject("SAVE_FAILED", "Could not create album folder.")
      return
    }

    val destination = getUniqueDestination(albumDir, source.name)

    FileInputStream(source).use { input ->
      FileOutputStream(destination).use { output ->
        input.copyTo(output)
      }
    }

    MediaScannerConnection.scanFile(
      reactContext,
      arrayOf(destination.absolutePath),
      arrayOf("image/jpeg"),
    ) { _, uri ->
      if (uri == null) {
        promise.reject("SAVE_FAILED", "Saved file could not be added to gallery.")
      } else {
        promise.resolve(uri.toString())
      }
    }
  }

  private fun getUniqueDestination(folder: File, fileName: String): File {
    val dotIndex = fileName.lastIndexOf('.')
    val baseName = if (dotIndex >= 0) fileName.substring(0, dotIndex) else fileName
    val extension = if (dotIndex >= 0) fileName.substring(dotIndex) else ".jpg"
    var destination = File(folder, fileName)
    var index = 1

    while (destination.exists()) {
      destination = File(folder, "${baseName}_$index$extension")
      index += 1
    }

    return destination
  }
}
