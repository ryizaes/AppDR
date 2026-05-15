package com.appdr

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import com.facebook.react.bridge.ActivityEventListener
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import java.io.File
import java.io.FileOutputStream

class ImagePickerModule(private val reactContext: ReactApplicationContext) :
  ReactContextBaseJavaModule(reactContext),
  ActivityEventListener {

  private var pendingPromise: Promise? = null

  init {
    reactContext.addActivityEventListener(this)
  }

  override fun getName(): String = "DRImagePicker"

  @ReactMethod
  fun pickImage(promise: Promise) {
    val activity = getCurrentActivity()

    if (activity == null) {
      promise.reject("NO_ACTIVITY", "Image picker cannot open because the app is not active.")
      return
    }

    if (pendingPromise != null) {
      promise.reject("PICKER_ACTIVE", "Image picker is already open.")
      return
    }

    pendingPromise = promise

    val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
      addCategory(Intent.CATEGORY_OPENABLE)
      type = "image/*"
      addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }

    try {
      activity.startActivityForResult(intent, PICK_IMAGE_REQUEST)
    } catch (error: Exception) {
      pendingPromise = null
      promise.reject("PICKER_OPEN_FAILED", error.message, error)
    }
  }

  override fun onActivityResult(
    activity: Activity,
    requestCode: Int,
    resultCode: Int,
    data: Intent?,
  ) {
    if (requestCode != PICK_IMAGE_REQUEST) {
      return
    }

    val promise = pendingPromise ?: return
    pendingPromise = null

    if (resultCode != Activity.RESULT_OK) {
      promise.reject("PICKER_CANCELLED", "No image was selected.")
      return
    }

    val uri = data?.data

    if (uri == null) {
      promise.reject("PICKER_EMPTY", "Selected image could not be read.")
      return
    }

    try {
      val pickedImage = copyImageToCache(uri)
      promise.resolve(pickedImage)
    } catch (error: Exception) {
      promise.reject("PICKER_COPY_FAILED", error.message, error)
    }
  }

  override fun onNewIntent(intent: Intent) = Unit

  private fun copyImageToCache(uri: Uri): com.facebook.react.bridge.WritableMap {
    val resolver = reactContext.contentResolver
    val mimeType = resolver.getType(uri) ?: "image/jpeg"
    val displayName = sanitizeFileName(getDisplayName(uri) ?: "retinal-upload")
    val extension = getExtension(displayName, mimeType)
    val baseName = displayName.substringBeforeLast('.', displayName)
    val output = File(
      reactContext.cacheDir,
      "dr_upload_${System.currentTimeMillis()}_${baseName}.${extension}",
    )

    resolver.openInputStream(uri).use { input ->
      if (input == null) {
        throw IllegalStateException("Selected image stream could not be opened.")
      }

      FileOutputStream(output).use { outputStream ->
        input.copyTo(outputStream)
      }
    }

    return Arguments.createMap().apply {
      putString("filePath", output.absolutePath)
      putString("fileUri", "file://${output.absolutePath}")
      putString("name", output.name)
      putString("type", mimeType)
    }
  }

  private fun getDisplayName(uri: Uri): String? {
    reactContext.contentResolver.query(uri, null, null, null, null).use { cursor ->
      if (cursor == null || !cursor.moveToFirst()) {
        return null
      }

      val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
      return if (nameIndex >= 0) cursor.getString(nameIndex) else null
    }
  }

  private fun sanitizeFileName(fileName: String): String =
    fileName.replace(Regex("[^A-Za-z0-9._-]"), "_").take(80).ifBlank {
      "retinal-upload"
    }

  private fun getExtension(fileName: String, mimeType: String): String {
    val extension = fileName.substringAfterLast('.', missingDelimiterValue = "")

    if (extension.isNotBlank() && extension.length <= 5) {
      return extension.lowercase()
    }

    return when (mimeType.lowercase()) {
      "image/png" -> "png"
      "image/webp" -> "webp"
      else -> "jpg"
    }
  }

  companion object {
    private const val PICK_IMAGE_REQUEST = 4201
  }
}
