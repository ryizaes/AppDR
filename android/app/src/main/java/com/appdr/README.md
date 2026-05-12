# Android Native Code

This folder contains Kotlin code that is exposed to React Native.

## Files

- `GallerySaverModule.kt` implements the `DRGallerySaver` native module. It
  copies captured JPEG files into the Android gallery.
- `ImageCropModule.kt` implements the `DRImageCropper` native module. It creates
  the cropped analysis copy that matches the centered capture square.
- `GallerySaverPackage.kt` registers `DRGallerySaver` with React Native.
- `MainApplication.kt` wires React Native packages into the Android app.
- `MainActivity.kt` hosts the React Native UI.

## Gallery Saving

`GallerySaverModule` supports two Android storage paths:

- Android 10 and newer use `MediaStore`.
- Android 9 and older use the public pictures directory and then notify the
  media scanner.

The JavaScript side calls:

```ts
NativeModules.DRGallerySaver.saveImage(filePath, 'DR_Screening')
```

## Analysis Crop

`ImageCropModule` decodes the captured photo, applies EXIF rotation, crops the
largest centered square, and writes a JPEG copy into the app cache. The frontend
uploads this cropped copy to FastAPI so analysis is limited to the visible square
capture target.

## Networking Note

The Android manifest allows local cleartext HTTP traffic because the FastAPI
development backend runs on `http://127.0.0.1:8000` through USB reverse or on a
local network address during testing.
