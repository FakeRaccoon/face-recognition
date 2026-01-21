import 'dart:typed_data';
import 'package:image/image.dart' as img;

class CameraImageMessage {
  final List<CameraPlaneMessage> planes;
  final int width;
  final int height;
  final int sensorOrientation;
  final bool isAndroid;
  final bool isIOS;

  CameraImageMessage({
    required this.planes,
    required this.width,
    required this.height,
    required this.sensorOrientation,
    required this.isAndroid,
    required this.isIOS,
  });
}

class CameraPlaneMessage {
  final Uint8List bytes;
  final int bytesPerRow;
  final int? bytesPerPixel;

  CameraPlaneMessage({
    required this.bytes,
    required this.bytesPerRow,
    this.bytesPerPixel,
  });
}

/// Function to be run in an Isolate
img.Image? convertCameraImageToUpright(CameraImageMessage message) {
  try {
    img.Image? image;

    if (message.isAndroid) {
      image = _convertYUV420ToImage(message);
    } else if (message.isIOS) {
      image = _convertBGRA8888ToImage(message);
    }

    if (image == null) return null;

    // Rotate image to upright based on sensor orientation
    if (message.sensorOrientation != 0) {
      image = img.copyRotate(
        image,
        angle: message.sensorOrientation.toDouble(),
      );
    }

    // Mirroring removed for performance (and potential accuracy) boost
    // if (message.isFrontCamera) {
    //   image = img.flipHorizontal(image);
    // }

    return image;
  } catch (e) {
    return null;
  }
}

img.Image? _convertYUV420ToImage(CameraImageMessage message) {
  final width = message.width;
  final height = message.height;
  final image = img.Image(width: width, height: height);
  final planes = message.planes;

  try {
    final planeCount = planes.length;

    if (planeCount == 1) {
      // NV21 single plane format: Y followed by interleaved VU
      final bytes = planes[0].bytes;
      final int ySize = width * height;

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * width + x;
          final int yValue = bytes[yIndex];

          // VU data starts after Y plane, interleaved
          final int uvIndex = ySize + (y ~/ 2) * width + (x ~/ 2) * 2;
          final int vValue = bytes[uvIndex]; // V first in NV21
          final int uValue = bytes[uvIndex + 1]; // U second

          image.setPixelRgb(
            x,
            y,
            _yuv2rgb(yValue, uValue, vValue, 0),
            _yuv2rgb(yValue, uValue, vValue, 1),
            _yuv2rgb(yValue, uValue, vValue, 2),
          );
        }
      }
    } else if (planeCount >= 3) {
      // YUV420 with separate planes
      final yPlane = planes[0];
      final uPlane = planes[1];
      final vPlane = planes[2];

      final int yRowStride = yPlane.bytesPerRow;
      final int uvRowStride = uPlane.bytesPerRow;
      final int uvPixelStride = uPlane.bytesPerPixel ?? 1;

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * yRowStride + x;
          final int yValue = yPlane.bytes[yIndex];

          final int uvY = y ~/ 2;
          final int uvX = x ~/ 2;
          final int uvIndex = uvY * uvRowStride + uvX * uvPixelStride;

          final int uValue = uPlane.bytes[uvIndex];
          final int vValue = vPlane.bytes[uvIndex];

          image.setPixelRgb(
            x,
            y,
            _yuv2rgb(yValue, uValue, vValue, 0),
            _yuv2rgb(yValue, uValue, vValue, 1),
            _yuv2rgb(yValue, uValue, vValue, 2),
          );
        }
      }
    }
    return image;
  } catch (e) {
    return null;
  }
}

img.Image? _convertBGRA8888ToImage(CameraImageMessage message) {
  final plane = message.planes[0];
  final width = message.width;
  final height = message.height;
  final image = img.Image(width: width, height: height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      final index = y * plane.bytesPerRow + x * 4;
      final b = plane.bytes[index];
      final g = plane.bytes[index + 1];
      final r = plane.bytes[index + 2];

      image.setPixelRgb(x, y, r, g, b);
    }
  }
  return image;
}

int _yuv2rgb(int y, int u, int v, int channel) {
  int r = (y + 1.370705 * (v - 128)).round().clamp(0, 255);
  int g = (y - 0.337633 * (u - 128) - 0.698001 * (v - 128)).round().clamp(
    0,
    255,
  );
  int b = (y + 1.732446 * (u - 128)).round().clamp(0, 255);

  if (channel == 0) return r;
  if (channel == 1) return g;
  return b;
}

/// Function to be run in an Isolate for decoding images
img.Image? decodeImageIsolate(Uint8List bytes) {
  try {
    return img.decodeImage(bytes);
  } catch (e) {
    return null;
  }
}

/// Function to be run in an Isolate for YUV conversion
Uint8List? convertYUV420ToNV21(CameraImageMessage message) {
  try {
    final int width = message.width;
    final int height = message.height;

    // NV21 size is Width * Height * 1.5
    final int ySize = width * height;
    final int uvSize = width * height ~/ 2;
    final Uint8List nv21 = Uint8List(ySize + uvSize);

    // Reconstruct planes from message
    // Note: CameraImageMessage structure might simplify planes to just bytes/strides
    // We assume message.planes[0] is Y, [1] is U, [2] is V based on input construction

    final yPlane = message.planes[0];
    final uPlane = message.planes[1];
    final vPlane = message.planes[2];

    final int yRowStride = yPlane.bytesPerRow;
    final int yPixelStride = yPlane.bytesPerPixel ?? 1;
    final int uvRowStride = uPlane.bytesPerRow;
    final int uvPixelStride = uPlane.bytesPerPixel ?? 1;

    // Copy Y plane
    var nv21Index = 0;
    for (int y = 0; y < height; y++) {
      final int srcOffset = y * yRowStride;
      for (int x = 0; x < width; x++) {
        nv21[nv21Index++] = yPlane.bytes[srcOffset + x * yPixelStride];
      }
    }

    // Copy UV planes (Interleaved V then U for NV21)
    // UV planes are subsampled 2x2
    for (int y = 0; y < height ~/ 2; y++) {
      final int srcRowOffset = y * uvRowStride;
      for (int x = 0; x < width ~/ 2; x++) {
        final int srcPixelOffset = srcRowOffset + x * uvPixelStride;

        final int v = vPlane.bytes[srcPixelOffset];
        final int u = uPlane.bytes[srcPixelOffset];

        nv21[nv21Index++] = v;
        nv21[nv21Index++] = u;
      }
    }

    return nv21;
  } catch (e) {
    return null;
  }
}
