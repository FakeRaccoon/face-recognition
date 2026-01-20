import 'dart:developer' show log;
import 'dart:io';
import 'dart:math' hide log;
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'services/face_recognition_service.dart' as recognition;
import 'face_comparison_screen.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Recognition',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const FaceDetectionScreen(),
    );
  }
}

class FaceDetectionScreen extends StatefulWidget {
  const FaceDetectionScreen({super.key});

  @override
  State<FaceDetectionScreen> createState() => _FaceDetectionScreenState();
}

class DetectedFaceInfo {
  final Face face;
  final String? recognizedName;
  final double? confidence;
  final String? bestMatchName;
  final double? bestMatchScore;

  DetectedFaceInfo({
    required this.face,
    this.recognizedName,
    this.confidence,
    this.bestMatchName,
    this.bestMatchScore,
  });
}

class _FaceDetectionScreenState extends State<FaceDetectionScreen> {
  CameraController? _cameraController;
  bool _isDetecting = false;
  bool _isCameraPaused = false;
  List<DetectedFaceInfo> _detectedFaces = [];
  Uint8List? _debugLiveFaceBytes;
  Uint8List? _capturedImageBytes;
  int _cameraIndex = 0;
  DateTime? _lastRecognitionTime;

  final recognition.FaceRecognitionService _recognitionService =
      recognition.FaceRecognitionService();
  final ImagePicker _imagePicker = ImagePicker();
  bool _isRecognitionReady = false;

  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableLandmarks: true,
      enableClassification: false,
      enableTracking: true,
      performanceMode: FaceDetectorMode.fast,
    ),
  );

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  Future<void> _initializeServices() async {
    await _initializeCamera();
    try {
      await _recognitionService.initialize();
      if (mounted) {
        setState(() {
          _isRecognitionReady = true;
        });
      }
    } catch (e) {
      log('Error initializing recognition service: $e');
    }
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) return;

    _cameraIndex = cameras.indexWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
    );
    if (_cameraIndex == -1) _cameraIndex = 0;

    _cameraController = CameraController(
      cameras[_cameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.nv21
          : ImageFormatGroup.bgra8888,
    );

    try {
      await _cameraController!.initialize();
      await _cameraController!.startImageStream(_processCameraImage);
      if (mounted) setState(() {});
    } catch (e) {
      log('Error initializing camera: $e');
    }
  }

  Future<void> _processCameraImage(CameraImage cameraImage) async {
    if (_isDetecting) return;
    _isDetecting = true;

    try {
      // Use original method for ML Kit detection (handles rotation properly)
      final inputImage = _convertCameraImage(cameraImage);
      if (inputImage == null) {
        _isDetecting = false;
        return;
      }

      final faces = await _faceDetector.processImage(inputImage);
      final List<DetectedFaceInfo> faceInfos = [];
      Uint8List? currentLiveFaceBytes;

      if (_isRecognitionReady &&
          _recognitionService.registeredFaces.isNotEmpty) {
        // Debounce: only perform recognition every 500ms to avoid freezing
        final now = DateTime.now();
        if (_lastRecognitionTime != null &&
            now.difference(_lastRecognitionTime!) <
                const Duration(milliseconds: 500)) {
          // Skip recognition, keep previous results to avoid flickering
          _isDetecting = false;
          return;
        }
        _lastRecognitionTime = now;

        // Get camera info for debug
        final camera = cameras[_cameraIndex];
        final sensorOrientation = camera.sensorOrientation;
        final isFrontCamera = camera.lensDirection == CameraLensDirection.front;

        log('\n=== REALTIME DEBUG ===');
        log('┌─ Camera Info ─────────────');
        log('│ Sensor orientation: $sensorOrientation°');
        log('│ Front camera: $isFrontCamera');
        log('│ Raw image: ${cameraImage.width}×${cameraImage.height}');
        log('│ Preview size: ${_cameraController!.value.previewSize}');
        log('└──────────────────────────');

        log('┌─ Face Detection ───────────');
        log('│ Faces detected: ${faces.length}');
        if (faces.isNotEmpty) {
          final bbox = faces[0].boundingBox;
          log(
            '│ Face bbox: L=${bbox.left.toInt()}, T=${bbox.top.toInt()}, R=${bbox.right.toInt()}, B=${bbox.bottom.toInt()}',
          );
        }
        log('└──────────────────────────');

        // Convert raw camera frame to upright image FIRST
        final uprightImage = _convertCameraImageToUpright(cameraImage);
        if (uprightImage == null) {
          _isDetecting = false;
          return;
        }

        log('Upright image: ${uprightImage.width}x${uprightImage.height}');

        // For comparison - log what capture does
        log(
          'Note: Capture uses takePicture() which creates properly oriented JPEG',
        );

        // ML Kit detected faces on rotated image and returned upright bounding boxes
        // Since we also have upright image, bounding boxes match directly
        for (final face in faces) {
          final croppedFace = _recognitionService.cropFace(
            uprightImage,
            recognition.Rect(
              left: face.boundingBox.left,
              top: face.boundingBox.top,
              right: face.boundingBox.right,
              bottom: face.boundingBox.bottom,
            ),
          );

          if (croppedFace != null) {
            // Store debug image for first face

            log('Cropped face: ${croppedFace.width}x${croppedFace.height}');
            final result = await _recognitionService.recognizeFace(croppedFace);
            log(
              'Recognition: name=${result?.name}, conf=${result?.confidence}, match=${result?.isMatch}',
            );

            faceInfos.add(
              DetectedFaceInfo(
                face: face,
                recognizedName: result?.isMatch == true ? result?.name : null,
                confidence: result?.isMatch == true ? result?.confidence : null,
                bestMatchName: result?.name,
                bestMatchScore: result?.confidence,
              ),
            );
          } else {
            log('Cropped face is null');
            faceInfos.add(DetectedFaceInfo(face: face));
          }
        }
      } else {
        for (final face in faces) {
          faceInfos.add(DetectedFaceInfo(face: face));
        }
      }

      if (mounted) {
        setState(() {
          _detectedFaces = faceInfos;
          _debugLiveFaceBytes = currentLiveFaceBytes;
        });
      }
    } catch (e) {
      log('Error detecting faces: $e');
    }

    _isDetecting = false;
  }

  img.Image? _convertCameraImageToUpright(CameraImage cameraImage) {
    try {
      final camera = cameras[_cameraIndex];
      final sensorOrientation = camera.sensorOrientation;
      final isFrontCamera = camera.lensDirection == CameraLensDirection.front;

      log(
        'Converting to upright: sensorOrientation=$sensorOrientation, isFront=$isFrontCamera',
      );

      // Convert to image first
      final rawImage = _convertCameraImageToImg(cameraImage);
      if (rawImage == null) {
        log('FAILED: _convertCameraImageToImg returned null');
        return null;
      }

      log('Raw image converted: ${rawImage.width}x${rawImage.height}');

      // Rotate image to upright based on sensor orientation
      img.Image uprightImage = rawImage;
      if (sensorOrientation != 0) {
        log('Rotating by $sensorOrientation degrees...');
        // Rotate counter-clockwise to make it upright
        uprightImage = img.copyRotate(
          rawImage,
          angle: sensorOrientation.toDouble(),
        );
        log('After rotation: ${uprightImage.width}x${uprightImage.height}');
      }

      // Flip horizontally for front camera to match mirror behavior
      if (isFrontCamera) {
        log('Flipping horizontally for front camera...');
        uprightImage = img.flipHorizontal(uprightImage);
        log('After flip: ${uprightImage.width}x${uprightImage.height}');
      }

      log('SUCCESS: Upright image ready');
      return uprightImage;
    } catch (e, stackTrace) {
      log('ERROR converting camera image to upright: $e');
      log('Stack trace: $stackTrace');
      return null;
    }
  }

  img.Image? _convertCameraImageToImg(CameraImage cameraImage) {
    try {
      if (Platform.isAndroid) {
        return _convertYUV420ToImage(cameraImage);
      } else if (Platform.isIOS) {
        return _convertBGRA8888ToImage(cameraImage);
      }
    } catch (e) {
      log('Error converting camera image: $e');
    }
    return null;
  }

  img.Image? _convertYUV420ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    final image = img.Image(width: width, height: height);

    try {
      final int planeCount = cameraImage.planes.length;
      log('YUV planes count: $planeCount, dims: ${width}x$height');

      if (planeCount == 1) {
        // NV21 single plane format: Y followed by interleaved VU
        final bytes = cameraImage.planes[0].bytes;
        final int ySize = width * height;

        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final int yIndex = y * width + x;
            final int yValue = bytes[yIndex];

            // VU data starts after Y plane, interleaved
            final int uvIndex = ySize + (y ~/ 2) * width + (x ~/ 2) * 2;
            final int vValue = bytes[uvIndex]; // V first in NV21
            final int uValue = bytes[uvIndex + 1]; // U second

            // YUV to RGB conversion
            final r =
                (yValue + 1.370705 * (vValue - 128)).clamp(0, 255).toInt();
            final g =
                (yValue - 0.337633 * (uValue - 128) - 0.698001 * (vValue - 128))
                    .clamp(0, 255)
                    .toInt();
            final b =
                (yValue + 1.732446 * (uValue - 128)).clamp(0, 255).toInt();

            image.setPixelRgb(x, y, r, g, b);
          }
        }
      } else if (planeCount >= 3) {
        // YUV420 with separate planes
        final yPlane = cameraImage.planes[0];
        final uPlane = cameraImage.planes[1];
        final vPlane = cameraImage.planes[2];

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

            final r =
                (yValue + 1.370705 * (vValue - 128)).clamp(0, 255).toInt();
            final g =
                (yValue - 0.337633 * (uValue - 128) - 0.698001 * (vValue - 128))
                    .clamp(0, 255)
                    .toInt();
            final b =
                (yValue + 1.732446 * (uValue - 128)).clamp(0, 255).toInt();

            image.setPixelRgb(x, y, r, g, b);
          }
        }
      } else {
        log('Unsupported plane count: $planeCount');
        return null;
      }
    } catch (e) {
      log('Error converting YUV: $e');
      log(
        'Planes: ${cameraImage.planes.map((p) => 'len=${p.bytes.length}, row=${p.bytesPerRow}, pixel=${p.bytesPerPixel}').join(', ')}',
      );
      return null;
    }

    return image;
  }

  img.Image? _convertBGRA8888ToImage(CameraImage cameraImage) {
    final plane = cameraImage.planes[0];
    final width = cameraImage.width;
    final height = cameraImage.height;

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

    // DO NOT rotate or flip - ML Kit bounding boxes are in raw image coordinates
    return image;
  }

  InputImage? _convertCameraImage(CameraImage image) {
    final camera = cameras[_cameraIndex];
    final sensorOrientation = camera.sensorOrientation;
    InputImageRotation? rotation;

    if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else if (Platform.isAndroid) {
      var rotationCompensation = sensorOrientation;
      if (camera.lensDirection == CameraLensDirection.front) {
        rotationCompensation = (sensorOrientation + 360) % 360;
      }
      rotation = InputImageRotationValue.fromRawValue(rotationCompensation);
    }

    if (rotation == null) return null;

    final format = InputImageFormatValue.fromRawValue(image.format.raw);

    // Allow both NV21 (17) and YUV420 (35)
    if (format == null ||
        (Platform.isAndroid &&
            format != InputImageFormat.nv21 &&
            format != InputImageFormat.yuv420) ||
        (Platform.isIOS && format != InputImageFormat.bgra8888)) {
      return null;
    }

    if (image.planes.isEmpty) return null;

    if (Platform.isAndroid && format == InputImageFormat.yuv420) {
      // Manually convert YUV420 planes to NV21 byte buffer for ML Kit
      final nv21Bytes = _yuv420ToNv21(image);
      return InputImage.fromBytes(
        bytes: nv21Bytes,
        metadata: InputImageMetadata(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          rotation: rotation,
          format: InputImageFormat.nv21, // We converted it to NV21
          bytesPerRow: image.width, // NV21 stride is usually width
        ),
      );
    }

    // Handle NV21 with multiple planes (Y + VU)
    if (Platform.isAndroid &&
        format == InputImageFormat.nv21 &&
        image.planes.length > 1) {
      // Manually concatenate all planes
      final int totalLength = image.planes.fold<int>(
        0,
        (sum, plane) => sum + plane.bytes.length,
      );
      final Uint8List bytes = Uint8List(totalLength);
      int offset = 0;
      for (final Plane plane in image.planes) {
        bytes.setRange(offset, offset + plane.bytes.length, plane.bytes);
        offset += plane.bytes.length;
      }

      return InputImage.fromBytes(
        bytes: bytes,
        metadata: InputImageMetadata(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          rotation: rotation,
          format: format,
          bytesPerRow: image.planes[0].bytesPerRow,
        ),
      );
    }

    // Fallback/Standard single plane
    return InputImage.fromBytes(
      bytes: image.planes[0].bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: image.planes[0].bytesPerRow,
      ),
    );
  }

  Uint8List _yuv420ToNv21(CameraImage image) {
    final int width = image.width;
    final int height = image.height;

    // NV21 size is Width * Height * 1.5
    final int ySize = width * height;
    final int uvSize = width * height ~/ 2;
    final Uint8List nv21 = Uint8List(ySize + uvSize);

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

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
  }

  Future<void> _captureAndRecognize() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    // Pause camera stream
    await _cameraController!.stopImageStream();
    setState(() {
      _isCameraPaused = true;
    });

    try {
      // Capture image
      final XFile? capturedFile = await _cameraController!.takePicture();
      if (capturedFile == null) {
        _showSnackBar('Failed to capture image');
        await _resumeCamera();
        return;
      }

      // Read captured image
      final bytes = await capturedFile.readAsBytes();

      // Process captured image
      final image = img.decodeImage(bytes);
      if (image == null) {
        _showSnackBar('Could not decode captured image');
        return;
      }

      // Flip captured image horizontally for front camera to match mirror behavior
      img.Image displayImage = image;
      final isFrontCamera =
          cameras[_cameraIndex].lensDirection == CameraLensDirection.front;
      if (isFrontCamera) {
        displayImage = img.flipHorizontal(image);
      }

      // Update state with flipped image for display
      setState(() {
        _capturedImageBytes = img.encodePng(displayImage);
      });

      // Detect face in captured image
      final inputImage = InputImage.fromFilePath(capturedFile.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        _showSnackBar('No face detected in captured image');
        return;
      }

      // Find largest face
      Face? largestFace;
      double maxArea = 0;
      for (final face in faces) {
        final area = face.boundingBox.width * face.boundingBox.height;
        if (area > maxArea) {
          maxArea = area;
          largestFace = face;
        }
      }

      if (largestFace == null) {
        _showSnackBar('Could not find face in image');
        return;
      }

      // Crop face from captured image
      final croppedFace = _recognitionService.cropFace(
        image,
        recognition.Rect(
          left: largestFace.boundingBox.left,
          top: largestFace.boundingBox.top,
          right: largestFace.boundingBox.right,
          bottom: largestFace.boundingBox.bottom,
        ),
      );

      if (croppedFace == null) {
        _showSnackBar('Could not crop face from image');
        return;
      }

      // Recognize face
      if (_recognitionService.registeredFaces.isNotEmpty) {
        final result = await _recognitionService.recognizeFace(croppedFace);

        // Update detected faces with recognition result
        final List<DetectedFaceInfo> faceInfos = [];
        faceInfos.add(
          DetectedFaceInfo(
            face: largestFace,
            recognizedName: result?.isMatch == true ? result?.name : null,
            confidence: result?.isMatch == true ? result?.confidence : null,
            bestMatchName: result?.name,
            bestMatchScore: result?.confidence,
          ),
        );

        setState(() {
          _detectedFaces = faceInfos;
        });

        if (result != null && result.isMatch) {
          _showSnackBar(
            'Matched: ${result.name} (${(result.confidence * 100).toStringAsFixed(1)}%)',
          );
        } else {
          _showSnackBar('No match found');
        }
      } else {
        _showSnackBar('No registered faces to compare');
      }
    } catch (e) {
      log('Error capturing and recognizing: $e');
      _showSnackBar('Error during capture');
    }
  }

  Future<void> _resumeCamera() async {
    if (_cameraController == null) return;

    try {
      await _cameraController!.startImageStream(_processCameraImage);
      setState(() {
        _isCameraPaused = false;
        _capturedImageBytes = null;
      });
    } catch (e) {
      log('Error resuming camera: $e');
    }
  }

  Future<void> _registerFaceFromGallery() async {
    try {
      final XFile? pickedFile = await _imagePicker.pickImage(
        source: ImageSource.gallery,
      );

      if (pickedFile == null) return;

      final bytes = await pickedFile.readAsBytes();
      final image = img.decodeImage(bytes);
      if (image == null) {
        _showSnackBar('Could not decode image');
        return;
      }

      // Detect face in the selected image
      final inputImage = InputImage.fromFilePath(pickedFile.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        _showSnackBar('No face detected in the image');
        return;
      }

      if (faces.length > 1) {
        _showSnackBar(
          'Multiple faces detected. Please select an image with one face.',
        );
        return;
      }

      final face = faces.first;
      final croppedFace = _recognitionService.cropFace(
        image,
        recognition.Rect(
          left: face.boundingBox.left,
          top: face.boundingBox.top,
          right: face.boundingBox.right,
          bottom: face.boundingBox.bottom,
        ),
      );

      if (croppedFace == null) {
        _showSnackBar('Could not crop face from image');
        return;
      }

      // Show dialog to enter name
      final name = await _showNameInputDialog();
      if (name == null || name.isEmpty) return;

      // Limit to 1 face: Clear existing before registering new
      _recognitionService.clearAllFaces();

      await _recognitionService.registerFace(name, croppedFace);
      _showSnackBar('Face registered for $name');
      setState(() {});
    } catch (e) {
      log('Error registering face: $e');
      _showSnackBar('Error registering face');
    }
  }

  Future<String?> _showNameInputDialog() async {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Register Face'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(
            labelText: 'Enter name',
            hintText: 'e.g., John',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, controller.text.trim()),
            child: const Text('Register'),
          ),
        ],
      ),
    );
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 2)),
    );
  }

  void _showRegisteredFaces() {
    showModalBottomSheet(
      context: context,
      builder: (context) => Container(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Registered Faces',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                if (_recognitionService.registeredFaces.isNotEmpty)
                  TextButton(
                    onPressed: () {
                      _recognitionService.clearAllFaces();
                      Navigator.pop(context);
                      setState(() {});
                      _showSnackBar('All faces cleared');
                    },
                    child: const Text('Clear All'),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            if (_recognitionService.registeredFaces.isEmpty)
              const Padding(
                padding: EdgeInsets.all(16),
                child: Text('No faces registered yet'),
              )
            else
              ...(_recognitionService.registeredFaces.map(
                (face) => ListTile(
                  leading: face.faceBytes != null
                      ? ClipOval(
                          child: Image.memory(
                            face.faceBytes!,
                            width: 40,
                            height: 40,
                            fit: BoxFit.cover,
                          ),
                        )
                      : const Icon(Icons.face),
                  title: Text(face.name),
                  trailing: IconButton(
                    icon: const Icon(Icons.delete),
                    onPressed: () {
                      _recognitionService.removeFace(face.name);
                      Navigator.pop(context);
                      setState(() {});
                      _showSnackBar('${face.name} removed');
                    },
                  ),
                ),
              )),
          ],
        ),
      ),
    );
  }

  Future<void> _switchCamera() async {
    if (cameras.length < 2) return;

    await _cameraController?.stopImageStream();
    await _cameraController?.dispose();

    _cameraIndex = (_cameraIndex + 1) % cameras.length;

    _cameraController = CameraController(
      cameras[_cameraIndex],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.yuv420
          : ImageFormatGroup.bgra8888,
    );

    try {
      await _cameraController!.initialize();
      await _cameraController!.startImageStream(_processCameraImage);
      if (mounted) {
        setState(() {
          _detectedFaces = [];
        });
      }
    } catch (e) {
      log('Error switching camera: $e');
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _faceDetector.close();
    _recognitionService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Face Recognition'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            icon: const Icon(Icons.compare),
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => FaceComparisonScreen(
                    recognitionService: _recognitionService,
                  ),
                ),
              );
            },
            tooltip: 'Compare Faces',
          ),
          IconButton(
            icon: const Icon(Icons.people),
            onPressed: _showRegisteredFaces,
            tooltip: 'Registered Faces',
          ),
          if (cameras.length > 1)
            IconButton(
              icon: const Icon(Icons.cameraswitch),
              onPressed: _switchCamera,
              tooltip: 'Switch Camera',
            ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          if (_isCameraPaused)
            FloatingActionButton.extended(
              onPressed: _resumeCamera,
              icon: const Icon(Icons.play_arrow),
              label: const Text('Resume'),
              backgroundColor: Colors.orange,
            )
          else
            FloatingActionButton.extended(
              onPressed: _isRecognitionReady ? _captureAndRecognize : null,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Capture'),
            ),
          const SizedBox(width: 16),
          FloatingActionButton.extended(
            onPressed: _isRecognitionReady ? _registerFaceFromGallery : null,
            icon: const Icon(Icons.add_a_photo),
            label: const Text('Register'),
          ),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(controller: _cameraController!),
        CustomPaint(
          painter: FacePainter(
            faces: _detectedFaces,
            imageSize: Size(
              _cameraController!.value.previewSize!.height,
              _cameraController!.value.previewSize!.width,
            ),
            isFrontCamera:
                cameras[_cameraIndex].lensDirection ==
                CameraLensDirection.front,
          ),
        ),
        Positioned(
          bottom: 80,
          left: 20,
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Faces: ${_detectedFaces.length}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  'Registered: ${_recognitionService.registeredFaces.length}',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
                if (!_isRecognitionReady)
                  const Text(
                    'Loading model...',
                    style: TextStyle(color: Colors.orange, fontSize: 12),
                  ),
              ],
            ),
          ),
        ),
        // Display captured image when camera is paused
        if (_capturedImageBytes != null && _isCameraPaused)
          Container(
            color: Colors.black87,
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text(
                    'Captured Image',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 20),
                  Container(
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.white, width: 3),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: Image.memory(
                        _capturedImageBytes!,
                        width: 300,
                        height: 300,
                        fit: BoxFit.contain,
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),
                  const Text(
                    'Tap "Resume" to continue',
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                ],
              ),
            ),
          ),
        // Display Registered Face Thumbnail for Debugging
        if (_recognitionService.registeredFaces.isNotEmpty &&
            _recognitionService.registeredFaces.first.faceBytes != null &&
            !_isCameraPaused)
          Positioned(
            top: 20,
            right: 20,
            child: Column(
              children: [
                Container(
                  padding: const EdgeInsets.all(4),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.green, width: 2),
                  ),
                  child: Column(
                    children: [
                      const Text(
                        'Registered',
                        style: TextStyle(color: Colors.white, fontSize: 10),
                      ),
                      const SizedBox(height: 4),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: Image.memory(
                          _recognitionService.registeredFaces.first.faceBytes!,
                          width: 80,
                          height: 80,
                          fit: BoxFit.cover,
                        ),
                      ),
                      Text(
                        _recognitionService.registeredFaces.first.name,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
                if (_debugLiveFaceBytes != null) ...[
                  const SizedBox(height: 10),
                  Container(
                    padding: const EdgeInsets.all(4),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.orange, width: 2),
                    ),
                    child: Column(
                      children: [
                        const Text(
                          'Live Input',
                          style: TextStyle(color: Colors.white, fontSize: 10),
                        ),
                        const SizedBox(height: 4),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(4),
                          child: Image.memory(
                            _debugLiveFaceBytes!,
                            width: 80,
                            height: 80,
                            fit: BoxFit.cover,
                          ),
                        ),
                        const Text(
                          'Model Input',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            ),
          ),
      ],
    );
  }
}

class CameraPreview extends StatelessWidget {
  final CameraController controller;

  const CameraPreview({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return SizedBox(
          width: constraints.maxWidth,
          height: constraints.maxHeight,
          child: FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(
              width: controller.value.previewSize!.height,
              height: controller.value.previewSize!.width,
              child: CameraPreviewWidget(controller: controller),
            ),
          ),
        );
      },
    );
  }
}

class CameraPreviewWidget extends StatelessWidget {
  final CameraController controller;

  const CameraPreviewWidget({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    return controller.buildPreview();
  }
}

class FacePainter extends CustomPainter {
  final List<DetectedFaceInfo> faces;
  final Size imageSize;
  final bool isFrontCamera;

  FacePainter({
    required this.faces,
    required this.imageSize,
    required this.isFrontCamera,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final Paint textBgPaint = Paint()
      ..color = Colors.black54
      ..style = PaintingStyle.fill;

    for (final info in faces) {
      final face = info.face;
      final color = info.recognizedName != null ? Colors.green : Colors.orange;
      paint.color = color;

      // Scale coordinates
      // ML Kit coordinates are based on the image size
      final double scaleX = size.width / imageSize.width;
      final double scaleY = size.height / imageSize.height;

      double left = face.boundingBox.left * scaleX;
      double top = face.boundingBox.top * scaleY;
      double right = face.boundingBox.right * scaleX;
      double bottom = face.boundingBox.bottom * scaleY;

      if (isFrontCamera) {
        // Mirror X for front camera
        left = size.width - right;
        right = size.width - (face.boundingBox.left * scaleX);
      }

      final rect = Rect.fromLTRB(left, top, right, bottom);
      canvas.drawRect(rect, paint);

      // Draw Landmarks (Optional, for debug)
      // _drawLandmarks(canvas, face, scaleX, scaleY, size);

      // Draw Name & Confidence
      final String label = info.recognizedName ?? 'Unknown';
      final String score = info.bestMatchScore != null
          ? '(${(info.bestMatchScore! * 100).toStringAsFixed(1)}%)'
          : '';

      final textSpan = TextSpan(
        text: '$label $score',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.bold,
        ),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );

      textPainter.layout();

      final double textX = left;
      final double textY = top - 30; // Above the box

      // Draw background for text
      canvas.drawRect(
        Rect.fromLTWH(
          textX - 5,
          textY - 5,
          textPainter.width + 10,
          textPainter.height + 10,
        ),
        textBgPaint,
      );

      textPainter.paint(canvas, Offset(textX, textY));
    }
  }

  Rect _scaleRect({
    required Rect rect,
    required Size imageSize,
    required Size widgetSize,
  }) {
    final scaleX = widgetSize.width / imageSize.width;
    final scaleY = widgetSize.height / imageSize.height;

    double left = rect.left * scaleX;
    double top = rect.top * scaleY;
    double right = rect.right * scaleX;
    double bottom = rect.bottom * scaleY;

    if (isFrontCamera) {
      final temp = left;
      left = widgetSize.width - right;
      right = widgetSize.width - temp;
    }

    return Rect.fromLTRB(left, top, right, bottom);
  }

  Offset _scalePoint({
    required Point<int> point,
    required Size imageSize,
    required Size widgetSize,
  }) {
    final scaleX = widgetSize.width / imageSize.width;
    final scaleY = widgetSize.height / imageSize.height;

    double x = point.x.toDouble() * scaleX;
    double y = point.y.toDouble() * scaleY;

    if (isFrontCamera) {
      x = widgetSize.width - x;
    }

    return Offset(x, y);
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return oldDelegate.faces != faces ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.isFrontCamera != isFrontCamera;
  }
}
