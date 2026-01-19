import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
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
  List<DetectedFaceInfo> _detectedFaces = [];
  Uint8List? _debugLiveFaceBytes;
  int _cameraIndex = 0;

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
      debugPrint('Error initializing recognition service: $e');
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
      debugPrint('Error initializing camera: $e');
    }
  }

  Future<void> _processCameraImage(CameraImage cameraImage) async {
    if (_isDetecting) return;
    _isDetecting = true;

    try {
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
        final fullImage = _convertCameraImageToImg(cameraImage);
        if (fullImage != null) {
          final camera = cameras[_cameraIndex];
          final sensorOrientation = camera.sensorOrientation;

          for (final face in faces) {
            // Transform bounding box based on rotation instead of rotating image
            // ML Kit returns coordinates in upright space, we need to map to raw image space
            final bbox = face.boundingBox;
            double left, top, right, bottom;
            final double imgW = fullImage.width.toDouble();
            final double imgH = fullImage.height.toDouble();

            // Transform coordinates based on sensor orientation
            // sensorOrientation tells us how the raw image is rotated relative to upright
            switch (sensorOrientation) {
              case 90:
                // Raw image is rotated 90° CW from upright
                // ML Kit coords (x,y) in upright -> (y, imgW-x) in raw
                left = bbox.top;
                top = imgW - bbox.right;
                right = bbox.bottom;
                bottom = imgW - bbox.left;
                break;
              case 180:
                left = imgW - bbox.right;
                top = imgH - bbox.bottom;
                right = imgW - bbox.left;
                bottom = imgH - bbox.top;
                break;
              case 270:
                // Raw image is rotated 270° CW (or 90° CCW) from upright
                left = imgH - bbox.bottom;
                top = bbox.left;
                right = imgH - bbox.top;
                bottom = bbox.right;
                break;
              default: // 0
                left = bbox.left;
                top = bbox.top;
                right = bbox.right;
                bottom = bbox.bottom;
            }

            final croppedFace = _recognitionService.cropFace(
              fullImage,
              recognition.Rect(
                left: left,
                top: top,
                right: right,
                bottom: bottom,
              ),
            );

            if (croppedFace != null) {
              // Rotate cropped face to upright orientation for recognition
              img.Image uprightFace = croppedFace;
              if (sensorOrientation != 0) {
                // Rotate the cropped face to make it upright
                uprightFace = img.copyRotate(croppedFace, angle: sensorOrientation.toDouble());
              }

              // Flip horizontally for front camera to match gallery selfie orientation
              if (camera.lensDirection == CameraLensDirection.front) {
                uprightFace = img.flipHorizontal(uprightFace);
              }

              // Encode and decode to normalize image format (match gallery image processing)
              final pngBytes = img.encodePng(uprightFace);
              final normalizedFace = img.decodeImage(pngBytes);

              // Store debug image for the first face
              if (currentLiveFaceBytes == null) {
                currentLiveFaceBytes = pngBytes;
              }

              if (normalizedFace == null) {
                debugPrint('Failed to normalize face image');
                faceInfos.add(DetectedFaceInfo(face: face));
                continue;
              }

              debugPrint('Sending face for recognition: ${normalizedFace.width}x${normalizedFace.height}');
              final result = await _recognitionService.recognizeFace(
                normalizedFace,
              );
              debugPrint('Recognition result: name=${result?.name}, confidence=${result?.confidence}, isMatch=${result?.isMatch}');
              if (result == null) {
                debugPrint('Recognition returned null - check if embedding generation failed');
              }
              faceInfos.add(
                DetectedFaceInfo(
                  face: face,
                  recognizedName: result?.isMatch == true ? result?.name : null,
                  confidence: result?.isMatch == true
                      ? result?.confidence
                      : null,
                  bestMatchName: result?.name,
                  bestMatchScore: result?.confidence,
                ),
              );
            } else {
              debugPrint('Cropped face is null');
              faceInfos.add(DetectedFaceInfo(face: face));
            }
          }
        } else {
          for (final face in faces) {
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
      debugPrint('Error detecting faces: $e');
    }

    _isDetecting = false;
  }

  img.Image? _convertCameraImageToImg(CameraImage cameraImage) {
    try {
      if (Platform.isAndroid) {
        return _convertYUV420ToImage(cameraImage);
      } else if (Platform.isIOS) {
        return _convertBGRA8888ToImage(cameraImage);
      }
    } catch (e) {
      debugPrint('Error converting camera image: $e');
    }
    return null;
  }

  img.Image? _convertYUV420ToImage(CameraImage cameraImage) {
    final width = cameraImage.width;
    final height = cameraImage.height;
    final image = img.Image(width: width, height: height);

    try {
      final int uvRowStride = cameraImage.planes[1].bytesPerRow;
      final int? uvPixelStride = cameraImage.planes[1].bytesPerPixel;

      final yPlane = cameraImage.planes[0].bytes;
      final uPlane = cameraImage.planes[1].bytes;
      final vPlane = cameraImage.planes[2].bytes;

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * cameraImage.planes[0].bytesPerRow + x;
          // Basic YUV420 assumption: UV subsampled 2x2.
          // For row stride and pixel stride usage:
          final int uvIndex =
              (y ~/ 2) * uvRowStride + (x ~/ 2) * (uvPixelStride ?? 1);

          final yValue = yPlane[yIndex];
          final uValue = uPlane[uvIndex];
          final vValue = vPlane[uvIndex];

          // YUV to RGB conversion
          final r = (yValue + 1.370705 * (vValue - 128)).clamp(0, 255).toInt();
          final g =
              (yValue - 0.337633 * (uValue - 128) - 0.698001 * (vValue - 128))
                  .clamp(0, 255)
                  .toInt();
          final b = (yValue + 1.732446 * (uValue - 128)).clamp(0, 255).toInt();

          image.setPixelRgb(x, y, r, g, b);
        }
      }
    } catch (e) {
      debugPrint('Error converting YUV420: $e');
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
      // Concatenate all planes
      final WriteBuffer allBytes = WriteBuffer();
      for (final Plane plane in image.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      final bytes = allBytes.done().buffer.asUint8List();

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
      debugPrint('Error registering face: $e');
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
      debugPrint('Error switching camera: $e');
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
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _isRecognitionReady ? _registerFaceFromGallery : null,
        icon: const Icon(Icons.add_a_photo),
        label: const Text('Register Face'),
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
        // Display Registered Face Thumbnail for Debugging
        if (_recognitionService.registeredFaces.isNotEmpty &&
            _recognitionService.registeredFaces.first.faceBytes != null)
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
                      const Text('Registered', style: TextStyle(color: Colors.white, fontSize: 10)),
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
                        style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)
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
                        const Text('Live Input', style: TextStyle(color: Colors.white, fontSize: 10)),
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
                          style: TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)
                        ),
                      ],
                    ),
                  ),
                ]
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
        Rect.fromLTWH(textX - 5, textY - 5, textPainter.width + 10, textPainter.height + 10),
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
