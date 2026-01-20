import 'dart:developer' show log;
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:face_detection/utils/image_converter_isolate.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'services/face_recognition_service.dart' as recognition;
import 'verified_screen.dart';

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

class _FaceDetectionScreenState extends State<FaceDetectionScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  CameraController? _cameraController;
  bool _isDetecting = false;
  int _cameraIndex = 0;
  DateTime? _lastRecognitionTime;
  double? _currentConfidence;

  final recognition.FaceRecognitionService _recognitionService =
      recognition.FaceRecognitionService();
  final ImagePicker _imagePicker = ImagePicker();
  bool _isRecognitionReady = false;

  // Verification state
  DateTime? _firstConsistentMatchTime;
  String? _consistentlyMatchedName;
  bool _isVerificationComplete = false;

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
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    );
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

      // Stop processing if already verified
      if (_isVerificationComplete) {
        _isDetecting = false;
        return;
      }

      final faces = await _faceDetector.processImage(inputImage);
      final List<DetectedFaceInfo> faceInfos = [];

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
        // Convert raw camera frame to upright image FIRST
        final uprightImage = await _convertCameraImageToUpright(cameraImage);
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

            if (result != null) {
              if (mounted) {
                setState(() {
                  _currentConfidence = result.confidence;
                });
              }

              if (result.isMatch) {
                if (result.confidence >= 0.60) {
                  if (_consistentlyMatchedName == result.name) {
                    // Same person continuing to match
                    if (_firstConsistentMatchTime == null) {
                      _firstConsistentMatchTime = DateTime.now();
                    } else {
                      final duration = DateTime.now().difference(
                        _firstConsistentMatchTime!,
                      );
                      log(
                        'Consistent match for ${result.name}: ${duration.inMilliseconds}ms',
                      );

                      if (duration.inSeconds >= 2 && !_isVerificationComplete) {
                        _isVerificationComplete = true;
                        log(
                          'Verification successful! Moving to verified page.',
                        );

                        if (mounted) {
                          // Stop camera before navigating (optional but good practice)
                          await _cameraController?.stopImageStream();

                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => const VerifiedScreen(),
                            ),
                          ).then((_) {
                            // Reset state when coming back
                            _isVerificationComplete = false;
                            _firstConsistentMatchTime = null;
                            _consistentlyMatchedName = null;
                            _animationController.reset();
                            if (mounted) {
                              setState(() {
                                _currentConfidence = null;
                              });
                            }
                            _resumeCamera();
                          });
                        }
                      }
                    }
                  } else {
                    // New person or first match
                    _consistentlyMatchedName = result.name;
                    _firstConsistentMatchTime = DateTime.now();
                    _animationController.forward(from: 0);
                  }
                } else {
                  // Low confidence match (shouldn't happen if isMatch is true but safe to keep)
                  if (_firstConsistentMatchTime != null) {
                    log('Low confidence match. Resetting timer.');
                  }
                  _firstConsistentMatchTime = null;
                  _consistentlyMatchedName = null;
                  _animationController.reset();
                }
              } else {
                // Not a match (confidence < threshold)
                if (_firstConsistentMatchTime != null) {
                  log('Match lost or low confidence. Resetting timer.');
                }
                _firstConsistentMatchTime = null;
                _consistentlyMatchedName = null;
                _animationController.reset();
                // Do NOT reset _currentConfidence here so we can show red border
              }
            } else {
              // Result is null
              _firstConsistentMatchTime = null;
              _consistentlyMatchedName = null;
              _animationController.reset();
              if (mounted) {
                setState(() {
                  _currentConfidence = null;
                });
              }
            }

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
            // Reset if basic crop fails
            _firstConsistentMatchTime = null;
            _consistentlyMatchedName = null;
            _animationController.reset();
            if (mounted) {
              setState(() {
                _currentConfidence = null;
              });
            }
          }
        }
      } else {
        for (final face in faces) {
          faceInfos.add(DetectedFaceInfo(face: face));
        }
      }

      if (faces.isEmpty && mounted) {
        setState(() {
          _currentConfidence = null;
        });
      }

      if (mounted) {
        // setState(() {
        //   _detectedFaces = faceInfos;
        //   _debugLiveFaceBytes = currentLiveFaceBytes;
        // });
      }
    } catch (e) {
      log('Error detecting faces: $e');
    }

    _isDetecting = false;
  }

  Future<img.Image?> _convertCameraImageToUpright(
    CameraImage cameraImage,
  ) async {
    try {
      final camera = cameras[_cameraIndex];
      final sensorOrientation = camera.sensorOrientation;

      // Extract plane data to send to isolate
      final planes = cameraImage.planes.map((p) {
        return CameraPlaneMessage(
          bytes: p.bytes,
          bytesPerRow: p.bytesPerRow,
          bytesPerPixel: p.bytesPerPixel,
        );
      }).toList();

      final message = CameraImageMessage(
        planes: planes,
        width: cameraImage.width,
        height: cameraImage.height,
        sensorOrientation: sensorOrientation,
        isAndroid: Platform.isAndroid,
        isIOS: Platform.isIOS,
      );

      return await compute(convertCameraImageToUpright, message);
    } catch (e) {
      log('Error converting camera image: $e');
      return null;
    }
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

  Future<void> _resumeCamera() async {
    if (_cameraController == null) return;

    try {
      await _cameraController!.startImageStream(_processCameraImage);
    } catch (e) {
      log('Error resuming camera: $e');
    }
  }

  Future<void> _registerFaceFromGallery() async {
    // Pause camera stream during registration
    await _cameraController?.stopImageStream();

    try {
      final XFile? pickedFile = await _imagePicker.pickImage(
        source: ImageSource.gallery,
      );

      if (pickedFile == null) {
        await _resumeCamera();
        return;
      }

      final bytes = await pickedFile.readAsBytes();
      final image = img.decodeImage(bytes);
      if (image == null) {
        _showSnackBar('Could not decode image');
        await _resumeCamera();
        return;
      }

      // Detect face in the selected image
      final inputImage = InputImage.fromFilePath(pickedFile.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        _showSnackBar('No face detected in the image');
        await _resumeCamera();
        return;
      }

      if (faces.length > 1) {
        _showSnackBar(
          'Multiple faces detected. Please select an image with one face.',
        );
        await _resumeCamera();
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
        await _resumeCamera();
        return;
      }

      // Show dialog to enter name
      final name = await _showNameInputDialog();
      if (name == null || name.isEmpty) {
        await _resumeCamera();
        return;
      }

      // Limit to 1 face: Clear existing before registering new
      _recognitionService.clearAllFaces();

      await _recognitionService.registerFace(name, croppedFace);
      _showSnackBar('Face registered for $name');
      setState(() {});
    } catch (e) {
      log('Error registering face: $e');
      _showSnackBar('Error registering face');
    } finally {
      await _resumeCamera();
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

  @override
  void dispose() {
    _animationController.dispose();
    _cameraController?.dispose();
    _faceDetector.close();
    _recognitionService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Scaffold(
        backgroundColor: Colors.white,
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final size = MediaQuery.of(context).size;
    final double circleSize = size.width * 0.75;

    String greeting = 'Good Evening';
    final hour = DateTime.now().hour;
    if (hour < 12) {
      greeting = 'Good Morning';
    } else if (hour < 18) {
      greeting = 'Good Afternoon';
    }

    return Scaffold(
      backgroundColor: Colors.white,
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _isRecognitionReady ? _registerFaceFromGallery : null,
        icon: const Icon(Icons.add_a_photo),
        label: const Text('Register'),
      ),
      body: SafeArea(
        child: SizedBox(
          width: double.infinity,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Text(
              //   '$greeting, User',
              //   style: const TextStyle(
              //     fontSize: 24,
              //     fontWeight: FontWeight.bold,
              //     color: Colors.black,
              //   ),
              // ),
              // const SizedBox(height: 50),
              Stack(
                alignment: Alignment.center,
                children: [
                  // Circular Camera Preview
                  Container(
                    width: circleSize,
                    height: circleSize,
                    decoration: const BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.black,
                    ),
                    child: ClipOval(
                      child: FittedBox(
                        fit: BoxFit.cover,
                        child: SizedBox(
                          width: _cameraController!.value.previewSize!.height,
                          height: _cameraController!.value.previewSize!.width,
                          child: CameraPreviewWidget(
                            controller: _cameraController!,
                          ),
                        ),
                      ),
                    ),
                  ),
                  // Border
                  SizedBox(
                    width: circleSize + 25,
                    height: circleSize + 25,
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        // Static red border for low confidence
                        if (_currentConfidence != null &&
                            _currentConfidence! < 0.60)
                          DecoratedBox(
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(color: Colors.red, width: 4),
                            ),
                            child: const SizedBox.expand(),
                          ),

                        // Animated green border for high confidence
                        if (_currentConfidence != null &&
                            _currentConfidence! >= 0.60)
                          AnimatedBuilder(
                            animation: _animationController,
                            builder: (context, child) {
                              return CircularProgressIndicator(
                                value: _animationController.value,
                                strokeWidth: 4,
                                backgroundColor: Colors.grey.withOpacity(0.3),
                                valueColor: const AlwaysStoppedAnimation<Color>(
                                  Color(0xFF00E676),
                                ),
                              );
                            },
                          ),

                        // Default grey border if no confidence yet?
                        // Optional: keep the grey background of the indicator or add a static grey border if null
                        if (_currentConfidence == null)
                          DecoratedBox(
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(
                                color: Colors.grey.withOpacity(0.3),
                                width: 4,
                              ),
                            ),
                            child: const SizedBox.expand(),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 50),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 40),
                child: Text(
                  _recognitionService.registeredFaces.isEmpty
                      ? 'Please register a face'
                      : (_currentConfidence != null &&
                            _currentConfidence! >= 0.60)
                      ? 'Please hold your position'
                      : _currentConfidence != null && _currentConfidence! < 0.60
                      ? 'Can not find similarity with registered face\nPlease try again'
                      : 'Position your face inside the circle\nand wait for verification',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.grey,
                    height: 1.5,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
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
