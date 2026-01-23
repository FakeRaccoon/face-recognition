import 'dart:async';
import 'dart:developer' show log;
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import 'package:face_detection/main.dart' show cameras;
import 'package:face_detection/services/face_recognition_service.dart'
    as recognition;
import 'package:face_detection/utils/image_converter_isolate.dart';

enum CaptureAngle { front, left, right, up, down }

class MultiAngleRegistrationScreen extends StatefulWidget {
  const MultiAngleRegistrationScreen({super.key});

  @override
  State<MultiAngleRegistrationScreen> createState() =>
      _MultiAngleRegistrationScreenState();
}

class _MultiAngleRegistrationScreenState
    extends State<MultiAngleRegistrationScreen> {
  CameraController? _cameraController;
  int _cameraIndex = 0;
  bool _isProcessing = false;
  bool _isSaving = false;

  final recognition.FaceRecognitionService _recognitionService =
      recognition.FaceRecognitionService();

  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableLandmarks: true,
      enableClassification: false,
      enableTracking: true,
      performanceMode: FaceDetectorMode.fast,
      minFaceSize: 0.1, // Smaller face size to allow more rotation
    ),
  );

  // Capture state
  CaptureAngle _currentAngle = CaptureAngle.front;
  final Map<CaptureAngle, img.Image> _capturedFaces = {};

  // Stability tracking
  bool _isInCorrectPosition = false;
  DateTime? _positionStableStartTime;
  static const Duration _stabilityDuration = Duration(milliseconds: 800);

  // Debug/Feedback state
  Rect? _faceBoundingBox;
  double? _currentYaw;
  double? _currentPitch;

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  Future<void> _initializeServices() async {
    if (!_recognitionService.isInitialized) {
      await _recognitionService.initialize();
    }
    await _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) return;

    _cameraIndex = cameras.indexWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
    );
    if (_cameraIndex == -1) _cameraIndex = 0;

    _cameraController = CameraController(
      cameras[_cameraIndex],
      ResolutionPreset.high,
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

  @override
  void dispose() {
    try {
      // Stop stream if it might be running, ignore error if not
      _cameraController?.stopImageStream();
    } catch (_) {}
    _cameraController?.dispose();
    _faceDetector.close();
    super.dispose();
  }

  Future<void> _processCameraImage(CameraImage cameraImage) async {
    if (_isProcessing || _isSaving) return;
    _isProcessing = true;

    try {
      final inputImage = await _convertCameraImage(cameraImage);
      if (inputImage == null) {
        _isProcessing = false;
        return;
      }

      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        if (mounted) {
          setState(() {
            _faceBoundingBox = null;
            _isInCorrectPosition = false;
            _positionStableStartTime = null;
          });
        }
        _isProcessing = false;
        return;
      }

      final face = faces.reduce(
        (a, b) =>
            (a.boundingBox.width * a.boundingBox.height) >
                (b.boundingBox.width * b.boundingBox.height)
            ? a
            : b,
      );

      final yaw = face.headEulerAngleY ?? 0;
      final pitch = face.headEulerAngleX ?? 0;

      if (mounted) {
        setState(() {
          _faceBoundingBox = face.boundingBox;
          _currentYaw = yaw;
          _currentPitch = pitch;
        });
      }

      final isCorrect = _isAngleCorrect(yaw, pitch, _currentAngle);

      if (isCorrect) {
        if (!_isInCorrectPosition) {
          _positionStableStartTime = DateTime.now();
          HapticFeedback.selectionClick();
        }

        if (mounted) {
          setState(() {
            _isInCorrectPosition = true;
          });
        }

        if (_positionStableStartTime != null) {
          final stableDuration = DateTime.now().difference(
            _positionStableStartTime!,
          );
          if (stableDuration >= _stabilityDuration) {
            await _captureCurrentAngle(cameraImage);
          }
        }
      } else {
        if (mounted) {
          setState(() {
            _isInCorrectPosition = false;
            _positionStableStartTime = null;
          });
        }
      }
    } catch (e) {
      log('Error processing camera image: $e');
    }

    _isProcessing = false;
  }

  bool _isAngleCorrect(double yaw, double pitch, CaptureAngle targetAngle) {
    const double mainAngleThreshold =
        15.0; // Reduced from 20 to make it easier but distinct
    const double centerThreshold =
        10.0; // Tightened from 15 for better frontal alignment

    switch (targetAngle) {
      case CaptureAngle.front:
        return yaw.abs() <= centerThreshold && pitch.abs() <= centerThreshold;

      case CaptureAngle.left:
        return yaw > mainAngleThreshold && pitch.abs() <= centerThreshold;

      case CaptureAngle.right:
        return yaw < -mainAngleThreshold && pitch.abs() <= centerThreshold;

      case CaptureAngle.up:
        return pitch > mainAngleThreshold && yaw.abs() <= centerThreshold;

      case CaptureAngle.down:
        return pitch < -mainAngleThreshold && yaw.abs() <= centerThreshold;
    }
  }

  Future<void> _captureCurrentAngle(CameraImage cameraImage) async {
    if (_capturedFaces.containsKey(_currentAngle)) return;

    try {
      HapticFeedback.mediumImpact();

      final uprightImage = await _convertCameraImageToUpright(cameraImage);
      if (uprightImage == null) return;

      if (_faceBoundingBox == null) return;

      final croppedFace = await _recognitionService.cropFace(
        uprightImage,
        recognition.Rect(
          left: _faceBoundingBox!.left,
          top: _faceBoundingBox!.top,
          right: _faceBoundingBox!.right,
          bottom: _faceBoundingBox!.bottom,
        ),
      );

      if (croppedFace == null) return;

      _capturedFaces[_currentAngle] = croppedFace;
      // log('Captured ${_currentAngle.name} angle');

      if (mounted) {
        setState(() {
          _isInCorrectPosition = false;
          _positionStableStartTime = null;
        });
      }

      await Future.delayed(const Duration(milliseconds: 200)); // Pause slightly
      _advanceToNextAngle();
    } catch (e) {
      log('Error capturing face: $e');
    }
  }

  void _advanceToNextAngle() {
    final angles = CaptureAngle.values;
    final currentIndex = angles.indexOf(_currentAngle);

    if (currentIndex < angles.length - 1) {
      if (mounted) {
        setState(() {
          _currentAngle = angles[currentIndex + 1];
        });
      }
    } else {
      _finishRegistration();
    }
  }

  Future<void> _finishRegistration() async {
    setState(() {
      _isSaving = true;
    });

    await _cameraController?.stopImageStream();

    if (!mounted) return;

    final name = await _showNameInputDialog();
    if (name == null || name.isEmpty) {
      // Logic to restart or exit?
      // Let's restart for now or just allow re-entry
      if (mounted) Navigator.pop(context);
      return;
    }

    final allFaces = _capturedFaces.values.toList();
    final primaryFace = _capturedFaces[CaptureAngle.front] ?? allFaces.first;

    await _recognitionService.registerFaceMultiAngle(
      name,
      allFaces,
      primaryFace,
    );

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Face registered successfully with ${allFaces.length} angles!',
          ),
        ),
      );
      Navigator.of(context).pop(true);
    }
  }

  Future<String?> _showNameInputDialog() async {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: const Text('Enter Name'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(
            labelText: 'Name',
            hintText: 'e.g., John Doe',
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

  // --- Helper Methods ---

  Future<InputImage?> _convertCameraImage(CameraImage image) async {
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
    if (format == null) return null;

    if (Platform.isAndroid &&
        format == InputImageFormat.nv21 &&
        image.planes.length > 1) {
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

  Future<img.Image?> _convertCameraImageToUpright(
    CameraImage cameraImage,
  ) async {
    try {
      final camera = cameras[_cameraIndex];
      final sensorOrientation = camera.sensorOrientation;

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

  String _getInstructionText() {
    switch (_currentAngle) {
      case CaptureAngle.front:
        return 'Look Straight';
      case CaptureAngle.left:
        return 'Turn Left';
      case CaptureAngle.right:
        return 'Turn Right';
      case CaptureAngle.up:
        return 'Look Up';
      case CaptureAngle.down:
        return 'Look Down';
    }
  }

  IconData _getDirectionIcon() {
    switch (_currentAngle) {
      case CaptureAngle.front:
        return Icons.face;
      case CaptureAngle.left:
        return Icons.arrow_back;
      case CaptureAngle.right:
        return Icons.arrow_forward;
      case CaptureAngle.up:
        return Icons.arrow_upward;
      case CaptureAngle.down:
        return Icons.arrow_downward;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Register Face Steps'),
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.white,
      ),
      body: _cameraController == null || !_cameraController!.value.isInitialized
          ? const Center(child: CircularProgressIndicator())
          : Stack(
              fit: StackFit.expand,
              children: [
                Center(child: CameraPreview(_cameraController!)),

                // Overlay Filter to focus attention
                ColorFiltered(
                  colorFilter: ColorFilter.mode(
                    Colors.black.withOpacity(0.3),
                    BlendMode.darken,
                  ),
                  child: Container(color: Colors.transparent),
                ),

                // Guide Box (only visual)
                Center(
                  child: Container(
                    width: 300,
                    height: 300,
                    decoration: BoxDecoration(
                      border: Border.all(
                        color: _isInCorrectPosition
                            ? Colors.green
                            : Colors.white,
                        width: 3,
                      ),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child:
                        _isInCorrectPosition && _positionStableStartTime != null
                        ? Center(
                            child: CircularProgressIndicator(
                              valueColor: const AlwaysStoppedAnimation(
                                Colors.green,
                              ),
                            ),
                          )
                        : null,
                  ),
                ),

                // Top Progress
                Positioned(
                  top: 20,
                  left: 0,
                  right: 0,
                  child: SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: CaptureAngle.values.map((angle) {
                        final isCompleted = _capturedFaces.containsKey(angle);
                        final isCurrent = angle == _currentAngle;
                        return Container(
                          margin: const EdgeInsets.symmetric(horizontal: 4),
                          width: 12,
                          height: 12,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: isCompleted
                                ? Colors.green
                                : (isCurrent ? Colors.white : Colors.white24),
                          ),
                        );
                      }).toList(),
                    ),
                  ),
                ),

                // Bottom Instructions
                Positioned(
                  bottom: 50,
                  left: 0,
                  right: 0,
                  child: Column(
                    children: [
                      Icon(_getDirectionIcon(), size: 60, color: Colors.white),
                      const SizedBox(height: 16),
                      Text(
                        _getInstructionText().toUpperCase(),
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 1.5,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        _isInCorrectPosition
                            ? "HOLD STILL..."
                            : "Align your face",
                        style: TextStyle(
                          color: _isInCorrectPosition
                              ? Colors.greenAccent
                              : Colors.white70,
                          fontSize: 16,
                        ),
                      ),

                      // Debug Text (Optional)
                      const SizedBox(height: 20),
                      Text(
                        'Yaw: ${_currentYaw?.toStringAsFixed(1) ?? 0}  Pitch: ${_currentPitch?.toStringAsFixed(1) ?? 0}',
                        style: const TextStyle(
                          color: Colors.white30,
                          fontSize: 10,
                        ),
                      ),
                    ],
                  ),
                ),

                if (_isSaving)
                  Container(
                    color: Colors.black87,
                    child: const Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          CircularProgressIndicator(),
                          SizedBox(height: 20),
                          Text(
                            "Saving Face Data...",
                            style: TextStyle(color: Colors.white),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
    );
  }
}
