import 'dart:async';
import 'dart:developer' show log;
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
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
      minFaceSize: 0.2,
    ),
  );

  // Capture state
  CaptureAngle _currentAngle = CaptureAngle.front;
  final Map<CaptureAngle, img.Image> _capturedFaces = {};

  // Stability tracking for auto-capture
  bool _isInCorrectPosition = false;
  DateTime? _positionStableStartTime;
  static const Duration _stabilityDuration = Duration(milliseconds: 1000);

  // Face detection state
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
    _cameraController?.stopImageStream();
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
            _currentYaw = null;
            _currentPitch = null;
          });
        }
        _isProcessing = false;
        return;
      }

      // Get the primary (largest) face
      final face = faces.reduce(
        (a, b) => (a.boundingBox.width * a.boundingBox.height) >
                (b.boundingBox.width * b.boundingBox.height)
            ? a
            : b,
      );

      final yaw = face.headEulerAngleY; // Left/right rotation
      final pitch = face.headEulerAngleX; // Up/down rotation

      if (mounted) {
        setState(() {
          _faceBoundingBox = face.boundingBox;
          _currentYaw = yaw;
          _currentPitch = pitch;
        });
      }

      // Check if face is in correct position for current angle
      final isCorrect = _isAngleCorrect(yaw, pitch, _currentAngle);

      if (isCorrect) {
        if (!_isInCorrectPosition) {
          _positionStableStartTime = DateTime.now();
        }

        if (mounted) {
          setState(() {
            _isInCorrectPosition = true;
          });
        }

        // Check if stable for enough time
        if (_positionStableStartTime != null) {
          final stableDuration =
              DateTime.now().difference(_positionStableStartTime!);
          if (stableDuration >= _stabilityDuration) {
            // Auto-capture!
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

  bool _isAngleCorrect(double? yaw, double? pitch, CaptureAngle targetAngle) {
    if (yaw == null || pitch == null) return false;

    switch (targetAngle) {
      case CaptureAngle.front:
        return yaw.abs() <= 10 && pitch.abs() <= 10;
      case CaptureAngle.left:
        return yaw >= 30 && yaw <= 55 && pitch.abs() <= 15;
      case CaptureAngle.right:
        return yaw <= -30 && yaw >= -55 && pitch.abs() <= 15;
      case CaptureAngle.up:
        return yaw.abs() <= 15 && pitch >= 15 && pitch <= 35;
      case CaptureAngle.down:
        return yaw.abs() <= 15 && pitch <= -15 && pitch >= -35;
    }
  }

  Future<void> _captureCurrentAngle(CameraImage cameraImage) async {
    if (_capturedFaces.containsKey(_currentAngle)) return;

    try {
      // Convert camera image to upright image
      final uprightImage = await _convertCameraImageToUpright(cameraImage);
      if (uprightImage == null) return;

      // Get the face bounding box
      if (_faceBoundingBox == null) return;

      // Crop the face
      final croppedFace = _recognitionService.cropFace(
        uprightImage,
        recognition.Rect(
          left: _faceBoundingBox!.left,
          top: _faceBoundingBox!.top,
          right: _faceBoundingBox!.right,
          bottom: _faceBoundingBox!.bottom,
        ),
      );

      if (croppedFace == null) return;

      // Store the captured face
      _capturedFaces[_currentAngle] = croppedFace;

      log('Captured ${_currentAngle.name} angle');

      // Move to next angle or finish
      if (mounted) {
        setState(() {
          _isInCorrectPosition = false;
          _positionStableStartTime = null;
        });
      }

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
      // All angles captured - prompt for name
      _finishRegistration();
    }
  }

  Future<void> _finishRegistration() async {
    // Stop camera processing while saving
    setState(() {
      _isSaving = true;
    });

    // Stop camera stream
    await _cameraController?.stopImageStream();

    // Prompt for name
    if (!mounted) return;

    final name = await _showNameInputDialog();
    if (name == null || name.isEmpty) {
      // User cancelled - restart camera
      await _cameraController?.startImageStream(_processCameraImage);
      if (mounted) {
        setState(() {
          _isSaving = false;
        });
      }
      return;
    }

    // Register face with all captured angles
    final faceImages = _capturedFaces.values.toList();
    final primaryFace = _capturedFaces[CaptureAngle.front] ?? faceImages.first;

    await _recognitionService.registerFaceMultiAngle(
      name,
      faceImages,
      primaryFace,
    );

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
              'Face registered successfully with ${faceImages.length} angles!'),
        ),
      );
      Navigator.of(context).pop(true); // Return success
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

    if (image.planes.isEmpty) return null;

    // Handle NV21 with multiple planes
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
        return 'Look straight at the camera';
      case CaptureAngle.left:
        return 'Turn your head LEFT';
      case CaptureAngle.right:
        return 'Turn your head RIGHT';
      case CaptureAngle.up:
        return 'Tilt your head UP';
      case CaptureAngle.down:
        return 'Tilt your head DOWN';
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
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.white,
        title: const Text('Register Face'),
        leading: IconButton(
          icon: const Icon(Icons.close),
          onPressed: () => Navigator.of(context).pop(false),
        ),
      ),
      body: _cameraController == null || !_cameraController!.value.isInitialized
          ? const Center(child: CircularProgressIndicator())
          : _isSaving
              ? const Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      CircularProgressIndicator(color: Colors.white),
                      SizedBox(height: 16),
                      Text(
                        'Saving...',
                        style: TextStyle(color: Colors.white, fontSize: 18),
                      ),
                    ],
                  ),
                )
              : Stack(
                  fit: StackFit.expand,
                  children: [
                    // Camera preview
                    Center(
                      child: CameraPreview(_cameraController!),
                    ),

                    // Face guide overlay
                    CustomPaint(
                      painter: FaceGuidePainter(
                        isInPosition: _isInCorrectPosition,
                        progress: _positionStableStartTime != null
                            ? DateTime.now()
                                    .difference(_positionStableStartTime!)
                                    .inMilliseconds /
                                _stabilityDuration.inMilliseconds
                            : 0.0,
                      ),
                    ),

                    // Progress indicator at top
                    Positioned(
                      top: 16,
                      left: 0,
                      right: 0,
                      child: _buildProgressIndicator(),
                    ),

                    // Instructions at bottom
                    Positioned(
                      bottom: 80,
                      left: 0,
                      right: 0,
                      child: _buildInstructions(),
                    ),

                    // Debug info (optional)
                    if (kDebugMode)
                      Positioned(
                        bottom: 160,
                        left: 16,
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Text(
                            'Yaw: ${_currentYaw?.toStringAsFixed(1) ?? '-'}°\n'
                            'Pitch: ${_currentPitch?.toStringAsFixed(1) ?? '-'}°',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
    );
  }

  Widget _buildProgressIndicator() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: CaptureAngle.values.map((angle) {
        final isCompleted = _capturedFaces.containsKey(angle);
        final isCurrent = angle == _currentAngle;

        return Container(
          width: 40,
          height: 40,
          margin: const EdgeInsets.symmetric(horizontal: 8),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: isCompleted
                ? Colors.green
                : isCurrent
                    ? Colors.deepPurple
                    : Colors.grey.shade700,
            border: isCurrent
                ? Border.all(color: Colors.white, width: 2)
                : null,
          ),
          child: Center(
            child: isCompleted
                ? const Icon(Icons.check, color: Colors.white, size: 20)
                : Text(
                    _getAngleLabel(angle),
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 10,
                      fontWeight:
                          isCurrent ? FontWeight.bold : FontWeight.normal,
                    ),
                  ),
          ),
        );
      }).toList(),
    );
  }

  String _getAngleLabel(CaptureAngle angle) {
    switch (angle) {
      case CaptureAngle.front:
        return 'F';
      case CaptureAngle.left:
        return 'L';
      case CaptureAngle.right:
        return 'R';
      case CaptureAngle.up:
        return 'U';
      case CaptureAngle.down:
        return 'D';
    }
  }

  Widget _buildInstructions() {
    return Column(
      children: [
        Icon(
          _getDirectionIcon(),
          color: _isInCorrectPosition ? Colors.green : Colors.white,
          size: 48,
        ),
        const SizedBox(height: 12),
        Text(
          _getInstructionText(),
          style: TextStyle(
            color: _isInCorrectPosition ? Colors.green : Colors.white,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        if (_isInCorrectPosition) ...[
          const SizedBox(height: 8),
          const Text(
            'Hold still...',
            style: TextStyle(
              color: Colors.green,
              fontSize: 16,
            ),
          ),
        ],
      ],
    );
  }
}

class FaceGuidePainter extends CustomPainter {
  final bool isInPosition;
  final double progress;

  FaceGuidePainter({
    required this.isInPosition,
    required this.progress,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2 - 40);
    final ovalWidth = size.width * 0.6;
    final ovalHeight = ovalWidth * 1.3;

    final rect = Rect.fromCenter(
      center: center,
      width: ovalWidth,
      height: ovalHeight,
    );

    // Draw oval guide
    final guidePaint = Paint()
      ..color = isInPosition ? Colors.green : Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    canvas.drawOval(rect, guidePaint);

    // Draw progress arc when in position
    if (isInPosition && progress > 0) {
      final progressPaint = Paint()
        ..color = Colors.green
        ..style = PaintingStyle.stroke
        ..strokeWidth = 6
        ..strokeCap = StrokeCap.round;

      canvas.drawArc(
        rect.inflate(8),
        -3.14159 / 2, // Start from top
        2 * 3.14159 * progress.clamp(0.0, 1.0),
        false,
        progressPaint,
      );
    }
  }

  @override
  bool shouldRepaint(FaceGuidePainter oldDelegate) {
    return oldDelegate.isInPosition != isInPosition ||
        oldDelegate.progress != progress;
  }
}
