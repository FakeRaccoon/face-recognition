import 'dart:developer' show log;
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:face_detection/utils/image_converter_isolate.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart'; // Add scheduler import

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import 'package:face_detection/services/face_recognition_service.dart'
    as recognition;

import 'registration_screen.dart';

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
      home: const RegistrationScreen(),
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
    with TickerProviderStateMixin {
  late AnimationController _animationController;
  CameraController? _cameraController;
  bool _isDetecting = false;
  int _cameraIndex = 0;
  DateTime? _lastRecognitionTime;
  double? _currentConfidence;

  // Painting state
  // Painting state
  Rect? _targetBoundingBox; // The latest detection result
  Rect? _currentBoundingBox; // The interpolated value for display
  Size? _imageSize;
  InputImageRotation? _imageRotation;
  late Ticker _ticker; // Ticker for smooth animation

  final recognition.FaceRecognitionService _recognitionService =
      recognition.FaceRecognitionService();

  bool _isRecognitionReady = false;

  // Verification state

  String? _consistentlyMatchedName;
  bool _isVerificationComplete = false;

  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: true,
      enableLandmarks: true,
      enableClassification: false,
      enableTracking: true,
      performanceMode: FaceDetectorMode.accurate,
    ),
  );

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    );

    // Initialize ticker for smooth interpolation
    _ticker = createTicker((elapsed) {
      if (_targetBoundingBox != null) {
        if (_currentBoundingBox == null) {
          _currentBoundingBox = _targetBoundingBox;
          return;
        }

        // Linear interpolation with 0.15 factor for smooth following
        // We handle nullable rect lerp manually to be safe
        final target = _targetBoundingBox!;
        final current = _currentBoundingBox!;

        // Check distance to avoid unnecessary repaints
        final dist = (target.center - current.center).distance;
        if (dist < 0.5 && (target.width - current.width).abs() < 0.5) {
          return;
        }

        // Adaptive lerp: fast for large movements (snappy), smooth for small adjustments (stable)
        final double lerpFactor = dist > 30.0 ? 0.7 : 0.2;
        final newRect = Rect.lerp(current, target, lerpFactor);
        if (newRect != null) {
          setState(() {
            _currentBoundingBox = newRect;
          });
        }
      } else {
        // If target is null (lost face), clear current
        if (_currentBoundingBox != null) {
          setState(() {
            _currentBoundingBox = null;
          });
        }
      }
    });
    _ticker.start();

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
      final inputImage = await _convertCameraImage(cameraImage);
      if (inputImage == null) {
        _isDetecting = false;
        return;
      }

      // Stop processing blocking removed to allow continuous tracking
      // if (_isVerificationComplete) { ... }

      final faces = await _faceDetector.processImage(inputImage);

      // Select the largest face if multiple are detected
      Face? primaryFace;
      if (faces.isNotEmpty) {
        primaryFace = faces.reduce(
          (a, b) =>
              (a.boundingBox.width * a.boundingBox.height) >
                  (b.boundingBox.width * b.boundingBox.height)
              ? a
              : b,
        );
      }

      final List<DetectedFaceInfo> faceInfos = [];

      // Update UI state for painting
      // NO setState here for bounding box anymore, let Ticker handle it
      // ONLY update 'target'
      if (mounted) {
        // We still need to update metadata
        _imageSize = inputImage.metadata?.size;
        _imageRotation = inputImage.metadata?.rotation;
        _targetBoundingBox = primaryFace?.boundingBox;
      }

      if (primaryFace != null) {
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

          // Convert raw camera frame to upright image FIRST
          final uprightImage = await _convertCameraImageToUpright(cameraImage);
          if (uprightImage == null) {
            _isDetecting = false;
            return;
          }

          final face = primaryFace;

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
            final result = await _recognitionService.recognizeFace(croppedFace);

            if (result != null) {
              if (mounted) {
                setState(() {
                  _currentConfidence = result.confidence;
                });
              }

              if (result.isMatch) {
                if (result.confidence >= 0.7) {
                  if (_consistentlyMatchedName == result.name) {
                    // Same person continuing to match
                    if (!_isVerificationComplete) {
                      _isVerificationComplete = true;

                      if (mounted) {
                        setState(() {
                          // Force UI update
                        });
                      }
                    }
                  } else {
                    // New person or first match
                    _consistentlyMatchedName = result.name;
                    _isVerificationComplete = true; // Immediate completion
                    if (mounted) {
                      setState(() {
                        // Force UI update
                      });
                    }
                  }
                } else {
                  _consistentlyMatchedName = null;
                  _isVerificationComplete = false; // Reset
                  _animationController.reset();
                }
              } else {
                _consistentlyMatchedName = null;
                _isVerificationComplete = false; // Reset
                _animationController.reset();
              }
            } else {
              _consistentlyMatchedName = null;
              _isVerificationComplete = false; // Reset
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
            faceInfos.add(DetectedFaceInfo(face: face));

            _consistentlyMatchedName = null;
            _isVerificationComplete = false; // Reset
            _animationController.reset();
            if (mounted) {
              setState(() {
                _currentConfidence = null;
              });
            }
          }
        } else {
          faceInfos.add(DetectedFaceInfo(face: primaryFace));
        }
      }

      if (faces.isEmpty && mounted) {
        // Just clear target, ticker will clear current
        _targetBoundingBox = null;

        setState(() {
          _currentConfidence = null;
          // _faceBoundingBox = null; // handled by target
          // Reset verification state when face is lost
          _isVerificationComplete = false;
          _consistentlyMatchedName = null;
        });

        _animationController.reset();
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
      // Run in background isolate to avoid blocking UI thread
      final nv21Bytes = await _convertYUV420ToNv21Background(image);

      if (nv21Bytes == null) return null;

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

  Future<Uint8List?> _convertYUV420ToNv21Background(CameraImage image) async {
    try {
      final planes = image.planes.map((p) {
        return CameraPlaneMessage(
          bytes: p.bytes,
          bytesPerRow: p.bytesPerRow,
          bytesPerPixel: p.bytesPerPixel,
        );
      }).toList();

      final message = CameraImageMessage(
        planes: planes,
        width: image.width,
        height: image.height,
        sensorOrientation: 0, // Not used for this conversion
        isAndroid: true,
        isIOS: false,
      );

      return await compute(convertYUV420ToNV21, message);
    } catch (e) {
      log('Error converting YUV420 to NV21 in background: $e');
      return null;
    }
  }

  @override
  void dispose() {
    _ticker.dispose();
    _animationController.dispose();
    _cameraController?.dispose();
    _faceDetector.close();

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

    // Status text logic replaced by visual indicators and bottom bar

    Color statusColor = Colors.white;
    if (_currentConfidence != null) {
      if (_currentConfidence! >= 0.7) {
        statusColor = const Color(0xFF00E676);
      } else {
        statusColor = Colors.redAccent;
      }
    }

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        leading: const BackButton(color: Colors.white),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Full screen camera preview
          FullScreenCameraPreview(
            controller: _cameraController!,
            child:
                _currentBoundingBox != null &&
                    _imageSize != null &&
                    _imageRotation != null
                ? CustomPaint(
                    painter: FacePainter(
                      boundingBox: _currentBoundingBox!,
                      imageSize: _imageSize!,
                      rotation: _imageRotation!,
                      cameraLensDirection:
                          _cameraController!.description.lensDirection,
                      color: statusColor,
                    ),
                  )
                : null,
          ),

          // Overlay Texts
          Positioned(
            bottom: 50,
            left: 0,
            right: 0,
            child: Column(
              children: [
                if (_isVerificationComplete && _consistentlyMatchedName != null)
                  Card(
                    margin: const EdgeInsets.symmetric(horizontal: 40),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    color: Colors.white,
                    elevation: 4,
                    child: Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16.0,
                        vertical: 12.0,
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 60,
                            height: 60,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(
                                color: Colors.grey.shade300,
                                width: 1,
                              ),
                              image: DecorationImage(
                                fit: BoxFit.cover,
                                image:
                                    (() {
                                          try {
                                            final face = _recognitionService
                                                .registeredFaces
                                                .firstWhere(
                                                  (f) =>
                                                      f.name ==
                                                      _consistentlyMatchedName,
                                                );
                                            if (face.faceBytes != null) {
                                              return MemoryImage(
                                                face.faceBytes!,
                                              );
                                            }
                                          } catch (e) {
                                            // Fallback
                                          }
                                          return const NetworkImage(
                                            'https://via.placeholder.com/150',
                                          );
                                        })()
                                        as ImageProvider,
                              ),
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Text(
                                  _consistentlyMatchedName!.toUpperCase(),
                                  style: const TextStyle(
                                    color: Colors.black87,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 22,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  '${DateTime.now().hour.toString().padLeft(2, '0')}:${DateTime.now().minute.toString().padLeft(2, '0')}:${DateTime.now().second.toString().padLeft(2, '0')}',
                                  style: TextStyle(
                                    color: Colors.grey.shade600,
                                    fontSize: 16,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                if (_consistentlyMatchedName != null &&
                    !_isVerificationComplete)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 20),
                    child: CircularProgressIndicator(
                      value: _animationController.value,
                      color: const Color(0xFF00E676),
                    ),
                  ),

                // Bottom Status Bar
                Container(
                  margin: const EdgeInsets.only(top: 20, left: 20, right: 20),
                  padding: const EdgeInsets.symmetric(
                    vertical: 16,
                    horizontal: 24,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.8),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    _recognitionService.registeredFaces.isEmpty
                        ? 'Please register a face'
                        : _isVerificationComplete &&
                              _consistentlyMatchedName != null
                        ? 'Verified, Welcome!'
                        : _currentConfidence != null
                        ? _currentConfidence! >= 0.7
                              ? 'Verifying...'
                              : 'Face Not Recognized'
                        : 'Face Not Detected',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w500,
                      letterSpacing: 0.5,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class FacePainter extends CustomPainter {
  final Rect boundingBox;
  final Size imageSize;
  final InputImageRotation rotation;
  final CameraLensDirection cameraLensDirection;
  final Color color;

  FacePainter({
    required this.boundingBox,
    required this.imageSize,
    required this.rotation,
    required this.cameraLensDirection,
    this.color = Colors.white,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = color;

    final rect = _scaleRect(
      rect: boundingBox,
      imageSize: imageSize,
      widgetSize: size,
      rotation: rotation,
      cameraLensDirection: cameraLensDirection,
    );

    // Draw Corner Brackets instead of full Rect
    final double cornerLength = rect.width * 0.2; // 20% of width
    final double strokeWidth = 5.0;

    paint
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round; // Update existing paint object

    final Path path = Path();

    // Top Left
    path.moveTo(rect.left, rect.top + cornerLength);
    path.lineTo(rect.left, rect.top + 10); // slightly rounded corner start
    path.quadraticBezierTo(rect.left, rect.top, rect.left + 10, rect.top);
    path.lineTo(rect.left + cornerLength, rect.top);

    // Top Right
    path.moveTo(rect.right - cornerLength, rect.top);
    path.lineTo(rect.right - 10, rect.top);
    path.quadraticBezierTo(rect.right, rect.top, rect.right, rect.top + 10);
    path.lineTo(rect.right, rect.top + cornerLength);

    // Bottom Right
    path.moveTo(rect.right, rect.bottom - cornerLength);
    path.lineTo(rect.right, rect.bottom - 10);
    path.quadraticBezierTo(
      rect.right,
      rect.bottom,
      rect.right - 10,
      rect.bottom,
    );
    path.lineTo(rect.right - cornerLength, rect.bottom);

    // Bottom Left
    path.moveTo(rect.left + cornerLength, rect.bottom);
    path.lineTo(rect.left + 10, rect.bottom);
    path.quadraticBezierTo(rect.left, rect.bottom, rect.left, rect.bottom - 10);
    path.lineTo(rect.left, rect.bottom - cornerLength);

    canvas.drawPath(path, paint);
  }

  Rect _scaleRect({
    required Rect rect,
    required Size imageSize,
    required Size widgetSize,
    required InputImageRotation rotation,
    required CameraLensDirection cameraLensDirection,
  }) {
    // 1. Convert to absolute coordinates in the image buffer
    // ML Kit returns coordinates relative to the "InputImage"
    // Since we create InputImage.fromBytes, the coordinates are relative to the raw buffer WxH

    // 2. Adjust for Rotation
    // On Android, the raw buffer is often landscape (e.g. 1920x1080), but rotation is 270/90.
    // The InputImage rotation metadata informs ML Kit.
    // However, the bounding box returned by ML Kit is typically axis aligned to the image as it "sits" in the buffer, OR un-rotated.
    // Documentation says it returns coordinates in the image coordinate system.

    // Let's assume the safe way: Normalize with respect to imageSize.
    // But we need to know if imageSize.width means the "visual width" or "buffer width".
    // InputImage.metadata.size is usually the buffer dimensions (1080x1920 or 1920x1080).

    // We need to swap width and height for rotation 90 or 270 on Android
    // The imageSize we receive here constitutes the "InputImage" metadata size.
    // However, ML Kit detections are relative to the *unrotated* buffer if we use `fromBytes`.
    // Wait, the documentation says "The bounding box is relative to the *image*...".
    // If we passed rotation metadata, ML Kit *internally* handles rotation for detection logic but returns coordinates in the logical image space?
    // Actually for `fromBytes` with rotation, ML Kit usually returns coordinates respecting that rotation?
    // Let's rely on standard practice:
    // 1. Android Buffer is landscape. Rotation 270 means phone is portrait.
    // 2. ML Kit bounding box is detected on the "rotated" image concept?
    // No, usually for `processImage(fromBytes)` ML Kit returns coords in the *buffer* coordinate system (Landscape), UNLESS we are very lucky.
    // Actually, widespread issue.

    // Let's implement the standard transformation:
    // Source: Google ML Kit Quickstart for Flutter

    final bool isRotated =
        rotation == InputImageRotation.rotation90deg ||
        rotation == InputImageRotation.rotation270deg;

    final double scaleX =
        widgetSize.width /
        (isRotated && Platform.isAndroid ? imageSize.height : imageSize.width);
    final double scaleY =
        widgetSize.height /
        (isRotated && Platform.isAndroid ? imageSize.width : imageSize.height);

    // Calculate the actual rect based on rotation
    double left = rect.left;
    double top = rect.top;
    double right = rect.right;
    double bottom = rect.bottom;

    // Manual rotation logic removed to avoid double-rotation.
    // The bounding box is already in the upright coordinate space.

    Rect scaledRect = Rect.fromLTRB(
      left * scaleX,
      top * scaleY,
      right * scaleX,
      bottom * scaleY,
    );

    // Mirroring for front camera
    if (cameraLensDirection == CameraLensDirection.front) {
      if (Platform.isAndroid &&
          isRotated &&
          rotation == InputImageRotation.rotation270deg) {
        // It seems for 270 (portrait), we might not need extra mirroring if the rotation logic above already accounted for "visual" orientation?
        // BUT usually front camera preview IS mirrored.
        // Drawing must be mirrored relative to the widget center.
        final centerX = widgetSize.width / 2;
        scaledRect = Rect.fromLTRB(
          centerX + (centerX - scaledRect.right),
          scaledRect.top,
          centerX + (centerX - scaledRect.left),
          scaledRect.bottom,
        );
      } else {
        final centerX = widgetSize.width / 2;
        scaledRect = Rect.fromLTRB(
          centerX + (centerX - scaledRect.right),
          scaledRect.top,
          centerX + (centerX - scaledRect.left),
          scaledRect.bottom,
        );
      }
    }

    return scaledRect;
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return oldDelegate.boundingBox != boundingBox ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.rotation != rotation ||
        oldDelegate.color != color;
  }
}

class FullScreenCameraPreview extends StatelessWidget {
  final CameraController controller;
  final Widget? child;

  const FullScreenCameraPreview({
    super.key,
    required this.controller,
    this.child,
  });

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
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CameraPreviewWidget(controller: controller),
                  if (child != null) child!,
                ],
              ),
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
