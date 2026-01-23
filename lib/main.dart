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

class VerificationState {
  final double? confidence;
  final String? matchedName;
  final bool isVerified;

  const VerificationState({
    this.confidence,
    this.matchedName,
    this.isVerified = false,
  });

  factory VerificationState.initial() => const VerificationState();

  VerificationState copyWith({
    double? confidence,
    String? matchedName,
    bool? isVerified,
    bool clearConfidence = false,
    bool clearMatchedName = false,
  }) {
    return VerificationState(
      confidence: clearConfidence ? null : (confidence ?? this.confidence),
      matchedName: clearMatchedName ? null : (matchedName ?? this.matchedName),
      isVerified: isVerified ?? this.isVerified,
    );
  }
}

class _FaceDetectionScreenState extends State<FaceDetectionScreen>
    with TickerProviderStateMixin {
  late AnimationController _animationController;
  CameraController? _cameraController;
  bool _isDetecting = false;
  bool _isRecognizing = false;
  int _cameraIndex = 0;
  DateTime? _lastRecognitionTime;

  final _confidenceThreshold = 0.75;

  // Painting state
  // Painting state
  Rect? _targetBoundingBox; // The latest detection result
  final ValueNotifier<Rect?> _boundingBoxNotifier = ValueNotifier(null);
  Size? _imageSize;
  InputImageRotation? _imageRotation;
  late Ticker _ticker; // Ticker for smooth animation

  final recognition.FaceRecognitionService _recognitionService =
      recognition.FaceRecognitionService();

  bool _isRecognitionReady = false;

  // Verification state
  final ValueNotifier<VerificationState> _verificationNotifier = ValueNotifier(
    VerificationState.initial(),
  );

  // Adaptive debounce settings
  static const int _debounceActiveMs = 200; // Fast when matching
  static const int _debounceIdleMs = 500; // Normal when searching
  static const int _debounceVerifiedMs = 1000; // Slow after verification
  int _currentDebounceMs = _debounceIdleMs;

  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableLandmarks: false,
      enableClassification: false,
      enableTracking: true,
      performanceMode: FaceDetectorMode.fast, // Better angle detection
      minFaceSize: 0.08, // Smaller to catch angled faces
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
        if (_boundingBoxNotifier.value == null) {
          _boundingBoxNotifier.value = _targetBoundingBox;
          return;
        }

        // Linear interpolation with 0.15 factor for smooth following
        // We handle nullable rect lerp manually to be safe
        final target = _targetBoundingBox!;
        final current = _boundingBoxNotifier.value!;

        // Check distance to avoid unnecessary repaints
        final dist = (target.center - current.center).distance;
        if (dist < 0.5 && (target.width - current.width).abs() < 0.5) {
          return;
        }

        // Adaptive lerp: fast for large movements (snappy), smooth for small adjustments (stable)
        final double lerpFactor = dist > 30.0 ? 0.7 : 0.2;
        final newRect = Rect.lerp(current, target, lerpFactor);
        if (newRect != null) {
          _boundingBoxNotifier.value = newRect;
        }
      } else {
        // If target is null (lost face), clear current
        if (_boundingBoxNotifier.value != null) {
          _boundingBoxNotifier.value = null;
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
        bool metadataChanged = false;
        if (_imageSize != inputImage.metadata?.size ||
            _imageRotation != inputImage.metadata?.rotation) {
          _imageSize = inputImage.metadata?.size;
          _imageRotation = inputImage.metadata?.rotation;
          metadataChanged = true;
        }

        // Only trigger rebuild if metadata changed (e.g. first frame or rotation change)
        // This ensures FullScreenCameraPreview builds the child with correct size/rotation
        if (metadataChanged) {
          setState(() {});
        }

        _targetBoundingBox = primaryFace?.boundingBox;
      }

      // Release detection lock early - bounding box is updated
      _isDetecting = false;

      if (primaryFace != null) {
        // Skip recognition for faces that are too small (< 5% of image area)
        if (_imageSize != null) {
          final faceArea =
              primaryFace.boundingBox.width * primaryFace.boundingBox.height;
          final imageArea = _imageSize!.width * _imageSize!.height;
          if (faceArea / imageArea < 0.05) {
            return;
          }
        }

        // Don't start recognition if already running
        if (_isRecognizing) return;

        if (_isRecognitionReady &&
            _recognitionService.registeredFaces.isNotEmpty) {
          // Adaptive debounce based on verification state
          final debounceMs = _verificationNotifier.value.isVerified
              ? _debounceVerifiedMs
              : _currentDebounceMs;
          final now = DateTime.now();
          if (_lastRecognitionTime != null &&
              now.difference(_lastRecognitionTime!) <
                  Duration(milliseconds: debounceMs)) {
            return;
          }
          _lastRecognitionTime = now;
          _isRecognizing = true;

          // Convert raw camera frame to upright image FIRST
          final uprightImage = await _convertCameraImageToUpright(cameraImage);
          if (uprightImage == null) {
            _isRecognizing = false;
            return;
          }

          final face = primaryFace;

          final croppedFace = await _recognitionService.cropFace(
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
              if (result.isMatch) {
                if (result.confidence >= _confidenceThreshold) {
                  // Speed up recognition when actively matching
                  _currentDebounceMs = _debounceActiveMs;
                  final currentState = _verificationNotifier.value;

                  if (currentState.matchedName == result.name) {
                    // Same person continuing to match
                    if (!currentState.isVerified) {
                      _verificationNotifier.value = currentState.copyWith(
                        isVerified: true,
                        confidence: result.confidence,
                        matchedName: result.name,
                      );
                    } else {
                      // Just update confidence
                      _verificationNotifier.value = currentState.copyWith(
                        confidence: result.confidence,
                      );
                    }
                  } else {
                    // New person or first match
                    _verificationNotifier.value = VerificationState(
                      isVerified: true,
                      confidence: result.confidence,
                      matchedName: result.name,
                    );
                  }
                } else {
                  _currentDebounceMs = _debounceIdleMs;
                  _verificationNotifier.value = const VerificationState(
                    isVerified: false,
                    confidence: null,
                    matchedName: null,
                  );
                  _animationController.reset();
                }
              } else {
                _currentDebounceMs = _debounceIdleMs;
                _verificationNotifier.value = const VerificationState(
                  isVerified: false,
                  confidence: null,
                  matchedName: null,
                );
                _animationController.reset();
              }
            } else {
              _verificationNotifier.value = const VerificationState(
                isVerified: false,
                confidence: null,
                matchedName: null,
              );
              _animationController.reset();
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

            _verificationNotifier.value = const VerificationState(
              isVerified: false,
              confidence: null,
              matchedName: null,
            );
            _animationController.reset();
          }
          _isRecognizing = false;
        } else {
          faceInfos.add(DetectedFaceInfo(face: primaryFace));
        }
      }

      if (faces.isEmpty && mounted) {
        // Just clear target, ticker will clear current
        _targetBoundingBox = null;

        if (mounted) {
          _targetBoundingBox = null;
          // _boundingBoxNotifier.value = null; // Ticker handles null current if target is null

          _verificationNotifier.value = VerificationState.initial();
        }

        _animationController.reset();
      }
    } catch (e) {
      log('Error detecting faces: $e');
      _isDetecting = false;
      _isRecognizing = false;
    }
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
    _boundingBoxNotifier.dispose();
    _verificationNotifier.dispose();
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

    // Status logic moved to ValueListenableBuilder inside build

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
            child: (_imageSize != null && _imageRotation != null)
                ? ValueListenableBuilder<VerificationState>(
                    valueListenable: _verificationNotifier,
                    builder: (context, verState, _) {
                      Color statusColor = Colors.white;
                      if (verState.confidence != null) {
                        if (verState.confidence! >= _confidenceThreshold) {
                          statusColor = const Color(0xFF00E676);
                        } else {
                          statusColor = Colors.redAccent;
                        }
                      }

                      return ValueListenableBuilder<Rect?>(
                        valueListenable: _boundingBoxNotifier,
                        builder: (context, boundingBox, child) {
                          if (boundingBox == null)
                            return const SizedBox.shrink();
                          return CustomPaint(
                            painter: FacePainter(
                              boundingBox: boundingBox,
                              imageSize: _imageSize!,
                              rotation: _imageRotation!,
                              cameraLensDirection:
                                  _cameraController!.description.lensDirection,
                              color: statusColor,
                            ),
                          );
                        },
                      );
                    },
                  )
                : null,
          ),

          // Use a ValueListenableBuilder for the overlays to avoid full rebuilds
          ValueListenableBuilder<VerificationState>(
            valueListenable: _verificationNotifier,
            builder: (context, state, child) {
              Color statusColor = Colors.white;
              if (state.confidence != null) {
                if (state.confidence! >= _confidenceThreshold) {
                  statusColor = const Color(0xFF00E676);
                } else {
                  statusColor = Colors.redAccent;
                }
              }

              // We need to pass this color to the painter above.
              // Nested builders are tricky for independent updates.
              // But since color depends on verification state, and box depends on box notifier.
              // We should probably nest them or simpler: just put CustomPaint here?
              // No, we want bounding box to be 60fps independent of verification logic.
              // Actually statusColor changes rarely (only when confidence crosses threshold).
              // So maybe it's fine.
              // BUT, the FacePainter needs to re-paint when Box changes.

              return Stack(
                fit: StackFit.expand,
                children: [
                  // To update painter color without rebuilding the whole camera stack...
                  // The painter is inside FullScreenCameraPreview child.
                  // Let's modify the Structure slightly.
                  // actually, let's keep the painter simple for now and maybe move the painter INTO this stack if possible?
                  // No, it needs to be on top of camera.

                  // Let's fix the painter color issue by just rebuilding the painter when verification status changes.
                  // The frequent update is the BoundingBox.

                  // Re-inserting the painter here, replacing the one above?
                  // No, let's leave the painter logic for a moment and focus on overlays.
                  if (state.confidence != null)
                    Positioned(
                      top: 50, // Below AppBar
                      right: 20,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.4), // Blend to UI
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: Colors.white.withOpacity(0.2),
                            width: 1,
                          ),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.analytics_outlined,
                              color: Colors.white.withOpacity(0.8),
                              size: 14,
                            ),
                            const SizedBox(width: 6),
                            Text(
                              '${(state.confidence! * 100).toStringAsFixed(1)}%',
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.9),
                                fontSize: 14,
                                fontWeight: FontWeight.w500,
                                fontFeatures: const [
                                  FontFeature.tabularFigures(),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),

                  Positioned(
                    bottom: 50,
                    left: 0,
                    right: 0,
                    child: Column(
                      children: [
                        if (state.isVerified && state.matchedName != null)
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
                                                    final face =
                                                        _recognitionService
                                                            .registeredFaces
                                                            .firstWhere(
                                                              (f) =>
                                                                  f.name ==
                                                                  state
                                                                      .matchedName,
                                                            );
                                                    if (face.faceBytes !=
                                                        null) {
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
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Text(
                                          state.matchedName!.toUpperCase(),
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

                        if (state.matchedName != null && !state.isVerified)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 20),
                            child: CircularProgressIndicator(
                              value: _animationController.value,
                              color: const Color(0xFF00E676),
                            ),
                          ),

                        // Bottom Status Bar
                        Container(
                          margin: const EdgeInsets.only(
                            top: 20,
                            left: 20,
                            right: 20,
                          ),
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
                                : state.isVerified && state.matchedName != null
                                ? 'Verified, Welcome!'
                                : state.confidence != null
                                ? state.confidence! >= _confidenceThreshold
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
              );
            },
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
