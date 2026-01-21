import 'package:flutter/material.dart';
import 'package:face_detection/main.dart';
import 'package:face_detection/services/face_recognition_service.dart'
    as recognition;
import 'package:image_picker/image_picker.dart';

import 'package:flutter/foundation.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import 'package:image/image.dart' as img;

class RegistrationScreen extends StatefulWidget {
  const RegistrationScreen({super.key});

  @override
  State<RegistrationScreen> createState() => _RegistrationScreenState();
}

class _RegistrationScreenState extends State<RegistrationScreen> {
  final recognition.FaceRecognitionService _recognitionService =
      recognition.FaceRecognitionService();
  final ImagePicker _imagePicker = ImagePicker();
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableLandmarks: true,
      enableClassification: false,
      enableTracking: true,
      performanceMode: FaceDetectorMode.fast,
    ),
  );

  RegistrationStatus _currentStatus = RegistrationStatus.idle;
  bool _isRegistered = false;

  @override
  void initState() {
    super.initState();
    _initializeService();
  }

  Future<void> _initializeService() async {
    if (!_recognitionService.isInitialized) {
      await _recognitionService.initialize();
      if (mounted) {
        setState(() {});
      }
    } else {
      // Even if initialized, we might need to refresh if faces were loaded externally
      // or if this screen was rebuilt. But typically registeredFaces is a getter.
      // Just to be safe, ensuring we reflect current state.
      if (mounted) {
        setState(() {});
      }
    }
  }

  Future<void> _registerFace() async {
    try {
      final ImageSource? source = await _showImageSourceDialog();
      if (source == null) {
        return;
      }

      setState(() => _currentStatus = RegistrationStatus.processingImage);
      await Future.delayed(const Duration(milliseconds: 500)); // UX delay

      final XFile? pickedFile = await _imagePicker.pickImage(source: source);

      if (pickedFile == null) {
        setState(() => _currentStatus = RegistrationStatus.idle);
        return;
      }

      setState(() => _currentStatus = RegistrationStatus.detectingFace);
      await Future.delayed(const Duration(milliseconds: 500));

      // Detect face
      final inputImage = InputImage.fromFilePath(pickedFile.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        _showSnackBar('No face detected in the image');
        setState(() => _currentStatus = RegistrationStatus.idle);
        return;
      }

      if (faces.length > 1) {
        _showSnackBar(
          'Multiple faces detected. Please select an image with one face.',
        );
        setState(() => _currentStatus = RegistrationStatus.idle);
        return;
      }

      final face = faces.first;
      final bytes = await pickedFile.readAsBytes();

      // Decode image in background to avoid UI freeze
      final image = await compute(img.decodeImage, bytes);
      if (image == null) {
        _showSnackBar('Could not decode image');
        setState(() => _currentStatus = RegistrationStatus.idle);
        return;
      }

      setState(() => _currentStatus = RegistrationStatus.croppingFace);
      await Future.delayed(const Duration(milliseconds: 500));

      // Crop face (on main thread)
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
        setState(() => _currentStatus = RegistrationStatus.idle);
        return;
      }

      // Temporarily hide status to show dialog
      setState(() => _currentStatus = RegistrationStatus.idle);

      if (mounted) {
        final name = await _showNameInputDialog();
        if (name != null && name.isNotEmpty) {
          setState(() => _currentStatus = RegistrationStatus.registering);
          await Future.delayed(const Duration(milliseconds: 500));

          // _recognitionService.clearAllFaces(); // Removed to allow multiple
          await _recognitionService.registerFace(name, croppedFace);

          setState(() {
            _isRegistered = true;
            _currentStatus = RegistrationStatus.complete;
          });

          await Future.delayed(const Duration(seconds: 1));
          if (mounted) {
            setState(() => _currentStatus = RegistrationStatus.idle);
            _showSnackBar('Face registered successfully!');
          }
        }
      }
    } catch (e) {
      _showSnackBar('Error registering face: $e');
      setState(() => _currentStatus = RegistrationStatus.idle);
    } finally {
      // handled inside steps
    }
  }

  void _navigateToFaceDetection() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (context) => const FaceDetectionScreen()),
    );
  }

  Future<String?> _showNameInputDialog() async {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Enter Name'),
        content: TextField(
          controller: controller,
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

  Future<ImageSource?> _showImageSourceDialog() async {
    return showDialog<ImageSource>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Select Image Source'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.camera_alt),
              title: const Text('Camera'),
              onTap: () => Navigator.pop(context, ImageSource.camera),
            ),
            ListTile(
              leading: const Icon(Icons.photo_library),
              title: const Text('Gallery'),
              onTap: () => Navigator.pop(context, ImageSource.gallery),
            ),
          ],
        ),
      ),
    );
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    final registeredFaces = _recognitionService.registeredFaces;

    return Scaffold(
      body: Stack(
        children: [
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                children: [
                  Expanded(
                    child: SingleChildScrollView(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          const SizedBox(height: 48),
                          const Icon(
                            Icons.face_retouching_natural,
                            size: 80,
                            color: Colors.deepPurple,
                          ),
                          const SizedBox(height: 24),
                          const Text(
                            'Face Registration',
                            style: TextStyle(
                              fontSize: 28,
                              fontWeight: FontWeight.bold,
                            ),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 12),
                          const Text(
                            'Register one or more faces to recognize.',
                            style: TextStyle(fontSize: 16, color: Colors.grey),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 32),
                          if (registeredFaces.isNotEmpty) ...[
                            const Text(
                              'Registered Faces:',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            const SizedBox(height: 12),
                            ListView.builder(
                              shrinkWrap: true,
                              physics: const NeverScrollableScrollPhysics(),
                              itemCount: registeredFaces.length,
                              itemBuilder: (context, index) {
                                final face = registeredFaces[index];

                                return Card(
                                  elevation: 2,
                                  margin: const EdgeInsets.symmetric(
                                    vertical: 6,
                                    horizontal: 4,
                                  ),
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: ListTile(
                                    contentPadding: const EdgeInsets.symmetric(
                                      horizontal: 16,
                                      vertical: 8,
                                    ),
                                    leading: Container(
                                      width: 56,
                                      height: 56,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        border: Border.all(
                                          color: Colors.deepPurple.shade100,
                                          width: 2,
                                        ),
                                        boxShadow: [
                                          BoxShadow(
                                            color: Colors.black.withOpacity(
                                              0.1,
                                            ),
                                            blurRadius: 4,
                                            offset: const Offset(0, 2),
                                          ),
                                        ],
                                      ),
                                      child: ClipOval(
                                        child: face.faceBytes != null
                                            ? Image.memory(
                                                face.faceBytes!,
                                                fit: BoxFit.cover,
                                              )
                                            : Container(
                                                color:
                                                    Colors.deepPurple.shade50,
                                                child: Icon(
                                                  Icons.person,
                                                  size: 30,
                                                  color: Colors
                                                      .deepPurple
                                                      .shade300,
                                                ),
                                              ),
                                      ),
                                    ),
                                    title: Text(
                                      face.name,
                                      style: const TextStyle(
                                        fontWeight: FontWeight.bold,
                                        fontSize: 16,
                                      ),
                                    ),
                                    subtitle: Text(
                                      'Registered',
                                      style: TextStyle(
                                        color: Colors.green.shade600,
                                        fontSize: 12,
                                      ),
                                    ),
                                    trailing: IconButton(
                                      icon: const Icon(
                                        Icons.delete_outline,
                                        color: Colors.redAccent,
                                      ),
                                      onPressed: () {
                                        _recognitionService.removeFace(
                                          face.name,
                                        );
                                        setState(() {
                                          _isRegistered = _recognitionService
                                              .registeredFaces
                                              .isNotEmpty;
                                        });
                                      },
                                    ),
                                  ),
                                );
                              },
                            ),
                            const SizedBox(height: 24),
                          ],
                        ],
                      ),
                    ),
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      FilledButton.icon(
                        onPressed: _registerFace,
                        icon: const Icon(Icons.add_a_photo),
                        label: const Text('Register New Face'),
                        style: FilledButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                        ),
                      ),
                      const SizedBox(height: 12),
                      if (_isRegistered)
                        OutlinedButton(
                          onPressed: _navigateToFaceDetection,
                          style: OutlinedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 16),
                          ),
                          child: const Text('Start Recognition'),
                        ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          if (_currentStatus != RegistrationStatus.idle)
            Container(
              color: Colors.black54,
              child: Center(
                child: RegistrationStepper(currentStatus: _currentStatus),
              ),
            ),
        ],
      ),
    );
  }
}

enum RegistrationStatus {
  idle,
  processingImage,
  detectingFace,
  croppingFace,
  registering,
  complete,
}

class RegistrationStepper extends StatelessWidget {
  final RegistrationStatus currentStatus;

  const RegistrationStepper({super.key, required this.currentStatus});

  @override
  Widget build(BuildContext context) {
    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Registration Progress',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 24),
            _buildStep('Processing Image', RegistrationStatus.processingImage),
            _buildStep('Detecting Face', RegistrationStatus.detectingFace),
            _buildStep('Optimizing Face', RegistrationStatus.croppingFace),
            _buildStep('Saving to Storage', RegistrationStatus.registering),
            _buildStep('Registration Complete', RegistrationStatus.complete),
          ],
        ),
      ),
    );
  }

  Widget _buildStep(String title, RegistrationStatus stepStatus) {
    final isCompleted = currentStatus.index > stepStatus.index;
    final isCurrent = currentStatus == stepStatus;
    final isPending = currentStatus.index < stepStatus.index;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12.0),
      child: Row(
        children: [
          Container(
            width: 28,
            height: 28,
            decoration: BoxDecoration(
              color: isCompleted
                  ? Colors.green
                  : isCurrent
                  ? Colors.deepPurple
                  : Colors.grey.shade300,
              shape: BoxShape.circle,
            ),
            child: Center(
              child: isCompleted
                  ? const Icon(Icons.check, size: 16, color: Colors.white)
                  : isCurrent
                  ? const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : Text(
                      (stepStatus.index).toString(),
                      style: TextStyle(
                        color: Colors.grey.shade600,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Text(
              title,
              style: TextStyle(
                color: isPending ? Colors.grey : Colors.black87,
                fontWeight: isCurrent ? FontWeight.bold : FontWeight.normal,
                fontSize: 16,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
