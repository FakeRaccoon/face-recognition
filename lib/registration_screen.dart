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

  bool _isLoading = false;
  bool _isRegistered = false;

  @override
  void initState() {
    super.initState();
    _initializeService();
  }

  Future<void> _initializeService() async {
    if (!_recognitionService.isInitialized) {
      await _recognitionService.initialize();
    }
  }

  Future<void> _registerFace() async {
    setState(() => _isLoading = true);
    try {
      final XFile? pickedFile = await _imagePicker.pickImage(
        source: ImageSource.gallery,
      );

      if (pickedFile == null) {
        setState(() => _isLoading = false);
        return;
      }

      // Detect face
      final inputImage = InputImage.fromFilePath(pickedFile.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        _showSnackBar('No face detected in the image');
        setState(() => _isLoading = false);
        return;
      }

      if (faces.length > 1) {
        _showSnackBar(
          'Multiple faces detected. Please select an image with one face.',
        );
        setState(() => _isLoading = false);
        return;
      }

      final face = faces.first;
      final bytes = await pickedFile.readAsBytes();

      // Decode image in background to avoid UI freeze
      final image = await compute(img.decodeImage, bytes);
      if (image == null) {
        _showSnackBar('Could not decode image');
        setState(() => _isLoading = false);
        return;
      }

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
        setState(() => _isLoading = false);
        return;
      }

      if (mounted) {
        final name = await _showNameInputDialog();
        if (name != null && name.isNotEmpty) {
          _recognitionService.clearAllFaces();
          await _recognitionService.registerFace(name, croppedFace);
          setState(() {
            _isRegistered = true;
          });
          _showSnackBar('Face registered successfully!');
        }
      }
    } catch (e) {
      _showSnackBar('Error registering face: $e');
    } finally {
      if (mounted) setState(() => _isLoading = false);
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

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Icon(
                Icons.face_retouching_natural,
                size: 100,
                color: Colors.deepPurple,
              ),
              const SizedBox(height: 32),
              const Text(
                'Welcome',
                style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              const Text(
                'Please register a face to continue to the recognition screen.',
                style: TextStyle(fontSize: 16, color: Colors.grey),
                textAlign: TextAlign.center,
              ),
              const Spacer(),
              if (_isLoading)
                const Center(child: CircularProgressIndicator())
              else
                FilledButton.icon(
                  onPressed: _registerFace,
                  icon: const Icon(Icons.add_a_photo),
                  label: const Text('Register Face from Gallery'),
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                  ),
                ),
              const SizedBox(height: 16),
              if (_isRegistered)
                FilledButton(
                  onPressed: _navigateToFaceDetection,
                  child: const Text('Start Recognition'),
                ),
              const Spacer(),
            ],
          ),
        ),
      ),
    );
  }
}
