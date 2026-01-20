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
          // _recognitionService.clearAllFaces(); // Removed to allow multiple
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
    final registeredFaces = _recognitionService.registeredFaces;

    return Scaffold(
      body: SafeArea(
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
                              margin: const EdgeInsets.only(bottom: 8),
                              child: ListTile(
                                leading: face.faceBytes != null
                                    ? ClipOval(
                                        child: Image.memory(
                                          face.faceBytes!,
                                          width: 50,
                                          height: 50,
                                          fit: BoxFit.cover,
                                        ),
                                      )
                                    : const CircleAvatar(
                                        backgroundColor:
                                            Colors.deepPurpleAccent,
                                        child: Icon(
                                          Icons.person,
                                          color: Colors.white,
                                        ),
                                      ),
                                title: Text(
                                  face.name,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                trailing: IconButton(
                                  icon: const Icon(
                                    Icons.delete,
                                    color: Colors.red,
                                  ),
                                  onPressed: () {
                                    _recognitionService.removeFace(face.name);
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

              if (_isLoading)
                const Center(child: CircularProgressIndicator())
              else
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
    );
  }
}
