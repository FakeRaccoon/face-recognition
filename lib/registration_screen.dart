import 'package:flutter/material.dart';
import 'package:face_detection/main.dart';
import 'package:face_detection/services/face_recognition_service.dart'
    as recognition;
import 'package:face_detection/multi_angle_registration_screen.dart';

class RegistrationScreen extends StatefulWidget {
  const RegistrationScreen({super.key});

  @override
  State<RegistrationScreen> createState() => _RegistrationScreenState();
}

class _RegistrationScreenState extends State<RegistrationScreen> {
  final recognition.FaceRecognitionService _recognitionService =
      recognition.FaceRecognitionService();

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

    if (mounted) {
      setState(() {
        _isRegistered = _recognitionService.registeredFaces.isNotEmpty;
      });
    }
  }

  Future<void> _registerFace() async {
    final result = await Navigator.of(context).push<bool>(
      MaterialPageRoute(
        builder: (context) => const MultiAngleRegistrationScreen(),
      ),
    );

    if (result == true && mounted) {
      setState(() {
        _isRegistered = _recognitionService.registeredFaces.isNotEmpty;
      });
    }
  }

  void _navigateToFaceDetection() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (context) => const FaceDetectionScreen()),
    );
  }

  Future<void> _generateDummyData() async {
    setState(() {
      _isRegistered = false; // Show loading or disable
    });

    // Show loading indicator
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => const Center(child: CircularProgressIndicator()),
    );

    await Future.delayed(const Duration(milliseconds: 100)); // UI update

    await _recognitionService.generateDummyFaces(1000);

    if (mounted) {
      Navigator.pop(context); // Close dialog
      setState(() {
        _isRegistered = _recognitionService.registeredFaces.isNotEmpty;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Generated 1000 Dummy Faces')),
      );
    }
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
                        'Register your face from multiple angles for better recognition.',
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
                                        color: Colors.black.withValues(
                                          alpha: 0.1,
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
                                            color: Colors.deepPurple.shade50,
                                            child: Icon(
                                              Icons.person,
                                              size: 30,
                                              color: Colors.deepPurple.shade300,
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
                                  '${face.embeddings.length} angle${face.embeddings.length > 1 ? 's' : ''} registered',
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
