import 'dart:developer' show log;
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'services/face_recognition_service.dart';

class FaceComparisonScreen extends StatefulWidget {
  final FaceRecognitionService recognitionService;

  const FaceComparisonScreen({super.key, required this.recognitionService});

  @override
  State<FaceComparisonScreen> createState() => _FaceComparisonScreenState();
}

class _FaceComparisonScreenState extends State<FaceComparisonScreen> {
  final ImagePicker _picker = ImagePicker();
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate),
  );

  File? _imageFile1;
  File? _imageFile2;
  img.Image? _faceImage1;
  img.Image? _faceImage2;
  double? _similarityScore;
  bool _isComparing = false;

  Future<void> _pickImage(int index) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
      );
      if (pickedFile == null) return;

      final file = File(pickedFile.path);
      final bytes = await file.readAsBytes();
      final image = img.decodeImage(bytes);

      if (image == null) {
        if (mounted) _showSnackBar('Could not decode image');
        return;
      }

      // Detect face
      final inputImage = InputImage.fromFilePath(file.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        if (mounted) _showSnackBar('No face detected in Image ${index + 1}');
        return;
      }

      // Get largest face
      Face? largestFace;
      double maxArea = 0;
      for (final face in faces) {
        final area = face.boundingBox.width * face.boundingBox.height;
        if (area > maxArea) {
          maxArea = area;
          largestFace = face;
        }
      }

      if (largestFace == null) return;

      final croppedFace = widget.recognitionService.cropFace(
        image,
        Rect(
          left: largestFace.boundingBox.left,
          top: largestFace.boundingBox.top,
          right: largestFace.boundingBox.right,
          bottom: largestFace.boundingBox.bottom,
        ),
      );

      setState(() {
        if (index == 0) {
          _imageFile1 = file;
          _faceImage1 = croppedFace;
        } else {
          _imageFile2 = file;
          _faceImage2 = croppedFace;
        }
        _similarityScore = null; // Reset result on new image
      });
    } catch (e) {
      log('Error picking image: $e');
      if (mounted) _showSnackBar('Error picking image');
    }
  }

  Future<void> _compareFaces() async {
    if (_faceImage1 == null || _faceImage2 == null) return;

    setState(() {
      _isComparing = true;
    });

    try {
      final score = await widget.recognitionService.compareFaces(
        _faceImage1!,
        _faceImage2!,
      );
      setState(() {
        _similarityScore = score;
      });
    } catch (e) {
      log('Error comparing faces: $e');
      if (mounted) _showSnackBar('Error comparing faces');
    } finally {
      setState(() {
        _isComparing = false;
      });
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  void dispose() {
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Compare Faces'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              Row(
                children: [
                  Expanded(
                    child: _buildImageSection(
                      title: 'Image 1',
                      file: _imageFile1,
                      onTap: () => _pickImage(0),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: _buildImageSection(
                      title: 'Image 2',
                      file: _imageFile2,
                      onTap: () => _pickImage(1),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 32),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed:
                      (_imageFile1 != null &&
                          _imageFile2 != null &&
                          !_isComparing)
                      ? _compareFaces
                      : null,
                  icon: _isComparing
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.compare_arrows),
                  label: Text(_isComparing ? 'Comparing...' : 'Compare Faces'),
                ),
              ),
              const SizedBox(height: 32),
              if (_similarityScore != null)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Theme.of(
                      context,
                    ).colorScheme.surfaceContainerHighest,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Column(
                    children: [
                      Text(
                        'Similarity Score',
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        '${(_similarityScore! * 100).toStringAsFixed(1)}%',
                        style: Theme.of(context).textTheme.displayMedium
                            ?.copyWith(
                              color:
                                  _similarityScore! >
                                      0.45 // Using standard threshold
                                  ? Colors.green
                                  : Colors.red,
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        _similarityScore! > 0.45 ? 'MATCH' : 'NO MATCH',
                        style: Theme.of(context).textTheme.headlineSmall
                            ?.copyWith(
                              color: _similarityScore! > 0.45
                                  ? Colors.green
                                  : Colors.red,
                              fontWeight: FontWeight.bold,
                            ),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImageSection({
    required String title,
    required File? file,
    required VoidCallback onTap,
  }) {
    return Column(
      children: [
        Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Container(
            height: 200,
            decoration: BoxDecoration(
              color: Colors.grey.shade200,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.grey.shade400),
            ),
            child: file != null
                ? ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.file(
                      file,
                      fit: BoxFit.cover,
                      width: double.infinity,
                    ),
                  )
                : const Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.add_a_photo, size: 40, color: Colors.grey),
                        SizedBox(height: 4),
                        Text(
                          'Tap to pick',
                          style: TextStyle(color: Colors.grey),
                        ),
                      ],
                    ),
                  ),
          ),
        ),
      ],
    );
  }
}
