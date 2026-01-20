import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';

class RegisteredFace {
  final String name;
  final List<double> embedding;
  final Uint8List? faceBytes;

  RegisteredFace({required this.name, required this.embedding, this.faceBytes});

  Map<String, dynamic> toJson() => {
    'name': name,
    'embedding': embedding,
    // faceBytes is not persisted to JSON for simplicity in this demo,
    // but could be base64 encoded if needed.
  };

  factory RegisteredFace.fromJson(Map<String, dynamic> json) => RegisteredFace(
    name: json['name'] as String,
    embedding: (json['embedding'] as List).cast<double>(),
  );
}

class RecognitionResult {
  final String name;
  final double confidence;
  final bool isMatch;

  RecognitionResult({
    required this.name,
    required this.confidence,
    required this.isMatch,
  });
}

class FaceRecognitionService {
  static const String _modelPath = 'assets/models/mobilefacenet.tflite';
  static const int _inputSize = 112;
  static const double _threshold = 0.5; // Standard threshold for face matching

  Interpreter? _interpreter;
  List<int>? _outputShape;
  final List<RegisteredFace> _registeredFaces = [];

  bool get isInitialized => _interpreter != null;
  List<RegisteredFace> get registeredFaces =>
      List.unmodifiable(_registeredFaces);

  Future<void> initialize() async {
    try {
      _interpreter = await Interpreter.fromAsset(_modelPath);

      // Get actual output shape from the model
      final outputTensor = _interpreter!.getOutputTensor(0);
      _outputShape = outputTensor.shape;
      debugPrint('MobileFaceNet model loaded successfully');
      debugPrint('Input shape: ${_interpreter!.getInputTensor(0).shape}');
      debugPrint('Output shape: $_outputShape');
    } catch (e) {
      debugPrint('Error loading MobileFaceNet model: $e');
      rethrow;
    }
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }

  int get _embeddingSize {
    if (_outputShape != null && _outputShape!.length >= 2) {
      return _outputShape![1];
    }
    return 192; // Default fallback
  }

  Future<List<double>?> getEmbedding(img.Image faceImage) async {
    if (_interpreter == null) {
      debugPrint('Interpreter not initialized');
      return null;
    }

    try {
      // Preprocess the image
      final input = _preprocessImage(faceImage);

      // Prepare output buffer based on actual model output shape
      final outputSize = _embeddingSize;
      final output = List.generate(1, (_) => List.filled(outputSize, 0.0));

      // Run inference
      _interpreter!.run(input, output);

      // Extract and normalize the embedding
      final embedding = output[0];
      final normalized = _normalizeEmbedding(embedding);

      debugPrint('Generated embedding with ${normalized.length} dimensions');
      return normalized;
    } catch (e) {
      debugPrint('Error getting embedding: $e');
      return null;
    }
  }

  List<List<List<List<double>>>> _preprocessImage(img.Image image) {
    // Resize to 112x112
    final resized = img.copyResize(
      image,
      width: _inputSize,
      height: _inputSize,
    );

    // Create input tensor [1, 112, 112, 3]
    final input = List.generate(
      1,
      (_) => List.generate(
        _inputSize,
        (y) => List.generate(_inputSize, (x) {
          final pixel = resized.getPixel(x, y);
          // Normalize to [-1, 1]
          return [
            (pixel.r.toDouble() - 127.5) / 127.5,
            (pixel.g.toDouble() - 127.5) / 127.5,
            (pixel.b.toDouble() - 127.5) / 127.5,
          ];
        }),
      ),
    );

    return input;
  }

  List<double> _normalizeEmbedding(List<double> embedding) {
    // L2 normalization
    double norm = 0;
    for (final value in embedding) {
      norm += value * value;
    }
    norm = sqrt(norm);

    if (norm > 0) {
      return embedding.map((v) => v / norm).toList();
    }
    return embedding;
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    if (a.length != b.length) {
      debugPrint('Embedding size mismatch: ${a.length} vs ${b.length}');
      return 0;
    }

    double dotProduct = 0;
    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
    }

    // Since embeddings are already L2 normalized, dot product = cosine similarity
    return dotProduct;
  }

  Future<void> registerFace(String name, img.Image faceImage) async {
    final embedding = await getEmbedding(faceImage);
    if (embedding != null) {
      // Remove existing registration with same name
      _registeredFaces.removeWhere((f) => f.name == name);

      // Encode image for display
      final faceBytes = img.encodePng(faceImage);

      _registeredFaces.add(
        RegisteredFace(name: name, embedding: embedding, faceBytes: faceBytes),
      );
      debugPrint(
        'Registered face for: $name (embedding size: ${embedding.length})',
      );
    } else {
      debugPrint('Failed to get embedding for: $name');
    }
  }

  void removeFace(String name) {
    _registeredFaces.removeWhere((f) => f.name == name);
  }

  void clearAllFaces() {
    _registeredFaces.clear();
  }

  Future<RecognitionResult?> recognizeFace(img.Image faceImage) async {
    if (_registeredFaces.isEmpty) {
      debugPrint('No registered faces to compare');
      return null;
    }

    final embedding = await getEmbedding(faceImage);
    if (embedding == null) {
      debugPrint('Failed to get embedding for recognition');
      return null;
    }

    String? bestMatch;
    double bestSimilarity = -1;

    for (final registered in _registeredFaces) {
      final similarity = _cosineSimilarity(embedding, registered.embedding);
      debugPrint(
        'Similarity with ${registered.name}: ${similarity.toStringAsFixed(3)}',
      );

      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestMatch = registered.name;
      }
    }

    debugPrint(
      'Best match: $bestMatch with similarity: ${bestSimilarity.toStringAsFixed(3)} (threshold: $_threshold)',
    );

    if (bestMatch != null) {
      return RecognitionResult(
        name: bestMatch,
        confidence: bestSimilarity,
        isMatch: bestSimilarity >= _threshold,
      );
    }

    return null;
  }

  img.Image? cropFace(img.Image fullImage, Rect boundingBox) {
    try {
      // Add padding around the face (20%)
      const padding = 0.2;
      final width = boundingBox.width;
      final height = boundingBox.height;

      int left = (boundingBox.left - width * padding).round().clamp(
        0,
        fullImage.width - 1,
      );
      int top = (boundingBox.top - height * padding).round().clamp(
        0,
        fullImage.height - 1,
      );
      int right = (boundingBox.right + width * padding).round().clamp(
        left + 1,
        fullImage.width,
      );
      int bottom = (boundingBox.bottom + height * padding).round().clamp(
        top + 1,
        fullImage.height,
      );

      final croppedWidth = right - left;
      final croppedHeight = bottom - top;

      if (croppedWidth <= 10 || croppedHeight <= 10) {
        debugPrint('Cropped face too small: ${croppedWidth}x$croppedHeight');
        return null;
      }

      debugPrint(
        'Cropping face: ($left, $top) to ($right, $bottom) size: ${croppedWidth}x$croppedHeight',
      );

      return img.copyCrop(
        fullImage,
        x: left,
        y: top,
        width: croppedWidth,
        height: croppedHeight,
      );
    } catch (e) {
      debugPrint('Error cropping face: $e');
      return null;
    }
  }

  Future<double?> compareFaces(img.Image face1, img.Image face2) async {
    final emb1 = await getEmbedding(face1);
    final emb2 = await getEmbedding(face2);

    if (emb1 == null || emb2 == null) {
      return null;
    }

    return _cosineSimilarity(emb1, emb2);
  }

  double calculateSimilarity(List<double> emb1, List<double> emb2) {
    return _cosineSimilarity(emb1, emb2);
  }
}

class Rect {
  final double left;
  final double top;
  final double right;
  final double bottom;

  Rect({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  double get width => right - left;
  double get height => bottom - top;
}
