import 'dart:developer' show log;
import 'dart:math' hide log;
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

const int _inputSize = 112;

/// Top-level function for isolate preprocessing.
/// Resizes image and creates normalized Float32List tensor.
/// Runs in background isolate to avoid UI jank.
Float32List? _preprocessImageIsolate(img.Image image) {
  try {
    // Resize to 112x112 using bilinear interpolation (faster than cubic)
    final resized = img.copyResize(
      image,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.linear,
    );

    // Create flat Float32List tensor [1 * 112 * 112 * 3]
    // Using typed data is much faster than nested List<double>
    final int tensorSize = _inputSize * _inputSize * 3;
    final Float32List tensor = Float32List(tensorSize);

    int idx = 0;
    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        // Normalize to [-1, 1] range expected by MobileFaceNet
        tensor[idx++] = (pixel.r.toDouble() - 127.5) / 127.5;
        tensor[idx++] = (pixel.g.toDouble() - 127.5) / 127.5;
        tensor[idx++] = (pixel.b.toDouble() - 127.5) / 127.5;
      }
    }

    return tensor;
  } catch (e) {
    return null;
  }
}

class RegisteredFace {
  final String name;
  final List<double> embedding;
  final Uint8List? faceBytes;

  RegisteredFace({required this.name, required this.embedding, this.faceBytes});

  factory RegisteredFace.fromJson(Map<String, dynamic> json) => RegisteredFace(
    name: json['name'] as String,
    embedding: (json['embedding'] as List).cast<double>(),
    faceBytes: json['faceBytes'] != null
        ? base64Decode(json['faceBytes'] as String)
        : null,
  );

  Map<String, dynamic> toJson() => {
    'name': name,
    'embedding': embedding,
    'faceBytes': faceBytes != null ? base64Encode(faceBytes!) : null,
  };
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
  static final FaceRecognitionService _instance =
      FaceRecognitionService._internal();

  factory FaceRecognitionService() {
    return _instance;
  }

  FaceRecognitionService._internal();

  static const String _modelPath = 'assets/models/mobilefacenet.tflite';
  static const double _threshold = 0.7;
  static const int _minFaceSize = 50;

  Interpreter? _interpreter;
  List<int>? _outputShape;
  final List<RegisteredFace> _registeredFaces = [];

  bool get isInitialized => _interpreter != null;
  List<RegisteredFace> get registeredFaces =>
      List.unmodifiable(_registeredFaces);

  Future<void> initialize() async {
    try {
      final options = InterpreterOptions();

      options.addDelegate(GpuDelegateV2());

      _interpreter = await Interpreter.fromAsset(_modelPath, options: options);

      // Get actual output shape from the model
      final outputTensor = _interpreter!.getOutputTensor(0);
      _outputShape = outputTensor.shape;
      log('MobileFaceNet model loaded successfully');
      log('Input shape: ${_interpreter!.getInputTensor(0).shape}');
      log('Output shape: $_outputShape');
      await _loadFaces();
    } catch (e) {
      log('Error loading MobileFaceNet model: $e');
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
      log('Interpreter not initialized');
      return null;
    }

    try {
      // Preprocess the image in a background isolate to avoid UI jank
      final inputData = await compute(_preprocessImageIsolate, faceImage);
      if (inputData == null) return null;

      // Reshape Float32List to 4D tensor for TFLite
      final input = inputData.buffer.asFloat32List().reshape([
        1,
        _inputSize,
        _inputSize,
        3,
      ]);

      // Prepare output buffer based on actual model output shape
      final outputSize = _embeddingSize;
      final output = List.generate(1, (_) => List.filled(outputSize, 0.0));

      // Run inference
      _interpreter!.run(input, output);

      // Extract and normalize the embedding
      final embedding = output[0];
      final normalized = _normalizeEmbedding(embedding);

      log('Generated embedding with ${normalized.length} dimensions');
      return normalized;
    } catch (e) {
      log('Error getting embedding: $e');
      return null;
    }
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
      log('Embedding size mismatch: ${a.length} vs ${b.length}');
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
      // Check if this face is already registered (by similarity)
      // If found, we'll remove the old one and replace it with this new one
      // effectively updating the registration.
      List<String> facesToRemove = [];

      // 1. Check for same name (explicit update)
      facesToRemove.add(name);

      // 2. Check for high similarity (implicit update/deduplication) REMOVED
      // We should not automatically remove other people just because they look similar.
      // We only strictly enforce uniqueness by Name.
      /*
      for (final face in _registeredFaces) {
        final similarity = _cosineSimilarity(embedding, face.embedding);
        if (similarity > _threshold) {
          log(
            'Found existing face "${face.name}" with similarity ${similarity.toStringAsFixed(3)}. Updating...',
          );
          facesToRemove.add(face.name);
        }
      }
      */

      // Remove duplicates
      _registeredFaces.removeWhere((f) => facesToRemove.contains(f.name));

      // Encode image for display
      final faceBytes = img.encodePng(faceImage);

      _registeredFaces.add(
        RegisteredFace(name: name, embedding: embedding, faceBytes: faceBytes),
      );
      log('Registered face for: $name (embedding size: ${embedding.length})');
      await _saveFaces();
    } else {
      log('Failed to get embedding for: $name');
    }
  }

  void removeFace(String name) {
    _registeredFaces.removeWhere((f) => f.name == name);
    _saveFaces();
  }

  void clearAllFaces() {
    _registeredFaces.clear();
    _saveFaces();
  }

  Future<RecognitionResult?> recognizeFace(img.Image faceImage) async {
    if (_registeredFaces.isEmpty) {
      log('No registered faces to compare');
      return null;
    }

    final embedding = await getEmbedding(faceImage);
    if (embedding == null) {
      log('Failed to get embedding for recognition');
      return null;
    }

    String? bestMatch;
    double bestSimilarity = -1;

    for (final registered in _registeredFaces) {
      final similarity = _cosineSimilarity(embedding, registered.embedding);
      log(
        'Similarity with ${registered.name}: ${similarity.toStringAsFixed(3)}',
      );

      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestMatch = registered.name;
      }
    }

    log(
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

      // Enforce minimum face size for reliable recognition
      if (croppedWidth < _minFaceSize || croppedHeight < _minFaceSize) {
        log(
          'Face too small for reliable recognition: ${croppedWidth}x$croppedHeight (min: $_minFaceSize)',
        );
        return null;
      }

      log(
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
      log('Error cropping face: $e');
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

  Future<void> _saveFaces() async {
    final prefs = await SharedPreferences.getInstance();
    final String facesJson = jsonEncode(
      _registeredFaces.map((f) => f.toJson()).toList(),
    );
    await prefs.setString('registered_faces', facesJson);
    log('Saved ${_registeredFaces.length} faces to storage');
  }

  Future<void> _loadFaces() async {
    final prefs = await SharedPreferences.getInstance();
    final String? facesJson = prefs.getString('registered_faces');
    if (facesJson != null) {
      try {
        final List<dynamic> decoded = jsonDecode(facesJson);
        _registeredFaces.clear();
        _registeredFaces.addAll(
          decoded.map((json) => RegisteredFace.fromJson(json)),
        );
        log('Loaded ${_registeredFaces.length} faces from storage');
      } catch (e) {
        log('Error loading faces: $e');
      }
    }
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
