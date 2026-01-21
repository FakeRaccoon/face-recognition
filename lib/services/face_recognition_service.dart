import 'dart:developer' show log;
import 'dart:math' hide log;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

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
  static const int _inputSize = 112;
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
      _interpreter = await Interpreter.fromAsset(_modelPath);

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

      log('Generated embedding with ${normalized.length} dimensions');
      return normalized;
    } catch (e) {
      log('Error getting embedding: $e');
      return null;
    }
  }

  List<List<List<List<double>>>> _preprocessImage(img.Image image) {
    // Step 1: Resize to 112x112 using Lanczos interpolation for better quality
    final resized = img.copyResize(
      image,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.cubic,
    );

    // Step 2: Apply contrast enhancement (histogram equalization on luminance)
    final enhanced = _enhanceContrast(resized);

    // Create input tensor [1, 112, 112, 3]
    final input = List.generate(
      1,
      (_) => List.generate(
        _inputSize,
        (y) => List.generate(_inputSize, (x) {
          final pixel = enhanced.getPixel(x, y);
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

  /// Apply adaptive contrast enhancement to improve face feature visibility
  img.Image _enhanceContrast(img.Image image) {
    // Convert to grayscale for histogram calculation
    final int width = image.width;
    final int height = image.height;

    // Calculate luminance histogram
    final histogram = List.filled(256, 0);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = image.getPixel(x, y);
        final luminance = (0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b)
            .round();
        histogram[luminance.clamp(0, 255)]++;
      }
    }

    // Calculate cumulative distribution function (CDF)
    final cdf = List.filled(256, 0);
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
      cdf[i] = cdf[i - 1] + histogram[i];
    }

    // Find min non-zero CDF value
    int cdfMin = 0;
    for (int i = 0; i < 256; i++) {
      if (cdf[i] > 0) {
        cdfMin = cdf[i];
        break;
      }
    }

    final totalPixels = width * height;
    final denominator = totalPixels - cdfMin;
    if (denominator <= 0) return image;

    // Create lookup table for histogram equalization
    final lut = List.filled(256, 0);
    for (int i = 0; i < 256; i++) {
      lut[i] = (((cdf[i] - cdfMin) * 255) / denominator).round().clamp(0, 255);
    }

    // Apply equalization with blending (50% original, 50% equalized)
    // This prevents over-enhancement
    final result = img.Image(width: width, height: height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = image.getPixel(x, y);
        final r = pixel.r.toInt();
        final g = pixel.g.toInt();
        final b = pixel.b.toInt();

        // Calculate luminance ratio
        final oldLum = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(
          0,
          255,
        );
        final newLum = lut[oldLum];

        if (oldLum > 0) {
          final ratio = newLum / oldLum;
          // Blend 60% original, 40% enhanced for subtle improvement
          final blendRatio = 0.6 + 0.4 * ratio;
          final newR = (r * blendRatio).round().clamp(0, 255);
          final newG = (g * blendRatio).round().clamp(0, 255);
          final newB = (b * blendRatio).round().clamp(0, 255);
          result.setPixelRgb(x, y, newR, newG, newB);
        } else {
          result.setPixelRgb(x, y, r, g, b);
        }
      }
    }

    return result;
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
