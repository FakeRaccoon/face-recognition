import 'dart:developer' show log;
import 'dart:math' hide log;
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

const int _inputSize = 160;

/// Message class for crop operation in isolate
class _CropMessage {
  final img.Image image;
  final int left;
  final int top;
  final int width;
  final int height;

  _CropMessage({
    required this.image,
    required this.left,
    required this.top,
    required this.width,
    required this.height,
  });
}

/// Top-level function to crop face in isolate
img.Image? _cropFaceIsolate(_CropMessage message) {
  try {
    return img.copyCrop(
      message.image,
      x: message.left,
      y: message.top,
      width: message.width,
      height: message.height,
    );
  } catch (e) {
    return null;
  }
}

/// Message class for similarity comparison in isolate
class _SimilarityMessage {
  final List<double> embedding;
  final List<RegisteredFaceData> registeredFaces;

  _SimilarityMessage({required this.embedding, required this.registeredFaces});
}

/// Lightweight data class for isolate (no Uint8List)
class RegisteredFaceData {
  final String name;
  final List<List<double>> embeddings;

  RegisteredFaceData({required this.name, required this.embeddings});
}

/// Result from similarity comparison
class _SimilarityResult {
  final String? bestMatch;
  final double bestSimilarity;

  _SimilarityResult({this.bestMatch, required this.bestSimilarity});
}

/// Top-level function to compare embeddings in isolate
_SimilarityResult _compareSimilarityIsolate(_SimilarityMessage message) {
  String? bestMatch;
  double bestSimilarity = -1;

  for (final registered in message.registeredFaces) {
    double maxSimilarityForPerson = -1;
    for (final storedEmbedding in registered.embeddings) {
      // Cosine similarity (embeddings are L2 normalized)
      double dotProduct = 0;
      for (
        int i = 0;
        i < message.embedding.length && i < storedEmbedding.length;
        i++
      ) {
        dotProduct += message.embedding[i] * storedEmbedding[i];
      }
      if (dotProduct > maxSimilarityForPerson) {
        maxSimilarityForPerson = dotProduct;
      }
    }

    if (maxSimilarityForPerson > bestSimilarity) {
      bestSimilarity = maxSimilarityForPerson;
      bestMatch = registered.name;
    }
  }

  return _SimilarityResult(
    bestMatch: bestMatch,
    bestSimilarity: bestSimilarity,
  );
}

/// Top-level function for isolate preprocessing.
/// Resizes image and creates normalized Float32List tensor.
/// Runs in background isolate to avoid UI jank.
Float32List? _preprocessImageIsolate(img.Image image) {
  try {
    // Resize to 160x160 using bilinear interpolation (faster than cubic)
    final resized = img.copyResize(
      image,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.linear,
    );

    // Create flat Float32List tensor [1 * 160 * 160 * 3]
    // Using typed data is much faster than nested List<double>
    final int tensorSize = _inputSize * _inputSize * 3;
    final Float32List tensor = Float32List(tensorSize);

    int idx = 0;
    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        // Normalize to [-1, 1] range expected by FaceNet
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
  final List<List<double>>
  embeddings; // Multiple embeddings for different angles
  final Uint8List? faceBytes;

  RegisteredFace({
    required this.name,
    required this.embeddings,
    this.faceBytes,
  });

  // Backward-compatible factory: handles both old (single) and new (multiple) format
  factory RegisteredFace.fromJson(Map<String, dynamic> json) {
    List<List<double>> embeddings;

    if (json.containsKey('embeddings')) {
      // New format: multiple embeddings
      embeddings = (json['embeddings'] as List)
          .map((e) => (e as List).cast<double>())
          .toList();
    } else if (json.containsKey('embedding')) {
      // Old format: single embedding - wrap in list for compatibility
      embeddings = [(json['embedding'] as List).cast<double>()];
    } else {
      embeddings = [];
    }

    return RegisteredFace(
      name: json['name'] as String,
      embeddings: embeddings,
      faceBytes: json['faceBytes'] != null
          ? base64Decode(json['faceBytes'] as String)
          : null,
    );
  }

  Map<String, dynamic> toJson() => {
    'name': name,
    'embeddings': embeddings,
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

  static const String _modelPath = 'assets/models/facenet_big.tflite';
  static const double _threshold =
      0.75; // Adjusted for FaceNet (usually slightly lower)
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

      // Load persistent faces first
      await _loadFaces();

      // Then load pre-computed/dummy faces from assets
      await _loadPrecomputedFaces();
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

  /// Register a face with a single image (legacy method, wraps in list)
  Future<void> registerFace(String name, img.Image faceImage) async {
    await registerFaceMultiAngle(name, [faceImage], faceImage);
  }

  /// Register a face with multiple angle images for better recognition
  Future<void> registerFaceMultiAngle(
    String name,
    List<img.Image> faceImages,
    img.Image primaryFaceImage,
  ) async {
    final List<List<double>> embeddings = [];

    for (final faceImage in faceImages) {
      final embedding = await getEmbedding(faceImage);
      if (embedding != null) {
        embeddings.add(embedding);
      }
    }

    if (embeddings.isNotEmpty) {
      // Remove existing face with same name
      _registeredFaces.removeWhere((f) => f.name == name);

      // Encode primary image for display
      final faceBytes = img.encodePng(primaryFaceImage);

      _registeredFaces.add(
        RegisteredFace(
          name: name,
          embeddings: embeddings,
          faceBytes: faceBytes,
        ),
      );
      log('Registered face for: $name (${embeddings.length} embeddings)');
      await _saveFaces();
    } else {
      log('Failed to get any embeddings for: $name');
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

    // Convert to lightweight data for isolate
    final facesData = _registeredFaces
        .map((f) => RegisteredFaceData(name: f.name, embeddings: f.embeddings))
        .toList();

    // Run similarity comparison in isolate to avoid UI jank
    final result = await compute(
      _compareSimilarityIsolate,
      _SimilarityMessage(embedding: embedding, registeredFaces: facesData),
    );

    log(
      'Best match: ${result.bestMatch} with similarity: ${result.bestSimilarity.toStringAsFixed(3)} (threshold: $_threshold)',
    );

    if (result.bestMatch != null) {
      return RecognitionResult(
        name: result.bestMatch!,
        confidence: result.bestSimilarity,
        isMatch: result.bestSimilarity >= _threshold,
      );
    }

    return null;
  }

  Future<img.Image?> cropFace(img.Image fullImage, Rect boundingBox) async {
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

      // Run crop in isolate to avoid UI jank
      return await compute(
        _cropFaceIsolate,
        _CropMessage(
          image: fullImage,
          left: left,
          top: top,
          width: croppedWidth,
          height: croppedHeight,
        ),
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
        final List<RegisteredFace> loadedFaces = decoded
            .map((json) => RegisteredFace.fromJson(json))
            .toList();

        final int currentEmbeddingSize = _embeddingSize;
        final List<RegisteredFace> validFaces = [];
        bool hasChanges = false;

        for (final face in loadedFaces) {
          if (face.embeddings.isNotEmpty &&
              face.embeddings.first.length == currentEmbeddingSize) {
            validFaces.add(face);
          } else {
            log(
              'Removing incompatible face "${face.name}" (Embedding size: ${face.embeddings.firstOrNull?.length} vs Model: $currentEmbeddingSize)',
            );
            hasChanges = true;
          }
        }

        _registeredFaces.clear();
        _registeredFaces.addAll(validFaces);
        log('Loaded ${_registeredFaces.length} valid faces from storage');

        if (hasChanges) {
          await _saveFaces(); // Save the cleaned list
        }
      } catch (e) {
        log('Error loading faces: $e');
      }
    }
  }

  Future<void> generateDummyFaces(int count) async {
    final random = Random();
    final int embeddingSize = _embeddingSize;

    log('Generating $count dummy faces with embedding size: $embeddingSize');

    // Create a large batch of faces
    final List<RegisteredFace> dummyFaces = [];

    for (int i = 0; i < count; i++) {
      final String name = 'Dummy User $i';
      final List<List<double>> embeddings = [];

      // Generate 5 random embeddings per user (simulating 5 angles)
      for (int j = 0; j < 5; j++) {
        // Generate random vector
        final List<double> rawEmbedding = List.generate(
          embeddingSize,
          (_) => random.nextDouble() * 2 - 1, // Range [-1, 1]
        );
        // Normalize it
        embeddings.add(_normalizeEmbedding(rawEmbedding));
      }

      dummyFaces.add(
        RegisteredFace(
          name: name,
          embeddings: embeddings,
          faceBytes: null, // No image data for dummy
        ),
      );
    }

    _registeredFaces.addAll(dummyFaces);
    await _saveFaces();
    log('Added ${dummyFaces.length} dummy faces to storage');
  }

  Future<void> _loadPrecomputedFaces() async {
    try {
      final String jsonString = await rootBundle.loadString(
        'assets/data/profiles_data.json',
      );
      final List<dynamic> decoded = jsonDecode(jsonString);
      final List<RegisteredFace> assetFaces = decoded
          .map((json) => RegisteredFace.fromJson(json))
          .toList();

      int addedCount = 0;
      for (final face in assetFaces) {
        // Only add if not already present
        if (!_registeredFaces.any((existing) => existing.name == face.name)) {
          _registeredFaces.add(face);
          addedCount++;
        }
      }

      log(
        'Loaded $addedCount pre-computed faces from assets (Total: ${_registeredFaces.length})',
      );
    } catch (e) {
      log('Error loading pre-computed faces: $e');
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
