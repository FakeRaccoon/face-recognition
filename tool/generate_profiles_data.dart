import 'dart:io';
import 'dart:convert';
import 'dart:math';

void main() async {
  final Directory seedsDir = Directory('seeds');
  final File outputFile = File('assets/data/seeds_data.json');
  int count = 0;

  if (!await seedsDir.exists()) {
    print('Error: seeds directory not found at ${seedsDir.path}');
    return;
  }

  // Ensure output directory exists
  if (!await outputFile.parent.exists()) {
    await outputFile.parent.create(recursive: true);
  }

  final List<Map<String, dynamic>> registeredFaces = [];
  final Random random = Random();
  final int embeddingSize = 192; // MobileFaceNet standard

  // Helper to normalize embedding
  List<double> normalize(List<double> embedding) {
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

  // Helper to generate a random embedding
  List<double> generateRandomEmbedding() {
    final raw = List.generate(
      embeddingSize,
      (_) => random.nextDouble() * 2 - 1,
    );
    return normalize(raw);
  }

  print('Scanning ${seedsDir.path}...');

  await for (final FileSystemEntity entity in seedsDir.list()) {
    if (entity is File) {
      final String filename = entity.uri.pathSegments.last;

      // Filter for image files if needed, but for now take all files in seeds
      if (!filename.toLowerCase().endsWith('.jpg') &&
          !filename.toLowerCase().endsWith('.png') &&
          !filename.toLowerCase().endsWith('.jpeg')) {
        continue;
      }

      final String name =
          filename; // Use full filename as requested or basename
      // To create a "compatible" entry without image processing, we just need registered data.
      // We'll generate 5 random embeddings per profile to match the "multi-angle" feature

      final List<List<double>> embeddings = List.generate(
        1,
        (_) => generateRandomEmbedding(),
      );

      final Map<String, dynamic> faceData = {
        'name': name,
        'embeddings': embeddings,
        'faceBytes': null, // No encoded image bytes to save space/time
      };

      registeredFaces.add(faceData);

      count++;
      if (count % 100 == 0) {
        print('Processed $count seeds...');
      }
    }
  }

  final String jsonOutput = jsonEncode(registeredFaces);
  await outputFile.writeAsString(jsonOutput);

  print('Successfully generated data for ${registeredFaces.length} seeds.');
  print('Output saved to: ${outputFile.path}');
}
