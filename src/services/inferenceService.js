const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat()

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = score[0] * 100;

    let label, explanation, suggestion;

    if (confidenceScore > 50) {
      label = 'Cancer';
      explanation = "Kemungkinan besar adalah kanker berdasarkan model prediksi.";
      suggestion = "Segera konsultasi dengan dokter terdekat untuk tindakan lebih lanjut.";
    } else {
      label = 'Non-cancer';
      explanation = "Kemungkinan besar bukan kanker berdasarkan model prediksi.";
      suggestion = "Tetap jaga kesehatan kulit dan lakukan pemeriksaan rutin.";
    }

    return { confidenceScore, label, explanation, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`);
  }
}

module.exports = predictClassification;
