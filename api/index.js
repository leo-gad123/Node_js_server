const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs');
const cors = require('cors');
const jpeg = require('jpeg-js');
const png = require('pngjs').PNG;
const fs = require('fs');
const path = require('path');

const CLASSES = ['Healthy', 'Gray_Leaf_Spot', 'Blight', 'Common_Rust'];
const IMG_SIZE = 300;

const app = express();
app.use(cors());

const upload = multer({ storage: multer.memoryStorage() });

let model = null;
let modelLoading = null;

async function loadModel() {
  const modelDir = path.join(__dirname, '..', 'model_json');
  const modelJson = JSON.parse(fs.readFileSync(path.join(modelDir, 'model.json'), 'utf-8'));

  const weightBuffers = [];
  for (const p of modelJson.weightsManifest[0].paths) {
    weightBuffers.push(fs.readFileSync(path.join(modelDir, p)).buffer);
  }

  const totalSize = weightBuffers.reduce((s, b) => s + b.byteLength, 0);
  const allWeights = new Uint8Array(totalSize);
  let offset = 0;
  for (const buf of weightBuffers) {
    allWeights.set(new Uint8Array(buf), offset);
    offset += buf.byteLength;
  }

  const weightSpecs = modelJson.weightsManifest[0].weights.map(w => ({
    name: w.name,
    shape: w.shape,
    dtype: w.dtype
  }));

  model = await tf.loadGraphModel(tf.io.fromMemory({
    modelTopology: modelJson.modelTopology,
    weightSpecs: weightSpecs,
    weightData: allWeights.buffer
  }));
}

function getModel() {
  if (model) return Promise.resolve(model);
  if (!modelLoading) {
    modelLoading = loadModel().then(() => model).catch(err => {
      modelLoading = null;
      throw err;
    });
  }
  return modelLoading;
}

function decodeImage(buffer) {
  const isPng =
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47;

  if (isPng) {
    const img = png.sync.read(buffer);
    return { data: new Uint8Array(img.data), width: img.width, height: img.height };
  }

  return jpeg.decode(buffer, { useTArray: true });
}

function preprocessImage(buffer) {
  const { data, width, height } = decodeImage(buffer);

  return tf.tidy(() => {
    let tensor = tf.tensor3d(data, [height, width, 4], 'float32');
    tensor = tensor.slice([0, 0, 0], [-1, -1, 3]);
    tensor = tf.image.resizeBilinear(tensor, [IMG_SIZE, IMG_SIZE]);
    return tensor.expandDims(0);
  });
}

app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image provided' });

    const loadedModel = await getModel();
    const tensor = preprocessImage(req.file.buffer);
    const prediction = loadedModel.predict(tensor);
    const scores = Array.from(await prediction.data());

    tf.dispose([tensor, prediction]);

    let maxIndex = 0;
    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > scores[maxIndex]) maxIndex = i;
    }

    res.json({
      class: CLASSES[maxIndex],
      confidence: Number(scores[maxIndex].toFixed(6)),
      scores: Object.fromEntries(CLASSES.map((c, i) => [c, Number(scores[i].toFixed(6))]))
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    modelLoaded: model !== null,
    classes: CLASSES,
    inputSize: IMG_SIZE
  });
});

module.exports = app;
