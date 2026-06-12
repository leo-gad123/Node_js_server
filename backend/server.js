'use strict';

const express = require('express');
const multer  = require('multer');
const tf      = require('@tensorflow/tfjs');
const fs      = require('fs');
const cors    = require('cors');
const jpeg    = require('jpeg-js');
const png     = require('pngjs');

const PORT        = process.env.PORT || 3000;
const HOST        = '0.0.0.0';
const MODEL_DIR   = 'model_json';
const UPLOADS_DIR = 'uploads';
const IMG_SIZE    = 300;

const CLASSES = [
    'Healthy',
    'Gray_Leaf_Spot',
    'Blight',
    'Common_Rust'
];

const app = express();
app.use(cors());
app.use('/model', express.static(MODEL_DIR));

fs.mkdirSync(UPLOADS_DIR, { recursive: true });

// ─── MULTER ─────────────────────────────────────────────

const upload = multer({
    dest: UPLOADS_DIR,
    limits: { fileSize: 10 * 1024 * 1024 },
    fileFilter: (req, file, cb) => {
        if (file.fieldname !== 'image') {
            return cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE', file.fieldname));
        }
        cb(null, true);
    }
});

// ─── MODEL ──────────────────────────────────────────────

let model = null;

async function loadModel() {
    const url = `http://${HOST}:${PORT}/model/model.json`;
    console.log('📦 Loading model:', url);

    try {
        model = await tf.loadGraphModel(url);
        console.log('✅ GraphModel loaded');
    } catch (e) {
        console.warn('⚠️ GraphModel failed → LayersModel');
        model = await tf.loadLayersModel(url);
        console.log('✅ LayersModel loaded');
    }
}

// ─── IMAGE DECODING ─────────────────────────────────────

function decodeImage(buffer) {
    const isPng =
        buffer[0] === 0x89 &&
        buffer[1] === 0x50 &&
        buffer[2] === 0x4e &&
        buffer[3] === 0x47;

    if (isPng) {
        const img = png.PNG.sync.read(buffer);
        return { data: new Uint8Array(img.data), width: img.width, height: img.height };
    }

    const decoded = jpeg.decode(buffer, { useTArray: true });
    return decoded;
}

// ─── PREPROCESSING ──────────────────────────────────────

function preprocessImage(filePath) {
    const buffer = fs.readFileSync(filePath);
    const { data, width, height } = decodeImage(buffer);

    return tf.tidy(() => {
        let tensor = tf.tensor3d(data, [height, width, 4], 'float32');

        // remove alpha
        tensor = tensor.slice([0, 0, 0], [-1, -1, 3]);

        // resize
        tensor = tf.image.resizeBilinear(tensor, [IMG_SIZE, IMG_SIZE]);

        // ⚠️ IMPORTANT FIX:
        // Your model training used raw np.array(img)
        // so DO NOT normalize unless retrained
        // tensor = tensor.div(255.0);  ❌ removed

        return tensor.expandDims(0);
    });
}

// ─── PREDICTION ROUTE ───────────────────────────────────

app.post('/predict', (req, res) => {
    upload.single('image')(req, res, async (err) => {

        if (err instanceof multer.MulterError) {
            return res.status(400).json({ error: 'Upload error', message: err.message });
        }
        if (err) {
            return res.status(400).json({ error: err.message });
        }

        const filePath = req.file?.path;

        try {
            if (!model) {
                return res.status(503).json({ error: 'Model not ready' });
            }

            const tensor = preprocessImage(filePath);

            const prediction = model.predict(tensor);

            const scoresTensor = await prediction.data();
            const scores = Array.from(scoresTensor);

            tensor.dispose();
            prediction.dispose();

            // 🔥 SAFE ARGMAX
            let maxIndex = 0;
            for (let i = 1; i < scores.length; i++) {
                if (scores[i] > scores[maxIndex]) {
                    maxIndex = i;
                }
            }

            const confidence = scores[maxIndex];

            return res.json({
                class: CLASSES[maxIndex],
                confidence: Number(confidence.toFixed(6)),
                scores: Object.fromEntries(
                    CLASSES.map((c, i) => [c, Number(scores[i].toFixed(6))])
                )
            });

        } catch (e) {
            console.error('❌ Prediction error:', e);
            return res.status(500).json({
                error: 'Prediction failed',
                message: e.message
            });

        } finally {
            if (filePath) fs.unlink(filePath, () => {});
        }
    });
});

// ─── HEALTH CHECK ───────────────────────────────────────

app.get('/health', (_req, res) => {
    res.json({
        status: 'ok',
        modelLoaded: model !== null,
        classes: CLASSES,
        inputSize: IMG_SIZE
    });
});

// ─── START SERVER ───────────────────────────────────────

app.listen(PORT, HOST, async () => {
    console.log(`🚀 Server running at http://${HOST}:${PORT}`);
    await loadModel();
});