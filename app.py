import os
import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms

app = Flask(__name__)

# ─────────────────────────────────────────────
# BASE DIRECTORY (RENDER SAFE PATHS)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# ─────────────────────────────────────────────
# CLASSES
# ─────────────────────────────────────────────
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

CLASS_EMOJI = {
    'buildings': '🏙️',
    'forest':    '🌲',
    'glacier':   '🧊',
    'mountain':  '⛰️',
    'sea':       '🌊',
    'street':    '🛣️',
}

# ─────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────
PYTORCH_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Julianna_model.pth')
TENSORFLOW_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Julianna_model.keras')

# ─────────────────────────────────────────────
# LAZY LOADING (IMPORTANT FOR RENDER)
# ─────────────────────────────────────────────
pytorch_model = None
pytorch_device = None
tensorflow_model = None


def load_pytorch_model():
    try:
        from models.cnn1 import CNN1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNN1(num_classes=6)

        checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        print(f"[PyTorch] Model loaded on {device}")
        return model, device

    except Exception as e:
        print(f"[PyTorch ERROR] {e}")
        return None, None


def load_tensorflow_model():
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(TENSORFLOW_MODEL_PATH)

        print("[TensorFlow] Model loaded")
        return model

    except Exception as e:
        print(f"[TensorFlow ERROR] {e}")
        return None


def get_pytorch_model():
    global pytorch_model, pytorch_device
    if pytorch_model is None:
        pytorch_model, pytorch_device = load_pytorch_model()
    return pytorch_model, pytorch_device


def get_tensorflow_model():
    global tensorflow_model
    if tensorflow_model is None:
        tensorflow_model = load_tensorflow_model()
    return tensorflow_model


# ─────────────────────────────────────────────
# PYTORCH TRANSFORM
# ─────────────────────────────────────────────
transform_pytorch = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


def predict_pytorch(image_bytes, model, device):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transform_pytorch(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    return (
        CLASSES[pred_idx],
        float(probs[pred_idx]) * 100,
        probs.cpu().numpy().tolist()
    )


def predict_tensorflow(image_bytes, model):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((228, 228))

    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return (
        CLASSES[pred_idx],
        float(probs[pred_idx]) * 100,
        probs.tolist()
    )


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template(
        'index.html',
        pytorch_ok=True,
        tensorflow_ok=True
    )


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image received'}), 400

    file = request.files['image']
    model_type = request.form.get('model', 'pytorch')

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    image_bytes = file.read()

    try:
        # ── PYTORCH ──
        if model_type == 'pytorch':
            model, device = get_pytorch_model()
            if model is None:
                return jsonify({'error': 'PyTorch model not available'}), 503

            label, confidence, all_probs = predict_pytorch(
                image_bytes, model, device
            )

        # ── TENSORFLOW ──
        elif model_type == 'tensorflow':
            model = get_tensorflow_model()
            if model is None:
                return jsonify({'error': 'TensorFlow model not available'}), 503

            label, confidence, all_probs = predict_tensorflow(
                image_bytes, model
            )

        else:
            return jsonify({'error': 'Unknown model type'}), 400

        return jsonify({
            'label': label,
            'emoji': CLASS_EMOJI.get(label, ''),
            'confidence': round(confidence, 2),
            'all_probs': {
                CLASSES[i]: round(all_probs[i] * 100, 2)
                for i in range(len(CLASSES))
            },
            'model_used': model_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# ENTRY POINT (RENDER SAFE)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)