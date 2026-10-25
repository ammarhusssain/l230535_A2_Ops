"""VisionGuard model API for Vercel: TFLite model + Postgres (Neon) history, protected by an API key."""
import io
import os
import sys
import threading
import time

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, os.path.dirname(__file__))
from db import init_db, save_prediction, recent_predictions, db_ok  # noqa: E402

# ---------- settings (all from environment variables) ----------
API_KEY = os.getenv('API_KEY', '')
THRESHOLD = float(os.getenv('THRESHOLD', '0.5'))
MODEL_NAME = os.getenv('MODEL_NAME', 'visionguard')
MODEL_VERSION = os.getenv('MODEL_VERSION', '2')
MAX_MB = int(os.getenv('MAX_UPLOAD_MB', '4'))          # Vercel rejects bodies over 4.5 MB
DB_URL = os.getenv('DATABASE_URL') or os.getenv('POSTGRES_URL') or 'sqlite:////tmp/visionguard.db'
DB_URL = DB_URL.replace('postgres://', 'postgresql+psycopg2://', 1).replace('postgresql://', 'postgresql+psycopg2://', 1)
ALLOWED = {'jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif'}

# ---------- load the model ONCE per server instance ----------
interpreter = Interpreter(model_path=os.path.join(os.path.dirname(__file__), 'model.tflite'))
interpreter.allocate_tensors()
INP = interpreter.get_input_details()[0]['index']
OUT = interpreter.get_output_details()[0]['index']
lock = threading.Lock()                                # the interpreter is not thread-safe

init_db(DB_URL)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_MB * 1024 * 1024


def preprocess(data):
    # RGB, 224x224 bilinear, raw 0-255 pixels; MobileNetV2 scaling is inside the model
    img = Image.open(io.BytesIO(data)).convert('RGB').resize((224, 224), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None]


def predict(data):
    t = time.perf_counter()
    x = preprocess(data)
    with lock:
        interpreter.set_tensor(INP, x)
        interpreter.invoke()
        p = float(interpreter.get_tensor(OUT)[0][0])
    is_ai = p >= THRESHOLD
    return {'label': 'AI-generated' if is_ai else 'Real', 'probability_ai': round(p, 4),
            'confidence': round(p if is_ai else 1 - p, 4), 'threshold': THRESHOLD,
            'model_name': MODEL_NAME, 'model_version': MODEL_VERSION,
            'latency_ms': round((time.perf_counter() - t) * 1000, 1)}


def check_key():
    if not API_KEY or request.headers.get('X-API-Key') != API_KEY:
        return jsonify(error='invalid or missing API key'), 401
    return None


def validate(file):
    if not file or file.filename == '':
        return None, None, (jsonify(error='No file selected.'), 400)
    name = secure_filename(file.filename)
    if '.' not in name or name.rsplit('.', 1)[1].lower() not in ALLOWED:
        return None, None, (jsonify(error='Only JPG, PNG, WEBP, BMP or GIF images are accepted.'), 415)
    data = file.read()                                 # kept in memory only
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None, None, (jsonify(error='This file has an image extension but cannot be decoded as an image.'), 422)
    return name, data, None


@app.get('/')
@app.get('/health')
def health():
    return jsonify(status='ok', service='visionguard-api', model=MODEL_NAME, version=MODEL_VERSION,
                   runtime='tflite', database='ok' if db_ok() else 'down')


@app.post('/api/predict')
def api_predict():
    denied = check_key()
    if denied:
        return denied
    name, data, err = validate(request.files.get('image'))
    if err:
        return err
    res = predict(data)
    save_prediction(name, res, source='api', client_ip=request.headers.get('X-Forwarded-For', request.remote_addr))
    return jsonify(res)


@app.get('/api/history')
def api_history():
    denied = check_key()
    if denied:
        return denied
    rows = recent_predictions(int(request.args.get('n', 20)))
    return jsonify([{'id': r.id, 'created_at': r.created_at.isoformat(), 'filename': r.filename, 'label': r.label,
                     'confidence': r.confidence, 'model': f'{r.model_name} v{r.model_version}'} for r in rows])


@app.errorhandler(HTTPException)
def handle(e):
    return jsonify(error=e.description), e.code
