import time
import requests
import tensorflow as tf
IMG = 224


class ModelServiceError(Exception):
    """Raised when the cloud model API cannot answer (unreachable, rejected, or failed)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


class LocalPredictor:
    """Loads the model ONCE when the app starts."""

    def __init__(self, cfg):
        self.model = tf.keras.models.load_model(cfg.MODEL_PATH)
        self.threshold, self.name, self.version = cfg.THRESHOLD, cfg.MODEL_NAME, cfg.MODEL_VERSION
        self.model(tf.zeros((1, IMG, IMG, 3)))

    def preprocess(self, data: bytes):
        img = tf.io.decode_image(data, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG, IMG])
        return tf.cast(img, tf.float32)[None]

    def predict(self, data: bytes, filename=None):
        t = time.perf_counter()
        p = float(self.model(self.preprocess(data), training=False)[0][0])
        is_ai = p >= self.threshold
        return {'label': 'AI-generated' if is_ai else 'Real', 'probability_ai': round(p, 4),
                'confidence': round(p if is_ai else 1 - p, 4), 'threshold': self.threshold,
                'model_name': self.name, 'model_version': self.version,
                'latency_ms': round((time.perf_counter() - t) * 1000, 1)}


class RemotePredictor:
    """Sends the image to the cloud model API (Task 8)."""

    def __init__(self, cfg):
        self.url, self.key = cfg.API_URL.rstrip('/') + '/api/predict', cfg.API_KEY

    def predict(self, data: bytes, filename='upload.jpg'):
        try:
            r = requests.post(self.url, files={'image': (filename, data)}, headers={'X-API-Key': self.key}, timeout=30)
        except requests.RequestException:
            raise ModelServiceError(503, 'The model service is unreachable. Please try again shortly.')
        if r.status_code != 200:
            try:
                msg = r.json().get('error', r.text)
            except ValueError:
                msg = r.text[:200]
            raise ModelServiceError(502 if r.status_code >= 500 else r.status_code, f'Model service error: {msg}')
        return r.json()


def build_predictor(cfg):
    return RemotePredictor(cfg) if cfg.INFERENCE_MODE == 'remote' else LocalPredictor(cfg)
