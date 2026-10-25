import time, requests, tensorflow as tf
IMG = 224

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

    def predict(self, data: bytes):
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
    def predict(self, data: bytes):
        r = requests.post(self.url, files={'image': ('upload', data)}, headers={'X-API-Key': self.key}, timeout=30)
        r.raise_for_status()
        return r.json()

def build_predictor(cfg):
    return RemotePredictor(cfg) if cfg.INFERENCE_MODE == 'remote' else LocalPredictor(cfg)