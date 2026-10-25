"""Convert the champion Keras model to TensorFlow Lite for the Vercel API, and check it gives the same answers."""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

KERAS_PATH = 'models/champion.keras'
SAVED_DIR = 'models/champion_savedmodel'
TFLITE_PATH = 'cloud_api/model.tflite'

# 1. Keras -> SavedModel -> TFLite
model = tf.keras.models.load_model(KERAS_PATH)
model.export(SAVED_DIR)
tflite_bytes = tf.lite.TFLiteConverter.from_saved_model(SAVED_DIR).convert()
os.makedirs('cloud_api', exist_ok=True)
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_bytes)
print(f'Keras model : {os.path.getsize(KERAS_PATH) / 1e6:.1f} MB')
print(f'TFLite model: {len(tflite_bytes) / 1e6:.1f} MB  ->  {TFLITE_PATH}')

# 2. Verify on 20 test images
interp = tf.lite.Interpreter(model_content=tflite_bytes)
interp.allocate_tensors()
inp, out = interp.get_input_details()[0]['index'], interp.get_output_details()[0]['index']


def tflite_prob(x):
    interp.set_tensor(inp, x)
    interp.invoke()
    return float(interp.get_tensor(out)[0][0])


def tf_preprocess(path):          # exactly as in training (src/pipeline.py)
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return tf.cast(tf.image.resize(img, [224, 224]), tf.float32)[None].numpy()


def pil_preprocess(path):         # what the Vercel API does (no TensorFlow there)
    img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)[None]


paths = pd.read_csv('data/splits/test.csv').path.head(20)
d_model, d_pre = [], []
for p in paths:
    x = tf_preprocess(p)
    k = float(model(x, training=False)[0][0])
    d_model.append(abs(k - tflite_prob(x)))                    # Keras vs TFLite, same input
    d_pre.append(abs(k - tflite_prob(pil_preprocess(p))))      # plus the Pillow preprocessing used on Vercel
print(f'Max |Keras - TFLite| (same input)       : {max(d_model):.6f}')
print(f'Max |Keras - TFLite with Pillow resize| : {max(d_pre):.4f}  (mean {np.mean(d_pre):.4f})')
