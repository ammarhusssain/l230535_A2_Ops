import pandas as pd, tensorflow as tf
IMG_DIMENSION = 224
AUTOTUNE = tf.data.AUTOTUNE

def load(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_DIMENSION, IMG_DIMENSION])
    return tf.cast(img, tf.float32), tf.cast(label, tf.float32)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name='augment')

def make_ds(csv, batch=32, training=False, seed=42):
    df = pd.read_csv(csv)
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.cache()
    ds = ds.batch(batch)
    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)