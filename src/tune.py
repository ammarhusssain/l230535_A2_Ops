import keras_tuner as kt, tensorflow as tf
from src.pipeline import make_ds
from src.model import mobilenet, compile_model
tr = make_ds('data/splits/train.csv', training=True); va = make_ds('data/splits/val.csv')
def build(hp):
    m, base = mobilenet(hp.Float('dropout', 0.1, 0.5, step=0.1))
    n = hp.Choice('unfrozen_layers', [0, 10, 20, 40])
    base.trainable = n > 0
    for l in base.layers[:-n] if n else base.layers:
        l.trainable = False
    compile_model(m, hp.Float('lr', 1e-5, 1e-3, sampling='log'))
    return m
t = kt.RandomSearch(build, objective='val_loss', max_trials=8, directory='tuner', project_name='visionguard', seed=42)
t.search(tr, validation_data=va, epochs=5,
         callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
t.results_summary()