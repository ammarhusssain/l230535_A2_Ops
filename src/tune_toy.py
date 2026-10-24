import keras_tuner as kt
import tensorflow as tf

(x, y), _ = tf.keras.datasets.fashion_mnist.load_data()
x = x / 255.0

def build(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp.Int('units', 32, 512, step=32), activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

t = kt.RandomSearch(build, objective='val_accuracy', max_trials=10,
                    directory='tuner', project_name='toy', overwrite=True)
t.search(x, y, epochs=3, validation_split=0.2)
t.results_summary()
print('Best:', t.get_best_hyperparameters()[0].values)