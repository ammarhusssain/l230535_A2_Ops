import sys, mlflow, matplotlib.pyplot as plt, tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

lr = float(sys.argv[1])
(x, y), (xt, yt) = tf.keras.datasets.mnist.load_data(); x, xt = x / 255., xt / 255.
mlflow.set_experiment('toy-mnist')

with mlflow.start_run(run_name=f'lr_{lr}'):
    mlflow.log_params({'lr': lr, 'epochs': 3}); mlflow.set_tag('purpose', 'tool-lab')
    m = tf.keras.Sequential([tf.keras.layers.Input(shape=(28, 28)), tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(10, activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    h = m.fit(x, y, epochs=3, validation_split=0.1)
    for e in range(3):
        mlflow.log_metrics({k: v[e] for k, v in h.history.items()}, step=e)
    plt.plot(h.history['loss']); plt.plot(h.history['val_loss']); plt.savefig('loss.png')
    mlflow.log_artifact('loss.png')
    mlflow.tensorflow.log_model(m, 'model')