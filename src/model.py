import tensorflow as tf
from tensorflow.keras import layers
IMG = 224

def baseline_cnn(dropout=0.3):
    inp = layers.Input((IMG, IMG, 3))
    x = layers.Rescaling(1/255.)(inp)
    for f in [32, 64, 128]:
        x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inp, out, name='baseline_cnn')

def mobilenet(dropout=0.3):
    base = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG, IMG, 3))
    base.trainable = False
    inp = layers.Input((IMG, IMG, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=False)      
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inp, out, name='mobilenetv2_tl'), base

def unfreeze_last_block(base, first_layer='block_16_expand'):
    base.trainable = True
    trainable = False
    for l in base.layers:
        if l.name == first_layer:
            trainable = True
        l.trainable = trainable
    return [l.name for l in base.layers if l.trainable]

def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                            tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])