import argparse, json, tensorflow as tf
from src.pipeline import make_ds
from src.model import baseline_cnn, mobilenet, unfreeze_last_block, compile_model

def callbacks(ckpt):
    return [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(ckpt, monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7)]

def run(a, extra_cb=()):
    tf.keras.utils.set_random_seed(a.seed)
    tf.config.experimental.enable_op_determinism()
    tr = make_ds('data/splits/train.csv', a.batch, training=not a.no_aug, seed=a.seed)
    va = make_ds('data/splits/val.csv', a.batch)
    hist = {}
    if a.model == 'cnn':
        m = baseline_cnn(a.dropout); compile_model(m, a.lr)
        h = m.fit(tr, validation_data=va, epochs=a.epochs, callbacks=callbacks('models/cnn.keras') + list(extra_cb))
        hist['stage1'] = h.history
    else:
        m, base = mobilenet(a.dropout); compile_model(m, a.lr)
        print('Stage 1 trainable:', [w.name for w in m.trainable_weights])
        h1 = m.fit(tr, validation_data=va, epochs=a.epochs, callbacks=callbacks('models/tl.keras') + list(extra_cb))
        print('Stage 2 trainable layers:', unfreeze_last_block(base, a.unfreeze_from))
        compile_model(m, a.lr / 100)                          # much smaller LR for fine-tuning
        h2 = m.fit(tr, validation_data=va, epochs=a.epochs, callbacks=callbacks('models/tl.keras') + list(extra_cb))
        hist['stage1'], hist['stage2'] = h1.history, h2.history
    json.dump(hist, open(f'reports/history_{a.model}.json', 'w'))
    return m, hist

def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='tl', choices=['cnn', 'tl'])
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_aug', action='store_true')
    p.add_argument('--unfreeze_from', default='block_16_expand')
    return p

if __name__ == '__main__':
    run(parser().parse_args())