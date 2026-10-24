import sys, numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve
from src.pipeline import make_ds
name = sys.argv[1]
m = tf.keras.models.load_model(f'models/{name}.keras')

def probs(split):
    return m.predict(make_ds(f'data/splits/{split}.csv'), verbose=0).ravel(), pd.read_csv(f'data/splits/{split}.csv').label.values

pv, yv = probs('val')
fpr, tpr, thr = roc_curve(yv, pv)
threshold = float(thr[np.where(fpr <= 0.05)[0][-1]])
print('chosen threshold', threshold)

pt, yt = probs('test'); yhat = (pt >= threshold).astype(int)
res = dict(accuracy=accuracy_score(yt, yhat), precision=precision_score(yt, yhat), recall=recall_score(yt, yhat),
           f1=f1_score(yt, yhat), roc_auc=roc_auc_score(yt, pt), threshold=threshold)
print(pd.Series(res).round(4))

ConfusionMatrixDisplay(confusion_matrix(yt, yhat), display_labels=['real', 'ai']).plot()
plt.savefig(f'reports/cm_{name}.png'); plt.close()

RocCurveDisplay.from_predictions(yt, pt); plt.savefig(f'reports/roc_{name}.png'); plt.close()
pd.DataFrame({'path': pd.read_csv('data/splits/test.csv').path, 'y': yt, 'p': pt, 'yhat': yhat}).to_csv(f'reports/test_preds_{name}.csv', index=False)