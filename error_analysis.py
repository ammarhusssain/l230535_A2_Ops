import pandas as pd, matplotlib.pyplot as plt
from PIL import Image
d = pd.read_csv('reports/test_preds_tl.csv'); wrong = d[d.y != d.yhat].head(10)
plt.figure(figsize=(15, 6))
for i, (_, r) in enumerate(wrong.iterrows()):
    plt.subplot(2, 5, i + 1); plt.imshow(Image.open(r.path).convert('RGB'))
    plt.title(f'true={r.y} p={r.p:.2f}', fontsize=9); plt.axis('off')
plt.savefig('reports/errors.png')