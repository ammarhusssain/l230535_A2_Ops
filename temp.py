import matplotlib.pyplot as plt
from src.pipeline import make_ds

x, y = next(iter(make_ds('data/splits/train.csv', training=True)))
plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i + 1); plt.imshow(x[i].numpy().astype('uint8')); plt.title(int(y[i])); plt.axis('off')

plt.savefig('reports/aug_batch.png')