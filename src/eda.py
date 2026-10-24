import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

df = pd.read_csv('data/audit.csv')
df = df[df['corrupt'] == False]
classes = df['class'].unique()

plt.figure(figsize=(14, 5))
i = 1
for c in classes:
    samples = df[df['class'] == c].sample(6, random_state=1)
    for path in samples['path']:
        plt.subplot(len(classes), 6, i)                         
        plt.imshow(Image.open(path).convert('RGB'))
        plt.title(c, fontsize=8)
        plt.axis('off')
        i += 1
plt.tight_layout()
plt.savefig('reports/sample_grid.png', dpi=120)
plt.close()

for col in ['width', 'height', 'size_kb']:
    sns.histplot(data=df, x=col, hue='class', bins=40)
    plt.title(f'{col} by class')
    plt.savefig(f'reports/hist_{col}.png', dpi=120)
    plt.close()

print('Saved 4 figures in reports/')