import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

rows = []

for img_path in Path('data/raw').glob('*/*'):
    row = {
        'path': str(img_path),
        'class': img_path.parent.name,
        'size_kb': img_path.stat().st_size / 1024,
        'sha256': hashlib.sha256(img_path.read_bytes()).hexdigest(),
        'corrupt': False,
    }
    try:
        img = Image.open(img_path)
        img.load()                                     
        row['width'] = img.width
        row['height'] = img.height
        row['mode'] = img.mode                         
        row['format'] = img.format                     
    except Exception:
        row['corrupt'] = True
    rows.append(row)

df = pd.DataFrame(rows)
df['duplicate'] = df.duplicated('sha256')
df.to_csv('data/audit.csv', index=False)

print(df['class'].value_counts())
print(df.groupby('class')[['width', 'height', 'size_kb']].agg(['min', 'median', 'max']))
print('Corrupt files:', df['corrupt'].sum())
print('Duplicates:', df['duplicate'].sum())