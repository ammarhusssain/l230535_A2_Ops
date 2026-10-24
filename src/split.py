import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/audit.csv')
df = df[df['corrupt'] == False]
df = df[df['duplicate'] == False]

df['label'] = 0
df.loc[df['class'] == 'AiArtData', 'label'] = 1

train, rest = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
val, test = train_test_split(rest, test_size=0.50, stratify=rest['label'], random_state=42)

train[['path', 'label']].to_csv('data/splits/train.csv', index=False)
val[['path', 'label']].to_csv('data/splits/val.csv', index=False)
test[['path', 'label']].to_csv('data/splits/test.csv', index=False)

print('train:', len(train), ' val:', len(val), ' test:', len(test))