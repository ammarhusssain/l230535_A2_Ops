import sys
import numpy as np
import mlflow
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

model = mlflow.tensorflow.load_model('models:/visionguard@champion')

img = Image.open(sys.argv[1]).convert('RGB').resize((224, 224))
x = np.array(img, dtype='float32')[None]

p = float(model.predict(x, verbose=0)[0][0])
THRESHOLD = 0.86                            
print(f'Image: {sys.argv[1]}')
print(f'P(AI-generated) = {p:.3f}  ->  {"AI-generated" if p >= THRESHOLD else "Real"}')

model.save('models/champion.keras')         
print('Saved models/champion.keras')