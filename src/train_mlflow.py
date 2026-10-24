import subprocess, sys, mlflow, yaml, tensorflow as tf
from dotenv import load_dotenv; load_dotenv()
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.train import parser, run

class MLflowLogger(tf.keras.callbacks.Callback):
    def __init__(self): 
        super().__init__(); self.step = 0
    
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=self.step); self.step += 1

p = parser()
p.add_argument('--run_name', default=None)
p.add_argument('--autolog', action='store_true')

a = p.parse_args()
mlflow.set_experiment('visionguard')
git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
data_ver = yaml.safe_load(open('data/raw.dvc'))['outs'][0]['md5']

if a.autolog:
    mlflow.tensorflow.autolog(log_models=False)   # DagsHub rejects MLflow 3 logged-model calls; the .keras file is logged as an artifact below
with mlflow.start_run(run_name=a.run_name):
    mlflow.set_tags({'git_commit': git_commit, 'dvc_data_md5': data_ver, 'seed': a.seed,
                     'logging': 'autolog' if a.autolog else 'manual'})
    if not a.autolog:
        mlflow.log_params(vars(a))
    model, hist = run(a, extra_cb=[] if a.autolog else [MLflowLogger()])

    subprocess.run([sys.executable, '-m', 'src.evaluate', a.model], check=True)

    d = pd.read_csv(f'reports/test_preds_{a.model}.csv')
    mlflow.log_metrics({'test_accuracy': accuracy_score(d.y, d.yhat), 'test_f1': f1_score(d.y, d.yhat), 'test_roc_auc': roc_auc_score(d.y, d.p)})
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))

    
    mlflow.log_artifact(f"reports/cm_{a.model}.png")
    mlflow.log_artifact(f'reports/roc_{a.model}.png')
    mlflow.log_artifact('data/splits/train.csv')
    mlflow.log_artifact('data/splits/val.csv')
    mlflow.log_artifact('data/splits/test.csv')
    mlflow.log_artifact('requirements.txt')
    mlflow.log_artifact(f'models/{a.model}.keras', artifact_path='models')

    if not a.autolog:
        mlflow.tensorflow.log_model(model, name='model')