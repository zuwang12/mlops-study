import yaml
import pandas as pd
from train import train_model
from predict import run_inference
from utils.preprocessing import preprocess

def load_config(path='./configs/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    cfg = load_config()

    train_df = pd.read_csv(cfg['train_path'])
    test_df = pd.read_csv(cfg['test_path'])

    model, X_full = train_model(train_df, test_df, cfg)
    _, _, X_test = preprocess(train_df, test_df)
    run_inference(model, X_test, test_df['UID'], cfg)
