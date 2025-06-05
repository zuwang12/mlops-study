import torch
from torch.utils.data import DataLoader
from datasets.tabular_dataset import TabularDataset
import pandas as pd

def run_inference(model, X_test, test_uid, cfg):
    test_loader = DataLoader(TabularDataset(X_test), batch_size=cfg['batch_size'])
    model.eval()
    device = next(model.parameters()).device
    preds = []

    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device)
            output = model(xb).squeeze()
            preds.extend(output.cpu().numpy())

    submission = pd.DataFrame({
        'UID': test_uid,
        '채무 불이행 여부': preds
    })
    submission.to_csv(cfg['output_path'], index=False)
    print(f"✅ Saved: {cfg['output_path']}")
