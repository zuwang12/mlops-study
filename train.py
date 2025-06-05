import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets.tabular_dataset import TabularDataset
from models.JK_model import JKModel
from utils.preprocessing import preprocess
from sklearn.metrics import roc_auc_score

def train_model(train_df, test_df, cfg):
    X, y, _ = preprocess(train_df, test_df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=cfg['random_seed'])

    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=cfg['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JKModel(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = torch.nn.BCELoss()

    for epoch in range(cfg['epochs']):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                output = model(xb).squeeze()
                preds.extend(output.cpu().numpy())
                targets.extend(yb.numpy())
        auc = roc_auc_score(targets, preds)
        print(f"[Epoch {epoch+1}] AUC: {auc:.4f}")

    return model, X
