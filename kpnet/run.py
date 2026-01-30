from typing import List, Tuple
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from kpnet.mstgcn import MSTGCN

train_epoch = 100
device = "cuda:1"
batch_size = 128

def fix_random_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_and_split_data(user_ids, train_size=19, val_size=3, test_size=4, clean_kp: bool=False):    
    """
    
    """
    random.shuffle(user_ids)
    
    train_users = user_ids[:train_size]
    val_users = user_ids[train_size:train_size+val_size]
    test_users = user_ids[train_size+val_size:train_size+val_size+test_size]
    
    print(f"Train users: {sorted(train_users)}")
    print(f"Val users: {sorted(val_users)}")
    print(f"Test users: {sorted(test_users)}")
    
    def load_user_data(user_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imu, kps, labels = torch.load(f"../data/user_{user_id}.pt")
        return imu[:, :, :], kps, labels
    
    train_data = [load_user_data(uid) for uid in train_users]
    val_data = [load_user_data(uid) for uid in val_users]
    test_data = [load_user_data(uid) for uid in test_users]
    
    def merge_data(data_list: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imus, kps, labels = zip(*data_list)
        merged_imus, merged_kps, merged_labels = torch.cat(imus), torch.cat(kps), torch.cat(labels)
        if clean_kp:
            mask = ~torch.isnan(merged_kps).any(dim=(1, 2))
            merged_imus, merged_kps, merged_labels = merged_imus[mask], merged_kps[mask], merged_labels[mask]
        N, T, _ = merged_kps.shape
        V, M, C = 17, 1, 2
        merged_kps = merged_kps.reshape(N, T, V, C)  # -> (N, T, 17, 2)
        merged_kps = merged_kps.permute(0, 3, 1, 2)  # -> (N, 2, T, 17, 1)
        merged_kps = merged_kps.unsqueeze(-1)
        # 随机索引打乱
        indices = torch.randperm(merged_imus.size(0))
        
        return merged_imus[indices], merged_kps[indices], merged_labels[indices]

    
    train_set = merge_data(train_data)
    val_set = merge_data(val_data)
    test_set = merge_data(test_data)

    print(f"Train set length: {train_set[1].shape}")      # (N_train, 200, 6)
    print(f"Validation set length: {val_set[1].shape[0]}") # (N_train, 50, 35)
    print(f"Teset set length: {test_set[1].shape[0]}")    # (N_train,)
    
    return train_set, val_set, test_set

def get_dataloaders(train_set, val_set, test_set, batch_size=32, data_type='imu'):
    X_imu_train, X_kp_train, y_train = train_set
    X_imu_val, X_kp_val, y_val = val_set
    X_imu_test, X_kp_test, y_test = test_set

    if data_type == 'imu':
        train_loader = DataLoader(TensorDataset(X_imu_train, y_train), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(3407))
        val_loader   = DataLoader(TensorDataset(X_imu_val, y_val),   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_imu_test, y_test), batch_size=batch_size, shuffle=False)
    elif data_type == 'kp':
        train_loader = DataLoader(TensorDataset(X_kp_train, y_train), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(3407))
        val_loader   = DataLoader(TensorDataset(X_kp_val, y_val),   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_kp_test, y_test), batch_size=batch_size, shuffle=False)

    print(f"Train set size: {len(train_loader)}")
    print(f"Validation set size: {len(val_loader)}")
    print(f"Test set size: {len(test_loader)}")

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', checkpoint_path: str = "checkpoint"):
    model.to(device)

    best_acc = 0

    for epoch in range(train_epoch):
        correct = total = 0
        total_loss = 0
        model.train()
        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).long()
            optimizer.zero_grad()
            output = model(x, 'stg1')
            loss = criterion(output, y)
            
            _, pred = output.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)
            last_loss = loss.item()

            loss.backward()
            optimizer.step()

        acc = correct/total
        print(f"Epoch {epoch}, Train acc {acc:.4f}, loss {total_loss/total}")
        
        # val_acc = validate(model, val_loader)
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     torch.save(model.state_dict(), "stgcn_kp_weight.pth")
        #     print(f"Epoch {epoch}, Val acc {val_acc:.4f}")

def validate(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in dataloader:
            x, y = x.to(device).float(), y.to(device).long()
            output = model(x, 'stg1')

            _, pred = output.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
    model.train()
    return correct / total

def evaluate_model(model, model_name: str, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).float()
            y = y.to(device).long()
            output = model(x)
            _, pred = output.max(1)
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    # 输出分类报告
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=range(0, 10), target_names=[f"Task_{i}" for i in range(1, 11)]))

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(0, 10))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[f"T{i}" for i in range(1, 11)], yticklabels=[f"T{i}" for i in range(1, 11)])
    plt.title(f"Confusion Matrix - {model_name} Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{model_name}-ConfusionMatrix.png")

    return y_true, y_pred

fix_random_seed(3407)

user_id_list = list(range(1, 16)) + list(range(17,28))

train_set, val_set, test_set = load_and_split_data(user_id_list,clean_kp=True)
train_loader, val_loader, test_loader = get_dataloaders(train_set, val_set, test_set, batch_size=batch_size, data_type='kp')

model = MSTGCN(2, 32, 10, graph_args={"layout": "openpose", "strategy": "spatial"}, edge_importance_weighting=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
    
train_model(model, train_loader, test_loader, criterion, optimizer, device=device)
# model.load_state_dict(torch.load("stgcn_kp_weight.pth"))
# test_acc = evaluate_model(model,"st-gcn", test_loader, device=device)