import random
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from trainer import train_test


def fix_random_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_and_split_data(user_ids, train_size=19, val_size=3, test_size=4, clean_kp: bool = False):
    """

    """
    random.shuffle(user_ids)

    train_users = user_ids[:train_size]
    val_users = user_ids[train_size:train_size + val_size]
    test_users = user_ids[train_size + val_size:train_size + val_size + test_size]

    print(f"Train users: {sorted(train_users)}")
    print(f"Val users: {sorted(val_users)}")
    print(f"Test users: {sorted(test_users)}")

    def load_user_data(user_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imu, kps, labels = torch.load(f"./data_full/user_{user_id}.pt")
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

    print(f"Train set length: {train_set[1].shape}")  # (N_train, 200, 6)
    print(f"Validation set length: {val_set[1].shape[0]}")  # (N_train, 50, 35)
    print(f"Teset set length: {test_set[1].shape[0]}")  # (N_train,)

    return train_set, val_set, test_set


def get_dataloaders(train_set, val_set, test_set, batch_size=32, data_type='imu'):
    X_imu_train, X_kp_train, y_train = train_set
    X_imu_val, X_kp_val, y_val = val_set
    X_imu_test, X_kp_test, y_test = test_set

    if data_type == 'imu':
        train_loader = DataLoader(TensorDataset(X_imu_train, y_train), batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(3407))
        val_loader = DataLoader(TensorDataset(X_imu_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_imu_test, y_test), batch_size=batch_size, shuffle=False)
    elif data_type == 'kp':
        train_loader = DataLoader(TensorDataset(X_kp_train, y_train), batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(3407))
        val_loader = DataLoader(TensorDataset(X_kp_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_kp_test, y_test), batch_size=batch_size, shuffle=False)
    elif data_type == 'all':
        train_loader = DataLoader(TensorDataset(X_imu_train, X_kp_train, y_train), batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(3407))
        val_loader = DataLoader(TensorDataset(X_imu_val, X_kp_val, y_val), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_imu_test, X_kp_test, y_test), batch_size=batch_size, shuffle=False)

    print(f"Train set size: {len(train_loader)}")
    print(f"Validation set size: {len(val_loader)}")
    print(f"Test set size: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    fix_random_seed(3407)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    seq_len = 200
    n_fft = 64
    hop_length = 8
    in_channels = 9
    patch_size = 16
    stride = 16
    depth = 12
    num_classes = 10
    out_channels = 32
    graph_args = {"layout": "openpose", "strategy": "spatial"}
    edge_importance_weighting = True
    lr = 0.001
    epoch = 300
    save_path = './results'

    user_id_list = list(range(1, 16)) + list(range(17, 28))

    train_set, val_set, test_set = load_and_split_data(user_id_list, clean_kp=True)
    torch.save({
        "train": {
            "imus": train_set[0],
            "kps": train_set[1],
            "labels": train_set[2],
        },
        "val": {
            "imus": val_set[0],
            "kps": val_set[1],
            "labels": val_set[2],
        },
        "test": {
            "imus": test_set[0],
            "kps": test_set[1],
            "labels": test_set[2],
        }
    }, "split_data5.pt")
    train_loader, val_loader, test_loader = get_dataloaders(train_set, val_set, test_set, batch_size=batch_size,
                                                            data_type='all')

    train_test(train_loader, val_loader, seq_len, n_fft, hop_length, device, in_channels, patch_size, stride,
               depth, num_classes, out_channels, graph_args, edge_importance_weighting, lr, epoch, save_path)
