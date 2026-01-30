import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imunet.stft import TFusion
from kpnet.mstgcn import MSTGCN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)

import warnings

warnings.filterwarnings("ignore")


def loss_cal(x, x_aug, T):
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss


def build_model(seq_len, n_fft, hop_length, device, in_channels, patch_size, stride, depth, num_classes, out_channels,
                graph_args, edge_importance_weighting, lr, epoch, imu_chkpoint=None, kp_chkpoint=None):
    imu_model = TFusion(seq_len, n_fft, hop_length, device, in_channels, patch_size, stride, depth, num_classes)
    kp_model = MSTGCN(2, out_channels, num_classes, graph_args, edge_importance_weighting)
    if imu_chkpoint:
        pretrained_dict = imu_chkpoint["model_state_dict"]
        model_dict = imu_model.state_dict()
        model_dict.update(pretrained_dict)
        imu_model.load_state_dict(model_dict)
    if kp_chkpoint:
        pretrained_dict = kp_chkpoint["model_state_dict"]
        model_dict = kp_model.state_dict()
        model_dict.update(pretrained_dict)
        kp_model.load_state_dict(model_dict)

    imu_model.to(device)
    kp_model.to(device)

    # Optimizer
    imu_optimizer = torch.optim.Adam(imu_model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0001)
    imu_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=imu_optimizer, T_max=epoch)

    kp_optimizer = torch.optim.Adam(kp_model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0001)
    kp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=kp_optimizer, T_max=epoch)

    return imu_model, kp_model, imu_optimizer, kp_optimizer, imu_scheduler, kp_scheduler


def model_train(imu_model, kp_model, imu_optimizer, imu_scheduler, kp_optimizer, kp_scheduler, dl_pretrain, epoch, device):
    imu_acc = []
    kp_acc = []
    criterion = nn.CrossEntropyLoss()

    imu_model.train()
    kp_model.train()
    for imu, kp, y in dl_pretrain:
        imu = imu.to(device).float()
        kp = kp.to(device).float()
        y = y.to(device).long()

        imu_optimizer.zero_grad()
        kp_optimizer.zero_grad()

        if epoch <= 100:
            imu_pred = imu_model(imu, 'stg1')
            loss1 = criterion(imu_pred, y)
            loss1.backward()
            imu_optimizer.step()

            kp_pred = kp_model(kp, 'stg1')
            loss2 = criterion(kp_pred, y)
            loss2.backward()
            kp_optimizer.step()

            acc = y.eq(imu_pred.detach().argmax(dim=1)).float().mean()
            imu_acc.append(acc)
            acc = y.eq(kp_pred.detach().argmax(dim=1)).float().mean()
            kp_acc.append(acc)

        elif epoch > 100 and epoch <= 200:
            imu_model.freeze_encoder()
            imu_model.freeze_classifier1()
            imu_model.freeze_classifier2()

            kp_model.freeze_encoder()
            kp_model.freeze_classifier1()
            kp_model.freeze_classifier2()

            imu_proj = imu_model(imu, 'stg2')
            kp_proj = kp_model(kp, 'stg2')
            loss = 0.5 * (loss_cal(imu_proj, kp_proj, 0.4) + loss_cal(kp_proj, imu_proj, 0.4))
            loss.backward()
            imu_optimizer.step()
            kp_optimizer.step()

        else:
            imu_model.freeze_proj_head()
            imu_model.unfreeze_classifier1()
            imu_model.unfreeze_classifier2()

            kp_model.freeze_proj_head()
            kp_model.unfreeze_classifier1()
            kp_model.unfreeze_classifier2()

            imu_pred = imu_model(imu, 'stg3')
            loss1 = criterion(imu_pred, y)
            loss1.backward()
            imu_optimizer.step()

            kp_pred = kp_model(kp, 'stg3')
            loss2 = criterion(kp_pred, y)
            loss2.backward()
            kp_optimizer.step()

            acc = y.eq(imu_pred.detach().argmax(dim=1)).float().mean()
            imu_acc.append(acc)
            acc = y.eq(kp_pred.detach().argmax(dim=1)).float().mean()
            kp_acc.append(acc)

    imu_acc = torch.tensor(imu_acc).mean()
    kp_acc = torch.tensor(kp_acc).mean()
    imu_scheduler.step()
    kp_scheduler.step()

    return imu_acc, kp_acc


def model_test(dl_test, imu_model, kp_model, epoch, device):
    imu_model.eval()
    kp_model.eval()

    # 记录所有数据
    all_labels = []
    imu_all_preds = []
    imu_all_probs = []
    kp_all_preds = []
    kp_all_probs = []

    with torch.no_grad():
        for imu, kp, labels in dl_test:
            imu = imu.to(device).float()
            kp = kp.to(device).float()
            labels = labels.to(device).long()

            if epoch <= 200:
                imu_pred = imu_model(imu, 'stg1')
                kp_pred = kp_model(kp, 'stg1')
            else:
                imu_pred = imu_model(imu, 'stg3')
                kp_pred = kp_model(kp, 'stg3')

            # detach转numpy
            labels_np = labels.cpu().numpy()
            imu_probs_np = torch.softmax(imu_pred, dim=-1).cpu().numpy()
            imu_preds_np = np.argmax(imu_probs_np, axis=-1)
            kp_probs_np = torch.softmax(kp_pred, dim=-1).cpu().numpy()
            kp_preds_np = np.argmax(kp_probs_np, axis=-1)

            all_labels.append(labels_np)
            imu_all_preds.append(imu_preds_np)
            imu_all_probs.append(imu_probs_np)
            kp_all_preds.append(kp_preds_np)
            kp_all_probs.append(kp_probs_np)

    # 合并全部batch
    all_labels = np.concatenate(all_labels)  # shape [N]
    imu_all_preds = np.concatenate(imu_all_preds)  # shape [N]
    imu_all_probs = np.concatenate(imu_all_probs)  # shape [N, num_classes]
    kp_all_preds = np.concatenate(kp_all_preds)
    kp_all_probs = np.concatenate(kp_all_probs)

    num_classes = imu_all_probs.shape[1]

    # acc
    imu_acc = accuracy_score(all_labels, imu_all_preds)
    kp_acc = accuracy_score(all_labels, kp_all_preds)
    # precision, recall, f1
    imu_precision = precision_score(all_labels, imu_all_preds, average="macro" if num_classes > 2 else "binary",
                                    zero_division=0)
    kp_precision = precision_score(all_labels, kp_all_preds, average="macro" if num_classes > 2 else "binary",
                                   zero_division=0)
    imu_recall = recall_score(all_labels, imu_all_preds, average="macro" if num_classes > 2 else "binary",
                              zero_division=0)
    kp_recall = recall_score(all_labels, kp_all_preds, average="macro" if num_classes > 2 else "binary",
                             zero_division=0)
    imu_f1 = f1_score(all_labels, imu_all_preds, average="macro" if num_classes > 2 else "binary", zero_division=0)
    kp_f1 = f1_score(all_labels, kp_all_preds, average="macro" if num_classes > 2 else "binary", zero_division=0)
    # mcc
    imu_mcc = matthews_corrcoef(all_labels, imu_all_preds)
    kp_mcc = matthews_corrcoef(all_labels, kp_all_preds)

    # AUC & AUPRC
    if num_classes == 2:
        imu_auc = roc_auc_score(all_labels, imu_all_probs[:, 1])
        kp_auc = roc_auc_score(all_labels, kp_all_probs[:, 1])
        imu_auprc = average_precision_score(all_labels, imu_all_probs[:, 1])
        kp_auprc = average_precision_score(all_labels, kp_all_probs[:, 1])
    else:
        # one-vs-rest
        imu_auc = roc_auc_score(all_labels, imu_all_probs, multi_class="ovr", average="macro")
        kp_auc = roc_auc_score(all_labels, kp_all_probs, multi_class="ovr", average="macro")
        imu_auprc = average_precision_score(all_labels, imu_all_probs, average="macro")
        kp_auprc = average_precision_score(all_labels, kp_all_probs, average="macro")

    imu_result = {"acc": imu_acc, "precision": imu_precision, "recall": imu_recall, "f1": imu_f1, "auc": imu_auc,
                  "auprc": imu_auprc, "mcc": imu_mcc}
    kp_result = {"acc": kp_acc, "precision": kp_precision, "recall": kp_recall, "f1": kp_f1, "auc": kp_auc,
                 "auprc": kp_auprc, "mcc": kp_mcc}

    return imu_result, kp_result


def train_test(dl_pretrain, dl_test, seq_len, n_fft, hop_length, device, in_channels, patch_size, stride,
               depth, num_classes, out_channels, graph_args, edge_importance_weighting, lr, epoch, save_path):
    save_fold = os.path.join(save_path, f"test/")
    os.makedirs(save_fold, exist_ok=True)
    imu_test_results = []
    imu_best_test_performance = 0
    imu_best_test_result = None
    kp_test_results = []
    kp_best_test_performance = 0
    kp_best_test_result = None
    best_test_clr = None
    imu_best_test_classifier = None
    kp_best_test_classifier = None

    print("*****Training started*****")
    imu_model, kp_model, imu_optimizer, kp_optimizer, imu_scheduler, kp_scheduler = build_model(
        seq_len, n_fft, hop_length, device, in_channels, patch_size, stride, depth, num_classes, out_channels,
        graph_args, edge_importance_weighting, lr, epoch, imu_chkpoint=None, kp_chkpoint=None)

    imu_model.stft_encoder.init_stft_embedder(dl_pretrain)

    for i in range(1, epoch + 1):
        if i == 101:
            imu_model, kp_model, imu_optimizer, kp_optimizer, imu_scheduler, kp_scheduler = build_model(
                seq_len, n_fft, hop_length, device, in_channels, patch_size, stride, depth, num_classes, out_channels,
                graph_args, edge_importance_weighting, lr, epoch, imu_chkpoint=imu_chkpoint, kp_chkpoint=kp_chkpoint)
            imu_model.stft_encoder.init_stft_embedder(dl_pretrain)

        imu_acc, kp_acc = model_train(imu_model, kp_model, imu_optimizer, imu_scheduler, kp_optimizer, kp_scheduler, dl_pretrain, i, device)
        print(f'Epoch: {i}\tIMU: {imu_acc:.3f}\tKP: {kp_acc:.3f}')

        imu_result, kp_result = model_test(dl_test, imu_model, kp_model, i, device)

        if imu_best_test_performance < sum(imu_result.values()) / len(imu_result):
            imu_best_test_performance = sum(imu_result.values()) / len(imu_result)
            imu_best_test_result = imu_result
            imu_best_test_result['epoch'] = i
            imu_test_results.append(imu_best_test_result)
            imu_best_test_clr = imu_model.state_dict()
            imu_chkpoint = {'epoch': i, 'model_state_dict': imu_model.state_dict()}
            torch.save(imu_best_test_clr, os.path.join(save_fold, f"imu_best_test_clr.pt"))
            print("IMU: ", imu_result)

        if kp_best_test_performance < sum(kp_result.values()) / len(kp_result):
            kp_best_test_performance = sum(kp_result.values()) / len(kp_result)
            kp_best_test_result = kp_result
            kp_best_test_result['epoch'] = i
            kp_test_results.append(kp_best_test_result)
            kp_best_test_clr = kp_model.state_dict()
            kp_chkpoint = {'epoch': i, 'model_state_dict': kp_model.state_dict()}
            torch.save(kp_best_test_clr, os.path.join(save_fold, f"kp_best_test_clr.pt"))
            print("KP: ", kp_result)

    print("*****Best results*****")
    print(imu_best_test_result)
    print(kp_best_test_result)

    test_results_df = pd.DataFrame(imu_test_results)
    test_results_df.to_csv(os.path.join(save_fold, 'imu_test_results.csv'), index=False)
    test_results_df = pd.DataFrame(kp_test_results)
    test_results_df.to_csv(os.path.join(save_fold, 'kp_test_results.csv'), index=False)
