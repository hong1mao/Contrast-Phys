import os

import numpy as np
import torch
import yaml
import random

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pure_dataset import PUREDataset
from model.PhysNetModel import PhysNet
from model.loss import ContrastLoss
from model.IrrelevantPowerRatio import IrrelevantPowerRatio
from torch import optim


def main():
    # 加载配置文件
    with open("./config/train.yaml", "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 设置CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # 创建数据集和数据加载器
    train_dataset = PUREDataset(data_path=config['train']['data']['data_path'],
                                cached_path=config['train']['data']['cached_path'],
                                file_list_path=config['train']['data']['file_list_path'],
                                split_ratio=config['train']['data']['split_ratio'],
                                chunk_length=config['train']['data']['chunk_length'],
                                preprocess=config['train']['data']['preprocess'],
                                re_size=config['train']['data']['re_size'],
                                crop_face=config['train']['data']['crop_face'],
                                larger_box_coef=config['train']['data']['larger_box_coef'],
                                backend=config['train']['data']['backend'],
                                use_face_detection=config['train']['data']['use_face_detection'],
                                label_type=config['train']['data']['label_type'],
                                data_type=config['train']['data']['data_type'])

    val_dataset = PUREDataset(data_path=config['val']['data']['data_path'],
                              cached_path=config['val']['data']['cached_path'],
                              file_list_path=config['val']['data']['file_list_path'],
                              split_ratio=config['val']['data']['split_ratio'],
                              chunk_length=config['val']['data']['chunk_length'],
                              preprocess=config['val']['data']['preprocess'],
                              re_size=config['val']['data']['re_size'],
                              crop_face=config['val']['data']['crop_face'],
                              larger_box_coef=config['val']['data']['larger_box_coef'],
                              backend=config['val']['data']['backend'],
                              use_face_detection=config['val']['data']['use_face_detection'],
                              label_type=config['val']['data']['label_type'],
                              data_type=config['val']['data']['data_type'])

    train_loader = DataLoader(train_dataset,
                              batch_size=config['training']['batch_size'],
                              shuffle=config['training']['shuffle'],
                              num_workers=config['training']['num_workers'],
                              pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config['training']['batch_size'],
                            shuffle=False,
                            num_workers=config['training']['num_workers'],
                            pin_memory=True, drop_last=True)

    # 设置设备
    if torch.cuda.is_available():
        print("✔ Using CUDA")
    else:
        print("❌ Using CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    S = config['model']['S']
    T = config['model']['T']
    in_ch = config['model']['in_ch']
    model = PhysNet(S, in_ch=in_ch).to(device).train()

    # 初始化损失函数
    delta_t = config['loss']['delta_t']
    K = config['loss']['K']
    fs = config['loss']['fs']
    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250)

    # 初始化无关功率比
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # 初始化优化器
    opt = optim.AdamW(model.parameters(), lr=config['training']['optimizer']['lr'])
    # 记录训练和验证损失
    train_losses = []
    iprs = []
    val_losses = []
    # 训练轮数
    total_epoch = config['training']['num_epochs']
    # 保存模型检查点
    save_path = config['checkpoints']['save_path']
    # 训练模型
    for e in range(total_epoch):
        # 训练阶段
        train_loss = 0.0
        ipr = 0.0
        for it in range(np.round(60 / (T / fs)).astype('int')):
            progress_bar = tqdm(train_loader,
                                desc=f"Epoch [{e + 1}/{total_epoch}] "
                                     f"- Sample [{it + 1}/{np.round(60 / (T / fs)).astype('int')}] "
                                     f"- Training",
                                total=len(train_loader))
            for i, batch in enumerate(progress_bar):
                imgs = batch[0].to(device)
                # 前向传播
                model_output = model(imgs)
                rppg = model_output[:, -1]  # ST块平均输出
                # 计算损失
                loss, p_loss, n_loss = loss_func(model_output)
                # 优化
                opt.zero_grad()
                loss.backward()
                opt.step()
                # 计算无关功率比
                ipr += torch.mean(IPR(rppg.clone().detach()))
                # 记录损失
                train_loss += loss.item()

        # 保存模型
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{e + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # 平均损失
        avg_train_loss = train_loss / len(train_loader)
        # 平均无关功率比
        avg_ipr = ipr / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for it in range(np.round(60 / (T / fs)).astype('int')):
                progress_bar = tqdm(val_loader,
                                    desc=f"Epoch [{e + 1}/{total_epoch}] "
                                         f"- Sample [{it + 1}/{np.round(60 / (T / fs)).astype('int')}] "
                                         f"- Validation",
                                    total=len(val_loader))
                for i, batch in enumerate(progress_bar):
                    imgs = batch[0].to(device)
                    labels = batch[1].to(device)
                    model_output = model(imgs)
                    rppg = model_output[:, -1]  # ST块平均输出
                    # 计算损失
                    loss, p_loss, n_loss = loss_func(model_output)
                    # 记录损失
                    val_loss += loss.item()

        # 平均损失
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{e + 1}/{total_epoch}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} "
              f"- IPR: {avg_ipr:.4f}")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        iprs.append(avg_ipr)

    print("Training finished.")

    # 保存训练结果
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()


if __name__ == "__main__":
    main()
