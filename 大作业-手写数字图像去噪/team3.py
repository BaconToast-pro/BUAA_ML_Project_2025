import os
import random
import time
import base64
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
from io import BytesIO
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed()

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 数据集加载
class NoisyMNISTDatasetBinary(Dataset):
    def __init__(self, root_dir, train):
        if train:
            npz_file = os.path.join(root_dir, "train.npz")
        else:
            npz_file = os.path.join(root_dir, "test.npz")
        data = np.load(npz_file)
        self.noise = data['noise']  # shape: (N, H, W)
        self.train = train
        if self.train:
            self.origin = data['origin']

    def __len__(self):
        return self.noise.shape[0]

    def __getitem__(self, idx):
        # 直接取出后归一化并转 Tensor
        noise = torch.from_numpy(self.noise[idx]).float().unsqueeze(0) / 255.0
        if not self.train:
            return noise
        origin = torch.from_numpy(self.origin[idx]).float().unsqueeze(0) / 255.0
        return noise, origin


# 加载数据集
data_root = "./"  # 根据实际路径修改
train_dataset = NoisyMNISTDatasetBinary(data_root, True)
test_dataset = NoisyMNISTDatasetBinary(data_root, False)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 数据可视化
visialization_num = 8

for i, (noisy, origin) in enumerate(train_loader):
    example_noisy = noisy
    example_origin = origin
    break  # 只显示第一个 batch

plt.figure(figsize=(14, 4))
for i in range(visialization_num):
    plt.subplot(2, visialization_num, i + 1)
    plt.imshow(example_origin[i][0], cmap='gray')
    plt.axis('off')
    plt.subplot(2, visialization_num, visialization_num + i + 1)
    plt.imshow(example_noisy[i][0], cmap='gray')
    plt.axis('off')
plt.suptitle(f"Dataset Examples (Top {visialization_num} Examples)")
plt.tight_layout()
plt.show()


# 模型定义
class DenoisingNet(nn.Module):
    def __init__(self):
        super(DenoisingNet, self).__init__()

        # 编码器 (下采样)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 解码器 (上采样)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 输出层
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码器路径 (带跳跃连接)
        x = self.upconv2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.dec1(x)

        # 输出
        return torch.sigmoid(self.out_conv(x))


# PSNR和SSIM计算函数
def batch_psnr(preds, targets):
    psnr = []
    for p, t in zip(preds, targets):
        p_img = p.cpu().numpy().squeeze()
        t_img = t.cpu().numpy().squeeze()
        psnr.append(compare_psnr(t_img, p_img, data_range=1.0))
    return np.mean(psnr)


def batch_ssim(preds, targets):
    ssim = []
    for p, t in zip(preds, targets):
        p_img = p.cpu().numpy().squeeze()
        t_img = t.cpu().numpy().squeeze()
        ssim.append(compare_ssim(t_img, p_img, data_range=1.0))
    return np.mean(ssim)


# 模型训练
# 超参数设置
learning_rate = 0.001
num_epochs = 50
weight_decay = 1e-5

# 拆分训练集和验证集 (90%训练, 10%验证)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = DenoisingNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

# 记录训练过程
losses = []
train_losses = []
train_psnrs = []
train_ssims = []
val_psnrs = []
val_ssims = []
best_score = 0.0

time_start = time.time()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_psnr = 0.0
    epoch_ssim = 0.0

    # 训练循环
    for noisy, origin in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        noisy = noisy.to(device)
        origin = origin.to(device)

        # 前向传播
        denoised = model(noisy)
        loss = criterion(denoised, origin)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录指标
        epoch_loss += loss.item()
        epoch_psnr += batch_psnr(denoised.detach(), origin)
        epoch_ssim += batch_ssim(denoised.detach(), origin)

    # 计算平均指标
    train_loss = epoch_loss / len(train_loader)
    train_psnr = epoch_psnr / len(train_loader)
    train_ssim = epoch_ssim / len(train_loader)

    # 验证集评估
    model.eval()
    val_psnr = 0.0
    val_ssim = 0.0
    with torch.no_grad():
        for noisy, origin in val_loader:
            noisy = noisy.to(device)
            origin = origin.to(device)
            denoised = model(noisy)
            val_psnr += batch_psnr(denoised, origin)
            val_ssim += batch_ssim(denoised, origin)

    val_psnr /= len(val_loader)
    val_ssim /= len(val_loader)

    # 记录指标
    losses.append(train_loss)
    train_psnrs.append(train_psnr)
    train_ssims.append(train_ssim)
    val_psnrs.append(val_psnr)
    val_ssims.append(val_ssim)

    # 动态调整学习率 (基于PSNR+SSIM加权得分)
    weighted_score = 0.5 * val_psnr + 0.5 * val_ssim * 100  # 归一化到相似量级
    scheduler.step(weighted_score)

    # 保存最佳模型
    if weighted_score > best_score:
        best_score = weighted_score
        torch.save(model.state_dict(), 'best_model.pth')

    # 打印进度
    clear_output(wait=True)
    print(f"Epoch {epoch + 1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Train PSNR: {train_psnr:.2f} | Train SSIM: {train_ssim:.4f} | "
          f"Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}")

# 训练时间
time_elapsed = time.time() - time_start
print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
print(f"Best validation score: {best_score:.4f}")

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 训练过程可视化
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_psnrs, label='Train PSNR')
plt.plot(val_psnrs, label='Val PSNR')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.legend()
plt.title('PSNR during Training')

plt.subplot(1, 2, 2)
plt.plot(train_ssims, label='Train SSIM')
plt.plot(val_ssims, label='Val SSIM')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.legend()
plt.title('SSIM during Training')
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()


# 推理并保存结果
def tensor_to_base64(tensor):
    """将 tensor 转换为 base64 编码的 PNG 图像"""
    # tensor shape: (1, 28, 28), 值范围 [0, 1]
    tensor = tensor.squeeze(0)  # 去掉通道维度，变成 (28, 28)
    tensor = (tensor * 255).clamp(0, 255).byte()  # 转换到 [0, 255] 并转为 uint8

    # 转换为 PIL 图像
    img = Image.fromarray(tensor.cpu().numpy(), mode='L')  # 'L' 表示灰度图

    # 转换为 base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return img_base64


# 在测试集上进行推理
model.eval()
submission_data = []
example_noisy, example_output = None, None  # 用于可视化

with torch.no_grad():
    for batch_idx, noisy in enumerate(tqdm(test_loader, desc="generating submission.csv")):
        noisy = noisy.to(device)
        denoised = model(noisy)

        # 仅记录第一批用于可视化
        if batch_idx == 0:
            example_noisy = noisy.cpu()
            example_output = denoised.cpu()

        # 处理每个样本
        for i in range(denoised.shape[0]):
            sample_id = batch_idx * batch_size + i
            denoised_base64 = tensor_to_base64(denoised[i])
            submission_data.append({
                'id': sample_id,
                'denoised_base64': denoised_base64
            })

# 创建 DataFrame 并保存为 CSV
submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

# 可视化测试结果
num_examples = min(8, len(example_noisy))
plt.figure(figsize=(16, 6))

# 第一行：噪声图像
for i in range(num_examples):
    plt.subplot(2, num_examples, i + 1)
    plt.imshow(example_noisy[i][0], cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.ylabel('Noisy Input', fontsize=14, rotation=90, labelpad=20)

# 第二行：去噪结果
for i in range(num_examples):
    plt.subplot(2, num_examples, num_examples + i + 1)
    plt.imshow(example_output[i][0], cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.ylabel('Denoised Output', fontsize=14, rotation=90, labelpad=20)

plt.suptitle(f"Denoising Results Comparison (First {num_examples} Test Examples)", fontsize=16)
plt.tight_layout()
plt.savefig('denoising_results.png')
plt.show()

# 计算并显示测试集指标（如果有标签）
if hasattr(test_dataset, 'origin'):
    test_origin = []
    for i in range(len(test_dataset)):
        test_origin.append(test_dataset[i][1])
    test_origin = torch.stack(test_origin).to(device)

    model.eval()
    all_denoised = []
    with torch.no_grad():
        for noisy in test_loader:
            noisy = noisy.to(device)
            denoised = model(noisy)
            all_denoised.append(denoised.cpu())

    all_denoised = torch.cat(all_denoised, dim=0)

    test_psnr = batch_psnr(all_denoised, test_origin.cpu())
    test_ssim = batch_ssim(all_denoised, test_origin.cpu())
    print(f"Test PSNR: {test_psnr:.2f}, Test SSIM: {test_ssim:.4f}")

print("All done!")