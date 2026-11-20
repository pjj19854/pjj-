"""
矿物分类项目 · 深度学习版本（神经网络 / Conv1D 卷积神经网络）

这份代码补的是课程 day0904 视频里讲、但 daily study 里没留存的那部分：
    · 全连接网络（MLP）：输入 13 -> 32 -> 64 -> 4
    · 一维卷积网络（Conv1D）：Conv1d 1->16->32->64 + 平均池化 + 全连接 64->4

与前面三份代码（清洗数据.py / fill_data.py / 模型最后训练.py）完全同一套路：
    读填好的 xlsx -> 转 Tensor -> 训练 -> 打印 loss / acc / 分类报告

用法：把下面 MODEL 改成 'mlp' 或 'conv1d' 再运行即可。
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report


warnings.filterwarnings('ignore')
torch.manual_seed(42)   # 固定初始权重，保证每次跑出来的结果一致

# ====================================================================#
# 0. 配置
# ====================================================================#
MODEL = 'mlp'      # 'mlp' = 全连接网络；'conv1d' = 一维卷积网络

# 数据目录：优先用「本文件同级」的 temp_data（跟代码放一起，自成一套）；
# 找不到就退回课程原目录 E:\daily study\机器学习\day08+day09\temp_data
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_TEMP = os.path.join(_HERE, 'temp_data')
_COURSE_TEMP = r'E:\daily study\机器学习\day08+day09\temp_data'
TEMP_DIR = _LOCAL_TEMP if os.path.isdir(_LOCAL_TEMP) else _COURSE_TEMP

# 想换填充法，只需把方括号里的名字改成 [均值] / [众数] / [线性回归] / [随机森林] 等
FILL_NAME = '中位数'
train_path = os.path.join(TEMP_DIR, f'训练数据集[{FILL_NAME}].xlsx')
test_path = os.path.join(TEMP_DIR, f'测试数据集[{FILL_NAME}].xlsx')

# ====================================================================#
# 1. 读取数据（首列是标签，其余 13 列是特征）
# ====================================================================#
train_data = pd.read_excel(train_path)
test_data = pd.read_excel(test_path)

X_train = train_data.drop(['矿物类型'], axis=1)
y_train = train_data['矿物类型']
X_test = test_data.drop(['矿物类型'], axis=1)
y_test = test_data['矿物类型']
print(f'训练集: {X_train.shape}，测试集: {X_test.shape}')

# 2. DataFrame 转 Tensor
#    torch 只认 numpy 数组，所以要 .values；特征用 float32
#    标签必须用 long：CrossEntropyLoss 要求目标是类别下标（整数），float32 会直接报错
x_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
x_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# ====================================================================#
# 3. 网络结构
# ====================================================================#
class MLP(nn.Module):
    """全连接网络：13 -> 32 -> 64 -> 4"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 32)   # 输入层13 -> 隐藏层1: 32个神经元
        self.fc2 = nn.Linear(32, 64)   # 隐藏层2: 64个神经元
        self.fc3 = nn.Linear(64, 4)    # 输出层: A/B/C/D 四类

    def forward(self, x):             # 前向传播，名字不能改
        x = torch.sigmoid(self.fc1(x))   # N×13 -> N×32，激活不改变维度
        x = torch.sigmoid(self.fc2(x))   # N×32 -> N×64
        x = self.fc3(x)                  # N×64 -> N×4，每行4个类别得分
        return x


class Conv1DNet(nn.Module):
    """一维卷积网络：Conv1d 1->16->32->64 + 长度维平均池化 + 全连接 64->4"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, 4)     # 压成 64 个值后接全连接，输出 4 类

    def forward(self, x):             # 前向传播，名字不能改
        # 这里用 ReLU 而不是课堂原版的 Sigmoid：三层 Sigmoid 卷积再对长度维求平均，
        # 各样本输出会被压得几乎一样，实测 loss 卡在 1.0551 不下降、acc 只有 0.5623
        # （等于全预测成 A 类）。想复现课堂原版，把下面三个 relu 改回 sigmoid 即可。
        x = x.unsqueeze(1)                 # (N,13) -> (N,1,13)：batch / channel / length
        x = torch.relu(self.conv1(x))      # (N,1,13) -> (N,16,13)
        x = torch.relu(self.conv2(x))      # (N,16,13) -> (N,32,13)
        x = torch.relu(self.conv3(x))      # (N,32,13) -> (N,64,13)
        x = x.mean(dim=2)                 # 对长度维求平均做池化 -> (N,64)
        x = self.fc(x)                    # (N,64) -> (N,4)
        return x


# ====================================================================#
# 4. 按 MODEL 选网络、定超参数
#    lr 取 0.1：实测 0.01 在这两个结构上都几乎不更新
#    （MLP 跑满 15000 轮只有 0.81；Conv1D loss 完全不动）
# ====================================================================#
if MODEL == 'mlp':
    model = MLP()
    EPOCHS = 15000       # 约 30 秒，实测测试集 acc≈0.96
elif MODEL == 'conv1d':
    model = Conv1DNet()
    EPOCHS = 3000        # Conv1D 比全连接慢很多；换 ReLU 后 3000 轮约 25 秒、acc≈0.88
else:
    raise ValueError("MODEL 只能是 'mlp' 或 'conv1d'")

# 5. 损失函数与优化器：分类用交叉熵，优化用 SGD
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ====================================================================#
# 6. 训练循环
# ====================================================================#
ACCS = []           # 保存每次评估的测试集准确率，方便画曲线

print(f'========== 开始训练 {"全连接神经网络" if MODEL == "mlp" else "卷积神经网络"} ==========\n')
start_time = time.time()
for epoch in range(EPOCHS):
    # 前向传播：model(x) 等价于 model.forward(x)
    output = model(x_train_tensor)
    loss = criterion(output, y_train_tensor)

    # 反向更新固定三步：梯度清零 -> 反向传播 -> 更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每 100 轮评估一次，省时间；loss 是张量，打印要用 .item()
    if epoch % 100 == 0:
        with torch.no_grad():   # 评估不需要求梯度，关掉省显存
            train_pred = model(x_train_tensor).argmax(dim=1)
            train_acc = (train_pred == y_train_tensor).float().mean().item()
            test_pred = model(x_test_tensor).argmax(dim=1)
            test_acc = (test_pred == y_test_tensor).float().mean().item()
        ACCS.append(test_acc)
        print(f'Epoch {epoch:>5}/{EPOCHS}  loss={loss.item():.4f}  '
              f'自测acc={train_acc:.4f}  测试acc={test_acc:.4f}')

# 7. 最终评估：取每行最大得分的下标作为预测类别
with torch.no_grad():
    y_pred = model(x_test_tensor).argmax(dim=1).numpy()
test_acc = accuracy_score(y_test, y_pred)

print(f'\n✅ 训练完成，耗时 {time.time() - start_time:.2f} 秒')
print(f'最终测试集准确率: {test_acc:.4f}')
print('分类报告:')
print(classification_report(y_test, y_pred, digits=6))
print('=' * 50 + '\n')
print('深度学习训练完成！')
