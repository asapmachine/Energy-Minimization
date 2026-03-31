import torch
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn, optim
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# ------------------------------------------------------------------
# 1. 辅助工具函数
# ------------------------------------------------------------------
def train_test_plot(y_train_true, y_train_pred, y_test_true, y_test_pred, save_dir):
    """
    绘制真实值与预测值的奇偶校验图 (Parity Plot)，并展示 R 和 R^2 指标。
    """
    # [步骤 1] 计算评估指标
    # 计算 Pearson 相关系数 (R)
    r_train, _ = pearsonr(y_train_true.flatten(), y_train_pred.flatten())
    r_test, _ = pearsonr(y_test_true.flatten(), y_test_pred.flatten())
    
    # 计算决定系数 (R^2)
    r2_train = r2_score(y_train_true.flatten(), y_train_pred.flatten())
    r2_test = r2_score(y_test_true.flatten(), y_test_pred.flatten())

    # [步骤 2] 构建图像
    plt.figure(figsize=(8, 8))
    
    # 绘制训练集散点，并在图例中加入 R 和 R^2
    plt.scatter(y_train_true, y_train_pred, c='blue', alpha=0.5, edgecolor='k',
                label=f'Train (R: {r_train:.3f}, $R^2$: {r2_train:.3f})')
    
    # 绘制测试集散点，并在图例中加入 R 和 R^2
    plt.scatter(y_test_true, y_test_pred, c='red', alpha=0.5, edgecolor='k',
                label=f'Test (R: {r_test:.3f}, $R^2$: {r2_test:.3f})')

    # [步骤 3] 绘制完美的奇偶校验线 (y = x)
    min_val = min(np.min(y_train_true), np.min(y_test_true))
    max_val = max(np.max(y_train_true), np.max(y_test_true))
    
    # 为了让对角线贯穿整个数据范围，稍微向外扩展一点
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin], 
             [min_val - margin, max_val + margin], 
             'k--', lw=2, label='Ideal Fit')

    # [步骤 4] 图像格式化与保存
    plt.xlabel('True log10(KVRH)', fontsize=14)
    plt.ylabel('Predicted log10(KVRH)', fontsize=14)
    plt.title('Parity Plot of KVRH Prediction', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 确保保存路径存在并保存图片
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'parity_plot_with_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

# ------------------------------------------------------------------
# 2. 模型定义与训练逻辑
# ------------------------------------------------------------------
class NeuralNetwork(object):
    def __init__(self, args, train_loader, val_loader, test_loader, model_save_path):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader, self.valid_loader, self.test_loader = train_loader, val_loader, test_loader
        self.model_save_path = model_save_path
        
        # 获取输入维度并构建网络
        input_size = next(iter(train_loader))[0].size(-1)
        layers = []
        in_dim = input_size
        for _ in range(args['depth']):
            layers.append(nn.Linear(in_dim, args['width']))
            # 加入批归一化层，有助于缓解梯度消失/爆炸，提升指标
            layers.append(nn.BatchNorm1d(args['width'])) 
            layers.append(nn.ReLU() if args['activation'] == 'relu' else nn.Tanh())
            layers.append(nn.Dropout(args['dropout']))
            in_dim = args['width']
            
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers).to(self.device)
        self.criterion = nn.MSELoss()

    def train(self):
        optimizer = optim.Adam(self.network.parameters(), lr=self.args['alpha'], weight_decay=self.args['weight_decay'])
        best_valid_loss = np.inf
        patience, counter = self.args['early_stopping'], 0
        
        self.train_losses, self.valid_losses = [], []
        
        print("Starting training...")
        for epoch in range(1000):
            self.network.train()
            t_loss = AverageMeter()
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.unsqueeze(1).to(self.device)
                out = self.network(x)
                loss = self.criterion(out, y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                t_loss.update(loss.item(), x.size(0))
            
            # 验证
            self.network.eval()
            v_loss = AverageMeter()
            with torch.no_grad():
                for x, y in self.valid_loader:
                    out = self.network(x.to(self.device))
                    loss = self.criterion(out, y.unsqueeze(1).to(self.device))
                    v_loss.update(loss.item(), x.size(0))
            
            self.train_losses.append(t_loss.avg); self.valid_losses.append(v_loss.avg)
            
            if v_loss.avg < best_valid_loss:
                best_valid_loss = v_loss.avg; counter = 0
                torch.save(self.network.state_dict(), f"{self.model_save_path}best_ANN_model.pth")
            else:
                counter += 1
            if counter >= patience: break
            if epoch % 50 == 0: print(f"Epoch {epoch}: Train Loss {t_loss.avg:.4f}, Val Loss {v_loss.avg:.4f}")

    def predict(self, loader):
        self.network.load_state_dict(torch.load(f"{self.model_save_path}best_ANN_model.pth"))
        self.network.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                out = self.network(x.to(self.device))
                preds += out.view(-1).tolist()
                targets += y.tolist()
        return np.array(preds), np.array(targets)

class XYDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = torch.Tensor(X), torch.Tensor(y)
    def __getitem__(self, i): return self.X[i], self.y[i]
    def __len__(self): return len(self.X)

# ------------------------------------------------------------------
# 3. 主程序入口
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 3. 主程序入口 (修改后)
# ------------------------------------------------------------------
if __name__ == "__main__":
    split = 'all'
    save_dir = f'./ANN_models_2/{split}/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # 精确的特征删除列表（恢复 172 维的关键）
    column_to_drop = [
        'data_type', 'net', 'KVRH', 'D_func-I-0-all', 'D_func-I-1-all', 'D_func-I-2-all', 'D_func-I-3-all',
        'D_func-S-0-all', 'D_func-T-0-all', 'D_func-Z-0-all', 'D_func-alpha-0-all', 'D_func-chi-0-all',
        'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all', 'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-T-0-all',
        'D_lc-Z-0-all', 'D_lc-alpha-0-all', 'D_lc-chi-0-all', 'lc-I-0-all', 'D_mc-I-0-all',
        'D_mc-I-1-all', 'D_mc-I-2-all', 'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-T-0-all', 'D_mc-Z-0-all',
        'D_mc-chi-0-all', 'mc-I-0-all'
    ]

    # [修改点 1] 加载整合后的单一数据文件
    # 假设您的整合文件存放在 './feature_files/' 目录下
    data_path = f'./feature_files/net_short_symb_combined_data_frame_{split}.csv'
    df_all = pd.read_csv(data_path).set_index('name').dropna()

    # [修改点 2] 提取全局特征矩阵 (X) 和 目标变量 (y)
    X_all_raw = df_all.drop(columns=[c for c in column_to_drop if c in df_all.columns]).to_numpy()
    y_all = np.log10(df_all['KVRH'].to_numpy()) # 论文通常使用 log10

    # [修改点 3] 首次划分：将全局数据划分为 训练集 和 测试集
    # 这里设置 test_size=0.2 表示 20% 的数据作为测试集，您可以根据需求调整
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all_raw, y_all, test_size=0.2, random_state=42
    )

    # 归一化并打印维度验证 (仅使用训练集 fit，防止数据泄露)
    scaler = StandardScaler().fit(X_train_raw)
    X_train, X_test = scaler.transform(X_train_raw), scaler.transform(X_test_raw)
    print(f"Verified Dimensions: Train {X_train.shape}, Test {X_test.shape}")

    # 手动注入最优参数（跳过 hyperopt）
    best_params = {
        "batch_size": 32,
        "depth": 4,
        "width": 256,
        "activation": 'relu',
        "alpha": 0.0001,
        "optimizer": 'adam',
        "early_stopping": 100,
        "dropout": 0.3,
        "weight_decay": 1e-05
    }

    # 第二次划分：从训练集中划分出 验证集 (用于 Early Stopping)
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.15, random_state=336)
    
    t_loader = DataLoader(XYDataset(X_t, y_t), batch_size=best_params['batch_size'], shuffle=True)
    v_loader = DataLoader(XYDataset(X_v, y_v), batch_size=best_params['batch_size'])
    te_loader = DataLoader(XYDataset(X_test, y_test), batch_size=best_params['batch_size'])

    # 运行模型
    model_handler = NeuralNetwork(best_params, t_loader, v_loader, te_loader, save_dir)
    model_handler.train()

    # 获取结果并绘图
    y_train_pred, y_train_true = model_handler.predict(DataLoader(XYDataset(X_train, y_train), batch_size=64))
    y_test_pred, y_test_true = model_handler.predict(te_loader)

    train_test_plot(y_train_true, y_train_pred, y_test_true, y_test_pred, save_dir)
    print(f"Process complete. Results saved in {save_dir}")