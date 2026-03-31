import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.metrics import r2_score
from torch import nn, optim
from scipy.stats import pearsonr
import joblib
# ------------------------------------------------------------------
# 1. 辅助工具与绘图
# ------------------------------------------------------------------
def train_test_plot(y_train_true, y_train_pred, y_test_true, y_test_pred, write_dir):
    """绘制奇偶校验图（Parity Plot），包含 R 和 R^2 指标"""
    plt.rcParams['axes.linewidth'] = 1.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    y_tr_t, y_tr_p = y_train_true.flatten(), y_train_pred.flatten()
    y_te_t, y_te_p = y_test_true.flatten(), y_test_pred.flatten()

    r_train, _ = pearsonr(y_tr_t, y_tr_p)
    r2_train = r2_score(y_tr_t, y_tr_p)
    r_test, _ = pearsonr(y_te_t, y_te_p)
    r2_test = r2_score(y_te_t, y_te_p)

    datasets = [
        (y_tr_t, y_tr_p, 'Training', r_train, r2_train),
        (y_te_t, y_te_p, 'Test', r_test, r2_test)
    ]
    
    for ax, y_t, y_p, title, r, r2 in zip(axes, *zip(*datasets)):
        ax.scatter(y_t, y_p, alpha=0.5, c='blue', edgecolors='white', 
                   label=f'R = {r:.3f}\n$R^2$ = {r2:.3f}')
        min_val, max_val = min(y_t), max(y_t)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Actual log10(KVRH)', fontsize=12)
        ax.set_ylabel('Predicted log10(KVRH)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(os.path.join(write_dir, 'results_parity_plot_with_PINN.pdf'), dpi=600)
    plt.show()

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

# ------------------------------------------------------------------
# 2. 合成样本生成逻辑 (PINN)
# ------------------------------------------------------------------
def generate_synthetic_samples(X, monotonicity_factors, num_samples_per_feature=100):
    monotonic_features = [i for i, factor in enumerate(monotonicity_factors) if factor != 0]
    if not monotonic_features:
        return np.empty((0, X.shape[1])), np.empty((0, len(monotonicity_factors)))

    num_samples = num_samples_per_feature * len(monotonic_features)
    synthetic_X = np.zeros((num_samples, X.shape[1]))

    for i, feature_idx in enumerate(monotonic_features):
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X[:, feature_idx].reshape(-1, 1))
        samples = kde.sample(num_samples_per_feature).flatten()
        if monotonicity_factors[feature_idx] == 1:
            synthetic_X[i * num_samples_per_feature:(i + 1) * num_samples_per_feature, feature_idx] = np.sort(samples)
        elif monotonicity_factors[feature_idx] == -1:
            synthetic_X[i * num_samples_per_feature:(i + 1) * num_samples_per_feature, feature_idx] = np.sort(samples)[::-1]

    non_monotonic_features = [i for i in range(X.shape[1]) if i not in monotonic_features]
    for feature_idx in non_monotonic_features:
        synthetic_X[:, feature_idx] = np.random.choice(X[:, feature_idx], num_samples)

    synthetic_monotonicity = np.tile(monotonicity_factors, (num_samples, 1))
    return synthetic_X, synthetic_monotonicity

# ------------------------------------------------------------------
# 3. 模型定义与带有物理损失的训练逻辑
# ------------------------------------------------------------------
class NeuralNetwork(object):
    def __init__(self, args, train_loader, val_loader, test_loader, model_save_path, syn_X_train, syn_m_train): 
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.train_loader, self.valid_loader, self.test_loader = train_loader, val_loader, test_loader
        self.model_save_path = model_save_path
        
        self.syn_X_train = torch.tensor(syn_X_train, dtype=torch.float32, device=self.device)
        self.syn_m_train = torch.tensor(syn_m_train, dtype=torch.float32, device=self.device)
        self.physics_batch_size = min(args.get('physics_batch_size', 256), len(self.syn_X_train)) if len(self.syn_X_train) > 0 else 0
        self.physics_interval = max(1, args.get('physics_interval', 4))
        
        input_size = next(iter(train_loader))[0].size(-1)
        layers = []
        in_dim = input_size
        
        for _ in range(args['depth']):
            layers.append(nn.Linear(in_dim, args['width']))
            # [关键修复 1]：将 BatchNorm1d 替换为 LayerNorm，彻底消除假数据排序对真实均值方差的污染
            layers.append(nn.LayerNorm(args['width'])) 
            layers.append(nn.ReLU() if args['activation'] == 'relu' else nn.Tanh())
            layers.append(nn.Dropout(args['dropout']))
            in_dim = args['width']
            
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers).to(self.device)
        self.criterion = nn.MSELoss()

    def physical_inconsistency_loss(self):
        if self.syn_X_train.size(0) == 0: return torch.tensor(0.0, device=self.device)
        if self.physics_batch_size < self.syn_X_train.size(0):
            sample_idx = torch.randint(0, self.syn_X_train.size(0), (self.physics_batch_size,), device=self.device)
            syn_inputs = self.syn_X_train[sample_idx].detach().requires_grad_(True)
            syn_m_train = self.syn_m_train[sample_idx]
        else:
            syn_inputs = self.syn_X_train.detach().requires_grad_(True)
            syn_m_train = self.syn_m_train

        syn_outputs = self.network(syn_inputs)
        
        grad_outputs = torch.ones_like(syn_outputs)
        gradients = torch.autograd.grad(
            outputs=syn_outputs, 
            inputs=syn_inputs,
            grad_outputs=grad_outputs,
            create_graph=True, 
            retain_graph=True
        )[0]

        active_mask = syn_m_train != 0
        if not torch.any(active_mask):
            return torch.tensor(0.0, device=self.device)

        penalties = (1 - syn_m_train * torch.tanh(gradients)) / 2.0
        return penalties.masked_select(active_mask).mean()

    def train(self, is_tuning=False):
        optimizer = optim.Adam(self.network.parameters(), lr=self.args['alpha'], weight_decay=self.args['weight_decay'])
        best_valid_loss = np.inf
        patience, counter = self.args['early_stopping'], 0
        lambda_p = self.args.get('lambda_p', 0.1) 
        
        for epoch in range(1000):
            self.network.train()
            t_loss_meter = AverageMeter()
            
            for step, (x, y) in enumerate(self.train_loader):
                if x.size(0) <= 1: continue 
                x = x.to(self.device, non_blocking=True)
                y = y.unsqueeze(1).to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # [关键修复 2]：将回归损失保留在半精度区加速，物理损失强制在全精度区运算，防止梯度数值下溢
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    regression_loss = self.criterion(self.network(x), y)
                
                if step % self.physics_interval == 0:
                    phys_loss = self.physical_inconsistency_loss()
                else:
                    phys_loss = torch.tensor(0.0, device=self.device)
                    
                total_loss = regression_loss + lambda_p * phys_loss

                self.scaler.scale(total_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                t_loss_meter.update(total_loss.item(), x.size(0))
            
            # 验证
            self.network.eval()
            v_loss = AverageMeter()
            with torch.no_grad():
                for x, y in self.valid_loader:
                    out = self.network(x.to(self.device, non_blocking=True))
                    loss = self.criterion(out, y.unsqueeze(1).to(self.device, non_blocking=True))
                    v_loss.update(loss.item(), x.size(0))
            
            if v_loss.avg < best_valid_loss:
                best_valid_loss = v_loss.avg; counter = 0
                if not is_tuning:
                    torch.save(self.network.state_dict(), os.path.join(self.model_save_path, "best_PINN_model.pth"))
            else:
                counter += 1
                
            if counter >= patience: 
                break
                
            if epoch % 10 == 0: 
                print(f"Epoch {epoch}: Train Loss {t_loss_meter.avg:.4f}, Val Loss {v_loss.avg:.4f}")    
                
        return best_valid_loss

    def predict(self, loader):
        self.network.load_state_dict(torch.load(os.path.join(self.model_save_path, "best_PINN_model.pth"), map_location=self.device))
        self.network.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                out = self.network(x.to(self.device, non_blocking=True))
                preds += out.view(-1).tolist()
                targets += y.tolist()
        return np.array(preds), np.array(targets)

class XYDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = torch.Tensor(X), torch.Tensor(y)
    def __getitem__(self, i): return self.X[i], self.y[i]
    def __len__(self): return len(self.X)

# ------------------------------------------------------------------
# 4. 主程序入口 (加入手动/自动寻优切换开关)
# ------------------------------------------------------------------
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("[*] Waking up GPU and initializing CUDA context...")
        torch.backends.cudnn.benchmark = True
        _ = torch.zeros(1).cuda() 

    split = 'all'
    save_dir = f'./PINN_models_2/{split}/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    column_to_drop = [
        'data_type', 'net', 'KVRH', 'D_func-I-0-all', 'D_func-I-1-all', 'D_func-I-2-all', 'D_func-I-3-all', 
        'D_func-S-0-all', 'D_func-T-0-all', 'D_func-Z-0-all', 'D_func-alpha-0-all', 'D_func-chi-0-all',
        'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all', 'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-T-0-all',
        'D_lc-Z-0-all', 'D_lc-alpha-0-all', 'D_lc-chi-0-all', 'lc-I-0-all', 'D_mc-I-0-all', 
        'D_mc-I-1-all', 'D_mc-I-2-all', 'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-T-0-all', 'D_mc-Z-0-all',
        'D_mc-chi-0-all', 'mc-I-0-all'
    ]

    data_path = f'./feature_files/net_short_symb_combined_data_frame_{split}.csv'
    df_all = pd.read_csv(data_path).set_index('name').dropna()

    df_features = df_all.drop(columns=[c for c in column_to_drop if c in df_all.columns])
    feature_names = df_features.columns.tolist() 
    
    X_all_raw = df_features.to_numpy()
    y_all = np.log10(df_all['KVRH'].to_numpy()) 

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_all_raw, y_all, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train_raw)
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    X_train, X_test = scaler.transform(X_train_raw), scaler.transform(X_test_raw)

    num_features = X_train.shape[1]
    monotonicity_factors = np.zeros(num_features)
    
    target_feature = 'POAV_vol_frac'
    if target_feature in feature_names:
        feature_idx = feature_names.index(target_feature)
        monotonicity_factors[feature_idx] = -1  
    
    # 稍微减少了物理数据的采样基数，在保证物理约束的同时大幅提速
    syn_X_train, syn_m_train = generate_synthetic_samples(X_train, monotonicity_factors, num_samples_per_feature=40)

    # [关键修复 3]：放宽正则化，让模型能充分拟合数据。恢复学习率以加快收敛。
    base_params = {
        "batch_size": 32,
        "depth": 4,
        "width": 256,
        "activation": 'relu',
        "alpha": 0.0005,       # 恢复正常的初始学习率
        "optimizer": 'adam',
        "early_stopping": 100,
        "dropout": 0.05,       # 调低 dropout 防止欠拟合
        "weight_decay": 1e-06,# 减轻 L2 正则化
        "physics_batch_size": 256,
        "physics_interval": 4
    }

    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.15, random_state=336)
    num_workers = min(4, os.cpu_count() or 1)
    loader_kwargs = {
        "batch_size": base_params['batch_size'],
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    t_loader = DataLoader(XYDataset(X_t, y_t), shuffle=True, **loader_kwargs)
    v_loader = DataLoader(XYDataset(X_v, y_v), shuffle=False, **loader_kwargs)
    te_loader = DataLoader(XYDataset(X_test, y_test), shuffle=False, **loader_kwargs)

    ENABLE_TUNING = False  

    if ENABLE_TUNING:
        print(f"\n[*] Starting Bayesian Optimization for Physical Loss Coefficient (lambda_p)...")
        def objective(trial):
            lambda_p = trial.suggest_float("lambda_p", 1e-3, 1.0, log=True) 
            trial_params = base_params.copy()
            trial_params['lambda_p'] = lambda_p
            model_handler = NeuralNetwork(trial_params, t_loader, v_loader, te_loader, save_dir, syn_X_train, syn_m_train)
            return model_handler.train(is_tuning=True)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15) 
        best_lambda_p = study.best_params['lambda_p']
        print(f"\n[*] Optimization finished! Best lambda_p found: {best_lambda_p:.4f}")
        
        final_params = base_params.copy()
        final_params['lambda_p'] = best_lambda_p

    else:
        MANUAL_LAMBDA_P = 0.08  
        
        print(f"\n[*] Manual mode active. Training PINN model with fixed lambda_p = {MANUAL_LAMBDA_P}...")
        final_params = base_params.copy()
        final_params['lambda_p'] = MANUAL_LAMBDA_P

    final_handler = NeuralNetwork(final_params, t_loader, v_loader, te_loader, save_dir, syn_X_train, syn_m_train)
    final_handler.train(is_tuning=False) 

    y_train_pred, y_train_true = final_handler.predict(DataLoader(XYDataset(X_train, y_train), shuffle=False, **loader_kwargs))
    y_test_pred, y_test_true = final_handler.predict(te_loader)

    train_test_plot(y_train_true, y_train_pred, y_test_true, y_test_pred, save_dir)
    print(f"Process complete. Results saved in {save_dir}")