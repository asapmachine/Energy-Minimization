import os
# [修复 3]：解决 Intel/LLVM OpenMP 冲突警告，防止程序卡死
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

# ==========================================
# 1. 基础配置与路径
# ==========================================
split = 'all'
base_dir = f'./PINN_models_2/{split}/'

data_path = f'./feature_files/net_short_symb_combined_data_frame_{split}.csv'
model_path = os.path.join(base_dir, 'best_PINN_model.pth')
scaler_path = os.path.join(base_dir, 'scaler.pkl')
save_dir = base_dir

column_to_drop = [
    'data_type', 'net', 'KVRH', 'D_func-I-0-all', 'D_func-I-1-all', 'D_func-I-2-all', 'D_func-I-3-all', 
    'D_func-S-0-all', 'D_func-T-0-all', 'D_func-Z-0-all', 'D_func-alpha-0-all', 'D_func-chi-0-all',
    'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all', 'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-T-0-all',
    'D_lc-Z-0-all', 'D_lc-alpha-0-all', 'D_lc-chi-0-all', 'lc-I-0-all', 'D_mc-I-0-all', 
    'D_mc-I-1-all', 'D_mc-I-2-all', 'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-T-0-all', 'D_mc-Z-0-all',
    'D_mc-chi-0-all', 'mc-I-0-all'
]

# ==========================================
# 2. 数据加载与预处理
# ==========================================
print("[*] Loading and preprocessing data...")
df_all = pd.read_csv(data_path).set_index('name')

target_col = 'KVRH'
cols_to_drop = [c for c in column_to_drop if c in df_all.columns and c != target_col]

df_temp = df_all.drop(columns=cols_to_drop)
df_clean = df_temp.dropna()

y_all = np.log10(df_clean[target_col].to_numpy())
df_features = df_clean.drop(columns=[target_col])
feature_names = df_features.columns.tolist()
X_all_raw = df_features.to_numpy()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_all_raw, y_all, test_size=0.2, random_state=42)

print(f"[*] Loading pre-fitted scaler from {scaler_path}...")
scaler = joblib.load(scaler_path)

X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ==========================================
# 3. 网络骨架重建与权重加载
# ==========================================
print("[*] Reconstructing Neural Network architecture...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X_train.shape[1]

depth = 4
width = 256
dropout_rate = 0.05

layers = []
in_dim = input_size
for _ in range(depth):
    layers.append(nn.Linear(in_dim, width))
    layers.append(nn.LayerNorm(width))  
    layers.append(nn.ReLU())            
    layers.append(nn.Dropout(dropout_rate))
    in_dim = width
layers.append(nn.Linear(in_dim, 1))

model = nn.Sequential(*layers).to(device)

print(f"[*] Loading weights from {model_path}...")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() 

# ==========================================
# 4. SHAP 计算
# ==========================================
print("[*] Preparing SHAP KernelExplainer...")

def predict_fn(x_numpy):
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
    with torch.no_grad(): 
        return model(x_tensor).cpu().numpy().flatten()

num_samples_to_explain = 200 
test_X_sample = X_test[:num_samples_to_explain]

# [修复 1]：先随机抽样 1000 个，再做 KMeans，极大提升 Preparing 速度
print("[*] Generating background summary (optimized)...")
X_train_sampled = shap.sample(X_train, min(1000, X_train.shape[0]))
background_summary = shap.kmeans(X_train_sampled, 20)

explainer = shap.KernelExplainer(predict_fn, background_summary)

print(f"[*] Calculating SHAP values for {len(test_X_sample)} samples...")
shap_values_list = []

for i in tqdm(range(len(test_X_sample)), desc="SHAP Calculation Progress"):
    single_shap = explainer.shap_values(test_X_sample[i:i+1], silent=True)
    shap_values_list.append(single_shap)

shap_values_array = np.vstack(shap_values_list)

# ==========================================
# 5. 可视化与保存
# ==========================================
print("[*] Generating SHAP Summary Plot...")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 12

# [修复 2]：将 shap.plots.summary_plot 改回通用的 shap.summary_plot
shap.summary_plot(
    shap_values_array, 
    test_X_sample, 
    feature_names=feature_names, 
    plot_type="violin",   
    plot_size=(12, 10), 
    show=False
)

plt.tight_layout()
shap_save_path = os.path.join(save_dir, 'results_shap_summary_PINN_violin.pdf')
plt.savefig(shap_save_path, dpi=600, bbox_inches='tight')
print(f"[*] Process complete. SHAP Plot saved to {shap_save_path}")
plt.show()