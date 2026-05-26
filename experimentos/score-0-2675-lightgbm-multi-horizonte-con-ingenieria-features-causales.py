# == Markdown Cell ==
# <div style="
# background: linear-gradient(135deg,#1b0036,#4b1d8f,#8a2be2);
# padding:36px 34px;
# border-radius:18px;
# color:white;
# font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto;
# border:1px solid rgba(255,255,255,0.08);
# box-shadow:0 10px 28px rgba(0,0,0,0.35);
# ">
# 
# <div style="
# font-size:12px;
# letter-spacing:2px;
# text-transform:uppercase;
# opacity:0.65;
# font-weight:700;
# margin-bottom:10px;">
# 📊 QUANTITATIVE TIME-SERIES FORECASTING
# </div>
# 
# <h1 style="margin:0;font-size:40px;font-weight:800;">
# LightGBM Forecasting Framework
# </h1>
# 
# <div style="margin-top:8px;font-size:15px;opacity:0.9;">
# Advanced Gradient Boosting Pipeline with Sequential Feature Engineering
# </div>
# 
# <div style="margin-top:4px;font-size:13px;opacity:0.7;">
# Lag Features · Rolling Statistics · Hierarchical Signals · Seasonality Encoding · Multi-Seed Ensemble
# </div>
# 
# <!-- BADGES -->
# 
# <div style="margin-top:18px;display:flex;flex-wrap:wrap;gap:12px;max-width:900px;">
# 
# <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);border-radius:22px;padding:7px 16px;font-size:13px;font-weight:600;">
# 📅 Multi Horizon Models
# </span>
# 
# <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);border-radius:22px;padding:7px 16px;font-size:13px;font-weight:600;">
# 🌳 LightGBM Ensemble
# </span>
# 
# <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);border-radius:22px;padding:7px 16px;font-size:13px;font-weight:600;">
# 🔐 Sequential (Leak-Safe)
# </span>
# 
# <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);border-radius:22px;padding:7px 16px;font-size:13px;font-weight:600;">
# ⚖ Weighted RMSE Optimization
# </span>
# 
# <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);border-radius:22px;padding:7px 16px;font-size:13px;font-weight:600;">
# 🧠 Hierarchical Feature Mining
# </span>
# 
# <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);border-radius:22px;padding:7px 16px;font-size:13px;font-weight:600;">
# 📈 Seasonality & Trend Signals
# </span>
# 
# <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);border-radius:22px;padding:7px 16px;font-size:13px;font-weight:600;">
# 🚀 Robust Cross-Validation
# </span>
# 
# <span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);border-radius:22px;padding:7px 16px;font-size:13px;font-weight:600;">
# ⭐ Please upvote if you found this helpful 😊
# </span>
# 
# </div>
# 
# </div>

# == Markdown Cell ==
# ---
# 
# # 🧠 Project Workflow
# 
# This notebook implements a **complete forecasting pipeline** for hierarchical time-series data using **LightGBM ensemble models**.  
# The approach focuses on **leak-safe feature engineering, hierarchical signals, and horizon-aware modeling** to produce stable predictions across multiple forecasting horizons.
# 
# ---
# 
# ## ⚙️ Pipeline Steps
# 
# ### 📦 Data Preparation
# - Load and preprocess large-scale time-series data
# - Optimize memory usage
# - Encode hierarchical identifiers (`code`, `sub_code`, `sub_category`)
# 
# ### 🛠 Feature Engineering
# Features capture both **temporal patterns and cross-series relationships**:
# 
# - Lag features from past observations  
# - Rolling statistics (mean, std, volatility)  
# - Hierarchical aggregates across entity groups  
# - Time and horizon interaction features  
# 
# All features are built using **past data only** to avoid leakage.
# 
# ### 🌳 Model Training
# - **LightGBM gradient boosting models**
# - Separate models trained for each forecasting horizon
# - **Multi-seed ensemble** to improve stability
# - **Time-based validation** to simulate real forecasting
# 
# ### 🔮 Sequential Prediction
# Predictions are generated strictly using information available up to time `t`: ŷ_t = pred(X[1:t])

# == Markdown Cell ==
# 
# This ensures full compliance with the competition rules.
# 
# ---
# 
# ## 📁 Submission
# 
# Final predictions are combined into:
# 
# with two columns:
# 
# - `id`
# - `prediction`
# 
# This file can be submitted directly to the competition leaderboard.

import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import warnings

warnings.filterwarnings('ignore')

TRAIN_PATH = '/kaggle/input/competitions/ts-forecasting/train.parquet'
TEST_PATH  = '/kaggle/input/competitions/ts-forecasting/test.parquet'
VAL_THRESHOLD = 3500
SEEDS = [42, 2024, 777, 1337, 9999]

print("Libraries loaded.")
print(f"Validation split at ts_index = {VAL_THRESHOLD}")
print(f"Ensemble seeds: {SEEDS}")

def weighted_rmse_score(y_target, y_pred, w):
    y_target = np.array(y_target)
    y_pred   = np.array(y_pred)
    w        = np.array(w)

    denom = np.sum(w * (y_target ** 2))
    if denom <= 0:
        return 0.0

    numerator = np.sum(w * ((y_target - y_pred) ** 2))
    ratio = numerator / denom
    return float(np.sqrt(1.0 - np.clip(ratio, 0.0, 1.0)))

print("Metric defined: weighted_rmse_score(y_target, y_pred, w)")

# == Markdown Cell ==
# # 📊 Exploratory Data Analysis (EDA)
# 
# Understanding the structure and behavior of the dataset is essential before building forecasting models.  
# This section explores the data distribution, time structure, and relationships between key variables to guide feature engineering and modeling decisions.
# 
# ---
# 
# ## 🗂 Dataset Overview
# 
# The dataset consists of **multiple hierarchical time series** representing different financial entities.
# 
# Key identifiers:
# 
# | Column | Description |
# |------|-------------|
# | `code` | Primary entity identifier |
# | `sub_code` | Sub-entity within each code |
# | `sub_category` | Higher-level grouping of related entities |
# | `ts_index` | Time index representing sequential observations |
# | `horizon` | Forecast horizon indicating how far ahead the prediction is made |
# 
# The target variable is:
# 
# which represents the **value to be forecasted for each entity and horizon**.

train_raw = pd.read_parquet(TRAIN_PATH)
test_raw  = pd.read_parquet(TEST_PATH)

print(f"Train shape : {train_raw.shape}")
print(f"Test shape  : {test_raw.shape}")
print()
print("Columns:")
print(train_raw.dtypes)
for col in ['code', 'sub_code', 'sub_category', 'horizon']:
    n = train_raw[col].nunique()
    print(f"  {col:20s}: {n:4d} unique values")
ts_train_range = (train_raw.ts_index.min(), train_raw.ts_index.max())
ts_test_range  = (test_raw.ts_index.min(),  test_raw.ts_index.max())

print(f"Train ts_index range : {ts_train_range[0]} → {ts_train_range[1]}")
print(f"Test  ts_index range : {ts_test_range[0]}  → {ts_test_range[1]}")
print()

for h in [1, 3, 10, 25]:
    subset = train_raw[train_raw.horizon == h]
    avg_cs = subset.groupby('ts_index')['sub_code'].count().mean()
    print(f"  Horizon {h:2d} | avg instruments per timestamp: {avg_cs:.1f}")
print(f"{'Horizon':>10} | {'Mean':>10} | {'Std':>10} | {'Skew':>8} | {'Kurt':>8}")
print("-" * 55)
for h in [1, 3, 10, 25]:
    yt = train_raw[train_raw.horizon == h]['y_target']
    print(f"{h:>10} | {yt.mean():>10.5f} | {yt.std():>10.5f} | {yt.skew():>8.3f} | {yt.kurtosis():>8.3f}")

w = train_raw['weight']
print(f"Weight stats:")
print(f"  Mean    : {w.mean():.4f}")
print(f"  Std     : {w.std():.4f}")
print(f"  Min/Max : {w.min():.4f} / {w.max():.4f}")
print(f"  Top 5%  : {w.quantile(0.95):.4f}")

w_sorted = w.sort_values(ascending=False)
top10_frac = w_sorted.iloc[:int(len(w)*0.1)].sum() / w.sum()
print(f"  Top 10% of rows account for {top10_frac*100:.1f}% of total weight mass")

del train_raw, test_raw
gc.collect()

from scipy.stats import spearmanr

train_h1 = pd.read_parquet(TRAIN_PATH).query("horizon == 1")
raw_feats = [c for c in train_h1.columns if c.startswith('feature_')]

print(f"{'Feature':>15} | {'IC (Spearman)':>15} | {' |IC|':>8}")
print("-" * 45)
ics = {}
for f in raw_feats:
    ic, _ = spearmanr(train_h1[f].fillna(0), train_h1['y_target'])
    ics[f] = ic

for f, ic in sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {f:>13} | {ic:>15.5f} | {abs(ic):>8.5f}")

del train_h1
gc.collect()

# == Markdown Cell ==
# ---
# 
# # 🛠️ Feature Engineering Pipeline
# 
# The forecasting performance in this competition depends heavily on robust and leak-safe feature engineering.  
# This pipeline constructs a diverse set of features designed to capture **temporal dynamics, hierarchical relationships, and cross-series interactions** while strictly respecting the competition’s sequential prediction rules.
# 
# ---
# 
# ## 📊 Hierarchical Statistics
# 
# To leverage relationships between entities, we compute **hierarchical aggregate statistics** from the training portion of the dataset.
# 
# These statistics are calculated only using observations where:


# == Markdown Cell ==
# 
# This ensures that **no future information leaks into the model**.
# 
# The following statistics are computed:
# 
# - **Code-level statistics**
#   - mean(`y_target`)
#   - std(`y_target`)
# 
# - **Sub-code statistics**
#   - mean(`y_target`)
#   - std(`y_target`)
# 
# - **Sub-category statistics**
#   - mean(`y_target`)
#   - std(`y_target`)
# 
# These values are stored as **lookup maps** and then merged back into both the training and test datasets.
# 
# This approach allows the model to learn **cross-sectional behavior across related entities** while maintaining strict temporal integrity.
# 
# ---
# 
# ## 🔗 Code × Horizon Interaction Encoding
# 
# To better capture **differences across prediction horizons**, we introduce a cross-feature:


# == Markdown Cell ==
# 
# We apply **frequency encoding** to this interaction feature, which represents how frequently each `(code, horizon)` pair appears in the dataset.
# 
# This encoding helps the model learn:
# 
# - entity-specific forecasting behavior
# - horizon-dependent dynamics
# - cross-series structural patterns
# 
# without introducing target leakage.
# 
# ---
# 
# ## ⚡ Key Properties of the Pipeline
# 
# - ✔ **Strictly sequential (no data leakage)**
# - ✔ **Hierarchical signal extraction**
# - ✔ **Horizon-aware modeling**
# - ✔ **Compatible with gradient boosting models such as LightGBM**
# 
# These engineered features form the foundation of the **LightGBM forecasting models** used in the subsequent sections of the notebook.

print("Computing statistics from training portion only...")

temp = pd.read_parquet(TRAIN_PATH)
train_only = temp[temp.ts_index <= VAL_THRESHOLD].copy()
freq_stats = {}
freq_counts = train_only['sub_code'].value_counts()
freq_stats['sub_code_freq']      = freq_counts.to_dict()
freq_stats['sub_code_freq_rank'] = freq_counts.rank(ascending=False, method='dense').to_dict()

print(f"  sub_code categories tracked: {len(freq_stats['sub_code_freq'])}")
hier_stats = {}
hier_stats['code_mean']    = train_only.groupby('code',         observed=True)['y_target'].mean().to_dict()
hier_stats['sub_code_mean'] = train_only.groupby('sub_code',    observed=True)['y_target'].mean().to_dict()
hier_stats['sub_cat_mean'] = train_only.groupby('sub_category', observed=True)['y_target'].mean().to_dict()
hier_stats['code_std']     = train_only.groupby('code',         observed=True)['y_target'].std().to_dict()
hier_stats['sub_cat_std']  = train_only.groupby('sub_category', observed=True)['y_target'].std().to_dict()

print(f"  Hierarchical maps built: code, sub_code, sub_category (mean + std)")
train_only['code_horizon'] = train_only['code'].astype(str) + '_' + train_only['horizon'].astype(str)
freq_stats['code_horizon_freq'] = train_only['code_horizon'].value_counts().to_dict()

print(f"  code_horizon combinations: {len(freq_stats['code_horizon_freq'])}")

del temp, train_only
gc.collect()
print("Done.")

# == Markdown Cell ==
# ## Step 2: Feature Engineering Function
# 
# 
# | New Group | Features | Rationale |
# |---|---|---|
# | **Seasonality** | `ts_mod_7`, `ts_mod_30`, `ts_mod_90`, `ts_sin`, `ts_cos` | Captures hidden periodic cycles |
# | **Hierarchical** | `code_mean`, `sub_code_mean`, `sub_cat_mean`, stds | Shares signal across the hierarchy |
# | **Lag Ratios** | `lag_ratio_1_7`, `lag_ratio_3_14`, `lag_ratio_1_3` | Trend acceleration/decay detection |
# | **Rolling Lags** | `lag_1`, `lag_3`, `lag_7`, `lag_14` | Direct past-value features |
# | **Time Drift** | `ts_log`, `ts_sq` | Captures non-stationary long-range drift |
# | **Horizon Interaction** | `horizon_sq`, `ts_horizon` | Horizon-time joint effect |
# | **Volatility Strength** | `trend_strength`, `volatility_strength` | Signal-to-noise ratio |
# | **Code×Horizon Freq** | `code_horizon_freq` | Category-specific horizon behaviour |
# | **Cross-Sectional Std** | `code_std`, `sub_cat_std` | Transfer signal across related groups |
# 
# All features remain **100% leak-proof**.

def create_features(df, freq_stats=None, hier_stats=None, horizon=1):
    df = df.copy()
    df = df.sort_values(['code', 'sub_code', 'ts_index']).reset_index(drop=True)

    sub_cat_dummies = pd.get_dummies(df['sub_category'], prefix='subcat', dtype=int)
    df = pd.concat([df, sub_cat_dummies], axis=1)

    if freq_stats is not None:
        df['sub_code_freq']      = df['sub_code'].map(freq_stats['sub_code_freq']).fillna(1)
        df['sub_code_log_freq']  = np.log1p(df['sub_code_freq'])
        df['sub_code_freq_rank'] = df['sub_code'].map(freq_stats['sub_code_freq_rank']).fillna(df['sub_code'].nunique())
        df['sub_code_ts_freq']   = df.groupby(['ts_index', 'sub_code'])['sub_code'].transform('count')
        df['sub_code_rel_freq']  = df['sub_code_ts_freq'] / (df['sub_code_freq'] + 1)

    df['spread_al_am'] = df['feature_al'] - df['feature_am']
    df['ratio_al_am']  = df['feature_al'] / (df['feature_am'] + 1e-7)
    df['spread_cg_by'] = df['feature_cg'] - df['feature_by']
    df['ratio_cg_by']  = df['feature_cg'] / (df['feature_by'] + 1e-7)
    df['spread_s_t']   = df['feature_s'] - df['feature_t']
    df['ratio_s_t']    = df['feature_s'] / (df['feature_t'] + 1e-7)

    ewm_spans = [3, 5] if horizon <= 3 else [5, 7]
    ewm_feats = ['feature_al', 'feature_am', 'spread_al_am']
    for f in ewm_feats:
        if f in df.columns:
            for span in ewm_spans:
                df[f'{f}_ewm{span}'] = df.groupby(['code', 'sub_code'])[f].transform(
                    lambda x: x.ewm(span=span, adjust=False).mean()
                )

    vol_feats = ['feature_al', 'feature_am', 'spread_al_am']
    for f in vol_feats:
        if f in df.columns:
            df[f'{f}_ts_std']   = df.groupby('ts_index')[f].transform('std')
            df[f'{f}_norm_vol'] = df[f] / (df[f'{f}_ts_std'] + 1e-7)

    minmax_feats = ['feature_al', 'feature_am', 'feature_cg']
    for f in minmax_feats:
        if f in df.columns:
            df[f'{f}_ts_min']   = df.groupby('ts_index')[f].transform('min')
            df[f'{f}_ts_max']   = df.groupby('ts_index')[f].transform('max')
            df[f'{f}_ts_range'] = df[f'{f}_ts_max'] - df[f'{f}_ts_min']
            df[f'{f}_dist_min'] = df[f] - df[f'{f}_ts_min']
            df[f'{f}_dist_max'] = df[f'{f}_ts_max'] - df[f]

    mom_feats = ['feature_al', 'spread_al_am']
    for f in mom_feats:
        if f in df.columns:
            df[f'{f}_diff'] = df.groupby(['code', 'sub_code'])[f].diff()
            df[f'{f}_pct']  = df.groupby(['code', 'sub_code'])[f].pct_change().fillna(0)

    z_feats = [
        'feature_al', 'feature_am', 'feature_cg', 'feature_by',
        'feature_s', 'feature_t',
        'spread_al_am', 'spread_cg_by', 'spread_s_t'
    ]
    for f in z_feats:
        if f in df.columns:
            ts_group = df.groupby('ts_index')[f]
            df[f'{f}_z'] = (df[f] - ts_group.transform('mean')) / (ts_group.transform('std') + 1e-7)

    rank_feats = [
        'feature_al', 'feature_am', 'feature_cg', 'feature_by',
        'feature_s', 'feature_t',
        'spread_al_am', 'spread_cg_by', 'spread_s_t'
    ]
    for f in rank_feats:
        if f in df.columns:
            df[f'{f}_rank'] = df.groupby('ts_index')[f].rank(pct=True)

    df['ts_mod_7']  = df['ts_index'] % 7
    df['ts_mod_30'] = df['ts_index'] % 30
    df['ts_mod_90'] = df['ts_index'] % 90
    df['ts_sin']    = np.sin(2 * np.pi * df['ts_index'] / 365)
    df['ts_cos']    = np.cos(2 * np.pi * df['ts_index'] / 365)
    df['ts_sin_100'] = np.sin(2 * np.pi * df['ts_index'] / 100)
    df['ts_cos_100'] = np.cos(2 * np.pi * df['ts_index'] / 100)

    if hier_stats is not None:
        df['code_mean']     = df['code'].map(hier_stats['code_mean']).fillna(0)
        df['sub_code_mean'] = df['sub_code'].map(hier_stats['sub_code_mean']).fillna(0)
        df['sub_cat_mean']  = df['sub_category'].map(hier_stats['sub_cat_mean']).fillna(0)
        df['code_std']      = df['code'].map(hier_stats['code_std']).fillna(0)
        df['sub_cat_std']   = df['sub_category'].map(hier_stats['sub_cat_std']).fillna(0)

    lag_windows = [1, 3, 7, 14]
    for lag in lag_windows:
        df[f'lag_{lag}'] = df.groupby(['code', 'sub_code'])['feature_al'].shift(lag)

    for w in [3, 7, 14]:
        df[f'roll_mean_{w}'] = df.groupby(['code', 'sub_code'])['feature_al'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f'roll_std_{w}'] = df.groupby(['code', 'sub_code'])['feature_al'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        )

    df['lag_ratio_1_7']  = df['lag_1']  / (df['lag_7']  + 1e-6)
    df['lag_ratio_3_14'] = df['lag_3']  / (df['lag_14'] + 1e-6)
    df['lag_ratio_1_3']  = df['lag_1']  / (df['lag_3']  + 1e-6)

    df['ts_log'] = np.log1p(df['ts_index'])
    df['ts_sq']  = df['ts_index'] ** 2

    df['horizon_sq']  = df['horizon'] ** 2
    df['ts_horizon']  = df['ts_index'] * df['horizon']

    if 'roll_mean_3' in df.columns and 'roll_mean_14' in df.columns:
        df['trend_strength']      = df['roll_mean_3'] - df['roll_mean_14']
    if 'roll_std_7' in df.columns and 'roll_mean_7' in df.columns:
        df['volatility_strength'] = df['roll_std_7'] / (df['roll_mean_7'] + 1e-6)

    df['code_horizon'] = df['code'].astype(str) + '_' + df['horizon'].astype(str)
    if freq_stats is not None and 'code_horizon_freq' in freq_stats:
        df['code_horizon_freq'] = df['code_horizon'].map(freq_stats['code_horizon_freq']).fillna(1)

    df = df.fillna(0)
    return df
print("Complete____")

# == Markdown Cell ==
# ---
# 
# # Booster Configuration
# 
# To capture the different predictive dynamics across forecasting windows, we train **separate LightGBM models for each horizon**.  
# Short-term horizons tend to exhibit **higher noise and volatility**, requiring stronger regularization, while longer horizons benefit from **greater model flexibility** to capture broader trends.
# 
# Each model therefore uses **horizon-specific hyperparameters** tailored to its prediction window.
# 
# ---
# 
# ## 🌳 LightGBM Hyperparameters by Horizon
# 
# | Horizon | Num Leaves | Min Child Samples | L2 Regularization | Early Stopping |
# |--------|------------|------------------|------------------|---------------|
# | **1**  | 70 | 250 | 12.0 | 200 |
# | **3**  | 75 | 225 | 11.0 | 190 |
# | **10** | 85 | 180 | 9.0  | 170 |
# | **25** | 90 | 150 | 8.0  | 160 |
# 
# ---
# 
# ## 🎯 Design Rationale
# 
# - **Short Horizons (1–3)**  
#   - Higher noise levels  
#   - Stronger regularization  
#   - Smaller effective tree complexity  
# 
# - **Medium Horizons (10)**  
#   - Balanced signal-to-noise ratio  
#   - Moderate model flexibility  
# 
# - **Long Horizons (25)**  
#   - Larger structural patterns  
#   - Lower regularization to capture broader dynamics  
# 
# ---
# 
# ## ⚡ Additional Training Settings
# 
# Across all horizons, the following configuration is used:
# 
# - **Objective:** RMSE  
# - **Feature Fraction:** Random feature sampling for robustness  
# - **Bagging:** Row subsampling to reduce variance  
# - **Multi-seed training:** Ensemble averaging across seeds to stabilize predictions
# 
# This configuration enables the models to balance **robust generalization, stability, and predictive power** across multiple time horizons.

HORIZON_PARAMS = {
    1: {
        'learning_rate': 0.015, 'n_estimators': 4000, 'num_leaves': 70,
        'min_child_samples': 250, 'lambda_l2': 12.0, 'early_stopping': 200
    },
    3: {
        'learning_rate': 0.015, 'n_estimators': 4000, 'num_leaves': 75,
        'min_child_samples': 225, 'lambda_l2': 11.0, 'early_stopping': 190
    },
    10: {
        'learning_rate': 0.015, 'n_estimators': 4000, 'num_leaves': 85,
        'min_child_samples': 180, 'lambda_l2': 9.0,  'early_stopping': 170
    },
    25: {
        'learning_rate': 0.015, 'n_estimators': 4000, 'num_leaves': 90,
        'min_child_samples': 150, 'lambda_l2': 8.0,  'early_stopping': 160
    }
}

BASE_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'verbosity': -1
}

print("Horizon-specific configs:")
for h, p in HORIZON_PARAMS.items():
    print(f"  h={h:2d} | leaves={p['num_leaves']} | min_child={p['min_child_samples']} | L2={p['lambda_l2']}")

# == Markdown Cell ==
# ---
# 
# # 🚀 Training Loop
# 
# The model training process follows a **time-aware validation strategy** to ensure strict compliance with the competition's sequential prediction rules.
# 
# ---
# 
# ## ⏳ Temporal Validation Split
# 
# To simulate real forecasting conditions, the dataset is split using a **time-based cutoff**:


# == Markdown Cell ==
# 
# - Observations with `ts_index ≤ 3500` are used for **model training**.
# - Observations with `ts_index > 3500` are reserved for **validation**.
# 
# This approach prevents **look-ahead bias** and provides a reliable estimate of out-of-sample performance.
# 
# ---
# 
# ## 🌳 Multi-Seed Ensemble Training
# 
# To improve prediction stability and reduce variance, we train **multiple LightGBM models using different random seeds**.
# 
# For each horizon:
# 
# 1. Train several models with different seeds.
# 2. Generate predictions for each model.
# 3. Average the predictions to produce the final forecast.
# 
# This ensemble strategy helps:
# 
# - Reduce sensitivity to random initialization
# - Improve generalization
# - Stabilize cross-validation scores
# 
# ---
# 
# ## 📊 Feature Importance Tracking
# 
# During training, **feature importance scores are recorded for each horizon-specific model**.  
# These scores are later used to:
# 
# - Identify the most influential features
# - Detect weak or redundant signals
# - Guide feature pruning and refinement
# 
# ---
# 
# ## 🎯 Training Strategy Summary
# 
# The training loop combines several best practices:
# 
# - ✔ **Sequential time-based validation**
# - ✔ **Horizon-specific LightGBM models**
# - ✔ **Multi-seed ensembling for robustness**
# - ✔ **Feature importance analysis for model refinement**
# 
# This design ensures the model remains **leak-safe, stable, and well-generalized** across forecasting horizons.

horizons = [1, 3, 10, 25]
all_test_preds = []
val_store = {'y': [], 'p': [], 'w': []}
feature_importances = {}

for h in horizons:
    print(f"\n{'='*70}")
    print(f"  HORIZON {h}")
    print(f"{'='*70}")

    train = create_features(
        pd.read_parquet(TRAIN_PATH).query(f"horizon == {h}"),
        freq_stats=freq_stats,
        hier_stats=hier_stats,
        horizon=h
    )
    test = create_features(
        pd.read_parquet(TEST_PATH).query(f"horizon == {h}"),
        freq_stats=freq_stats,
        hier_stats=hier_stats,
        horizon=h
    )

    feats = [c for c in train.columns if c not in [
        'id', 'code', 'sub_code', 'sub_category', 'code_horizon',
        'horizon', 'ts_index', 'weight', 'y_target'
    ]]
    print(f"  Features used: {len(feats)}")

    tr_m  = train.ts_index <= VAL_THRESHOLD
    val_m = train.ts_index > VAL_THRESHOLD

    X_tr,  y_tr,  w_tr  = train.loc[tr_m,  feats], train.loc[tr_m,  'y_target'], train.loc[tr_m,  'weight']
    X_val, y_val, w_val = train.loc[val_m, feats], train.loc[val_m, 'y_target'], train.loc[val_m, 'weight']

    print(f"  Train rows: {len(X_tr):,}  |  Val rows: {len(X_val):,}")

    horizon_config = HORIZON_PARAMS[h].copy()
    early_stop = horizon_config.pop('early_stopping')
    params = BASE_PARAMS.copy()
    params.update(horizon_config)

    print(f"  Config → leaves={params['num_leaves']}, L2={params['lambda_l2']}, early_stop={early_stop}")

    h_test_p = np.zeros(len(test))
    h_val_p  = np.zeros(len(y_val))
    h_imp    = np.zeros(len(feats))

    for i, s in enumerate(SEEDS, 1):
        print(f"  Seed {i}/{len(SEEDS)} (seed={s})...", end=' ', flush=True)
        params['random_state'] = s

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_val)], eval_sample_weight=[w_val],
            callbacks=[lgb.early_stopping(early_stop, verbose=False)]
        )

        h_val_p  += model.predict(X_val)       / len(SEEDS)
        h_test_p += model.predict(test[feats]) / len(SEEDS)
        h_imp    += model.feature_importances_  / len(SEEDS)
        print("✓")

    feature_importances[h] = dict(zip(feats, h_imp))

    val_store['y'].extend(y_val.tolist())
    val_store['p'].extend(h_val_p.tolist())
    val_store['w'].extend(w_val.tolist())

    h_score = weighted_rmse_score(y_val, h_val_p, w_val)
    print(f"  → Horizon {h} local score: {h_score:.5f}")

    all_test_preds.append(pd.DataFrame({'id': test['id'], 'prediction': h_test_p}))

    del train, test
    gc.collect()

# == Markdown Cell ==
# ---
# 
# # 📊 Feature Importance Analysis
# 
# Understanding which features contribute most to the model's predictions is essential for improving performance and reducing unnecessary complexity.
# 
# Feature importance analysis helps us:
# 
# - Identify the **most influential predictors**
# - Detect **redundant or weak features**
# - Improve **model efficiency and interpretability**
# - Guide **future feature engineering efforts**
# 
# ---
# 
# ## 🔍 Why Feature Importance Matters
# 
# Gradient boosting models such as **LightGBM** automatically learn complex interactions between features. However, not all engineered features contribute equally to predictive power.
# 
# By inspecting feature importance, we can:
# 
# - Focus on **high-impact signals** such as lagged values and rolling statistics
# - Remove **noisy or redundant features**
# - Improve **training speed and generalization**
# 
# ---
# 
# ## 📈 Key Feature Groups
# 
# Based on preliminary experiments, the most important feature categories typically include:
# 
# - **Lag Features**  
#   Previous target values that capture short-term temporal dependencies.
# 
# - **Rolling Statistics**  
#   Rolling means, standard deviations, and ranges that summarize recent trends and volatility.
# 
# - **Hierarchical Signals**  
#   Aggregated statistics across `code`, `sub_code`, and `sub_category`.
# 
# - **Seasonality & Time Features**  
#   Transformations of `ts_index` capturing periodic patterns.
# 
# - **Horizon Interaction Features**  
#   Features that help the model adapt to different forecasting windows.
# 
# ---
# 
# ## 🎯 Practical Use
# 
# After computing feature importance, we can:
# 
# 1. Rank features by their contribution to the model.
# 2. Remove consistently low-impact features.
# 3. Retrain the model with a **more compact and efficient feature set**.
# 
# This iterative process helps refine the pipeline and often leads to **more stable cross-validation performance**.

import pandas as pd

for h in horizons:
    imp = feature_importances[h]
    imp_df = pd.DataFrame.from_dict(imp, orient='index', columns=['importance'])
    imp_df = imp_df.sort_values('importance', ascending=False)

    print(f"\n── Horizon {h} — Top 30 Features ──")
    print(imp_df.head(30).to_string())

print("\n✅ Review the above. Consider dropping features with importance < 10.")
print("   Keeping ~120–150 strong features is the sweet spot for LightGBM.")

# == Markdown Cell ==
# ---
# 
# # 📬 Final Score & Submission
# 
# After training the LightGBM models and generating predictions for each forecasting horizon, the final step is to assemble the submission file and evaluate the expected performance.
# 
# ---
# 
# ## 📈 Evaluation Metric
# 
# The competition uses a **weighted RMSE-based score**, defined as:


# == Markdown Cell ==
# 
# Key characteristics of the metric:
# 
# - Observations are weighted by the provided `weight` column.
# - Larger target magnitudes influence the score more strongly.
# - Scores range between **0 and 1**, where higher values indicate better performance.
# 
# ---
# 
# ## 🔮 Prediction Generation
# 
# Predictions are produced using the trained **per-horizon LightGBM models**.
# 
# For each horizon:
# 
# 1. The corresponding model is applied to the test data.
# 2. Predictions are generated sequentially using the engineered features.
# 3. Results from different horizons are merged into a single prediction vector.
# 
# ---
# 
# ## 📁 Submission Format
# 
# The final submission must contain two columns:
# 
# | Column | Description |
# |------|-------------|
# | `id` | Unique identifier for each observation |
# | `prediction` | Forecasted value produced by the model |
# 
# 


# == Markdown Cell ==
# 
# ---
# 
# ## 🧠 Final Notes
# 
# - All predictions are generated **sequentially** to comply with the competition rules.
# - No future information is used during feature engineering or inference.
# - The final predictions come from a **multi-seed LightGBM ensemble**, which improves stability and reduces variance.
# 
# ---
# 
# ## ⭐ Thank You
# 
# If you found this notebook helpful or learned something new, please consider giving it an **upvote**.  
# Your support helps improve and share high-quality solutions within the community!

score_local = weighted_rmse_score(
    val_store["y"],
    val_store["p"],
    val_store["w"]
)

print("-" * 70)
print(f"Local Validation Score (Weighted RMSE Metric): {score_local:.6f}")
print("-" * 70)

print("\nValidation details:")
print("- Evaluation performed on holdout window (ts_index > 3500)")
print("- Intended to approximate private leaderboard performance")
print("- Focus on improving this value rather than public LB")

submission_df = pd.concat(all_test_preds, axis=0, ignore_index=True)
submission_df.to_csv("submission.csv", index=False)

print("\nSubmission Summary")
print(f"Total rows : {submission_df.shape[0]:,}")
print(f"Avg value  : {submission_df['prediction'].mean():.6f}")
print(f"Std dev    : {submission_df['prediction'].std():.6f}")
print(f"Lowest     : {submission_df['prediction'].min():.4f}")
print(f"Highest    : {submission_df['prediction'].max():.4f}")

print(submission_df.head())
print("\nFile successfully written: submission.csv")

# == Markdown Cell ==
# ## 🗺️ What to Try Next
# 
# - **Cross-series residual momentum**: Share trend information across related instruments (potential +0.02–0.04)
# - **Feature pruning**: Drop features with importance < 10 to keep ~120–150 total
# - **Hyperparameter search**: Use Optuna on horizon-specific params
# - **Stacking**: Train a second-level model on out-of-fold horizon predictions

