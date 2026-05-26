import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
import warnings
import os
import xgboost as xgb 
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
import sys

# ==================== callback ==================
def lgb_dynamic_log(period=50):
    def callback(env):
        if env.iteration % period == 0:
            log_str = f"\r      ⏳ [LGBM] Vòng {env.iteration:4d} | "
            for data_name, metric_name, value, _ in env.evaluation_result_list:
                log_str += f"{data_name}: {value:.5f} | "
            print(log_str, end="", flush=True)
    return callback

class XGBCleanLog(xgb.callback.TrainingCallback):
    def __init__(self, period=50):
        self.period = period
    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period == 0:
            log_str = f"\r      ⏳ [XGB] Vòng {epoch:4d} | "
            for data_name, metrics in evals_log.items():
                for metric_name, loss_vals in metrics.items():
                    log_str += f"{data_name}: {loss_vals[-1]:.5f} | "
            print(log_str, end="", flush=True)
        return False
        
# =============================
# 1️⃣ METRICS & UTILITIES
# =============================
def weighted_rmse_score(y_target, y_pred, w):
    y_target, y_pred, w = np.array(y_target), np.array(y_pred), np.array(w)
    denom = np.sum(w * (y_target ** 2))
    if denom <= 0: return 0.0
    ratio = np.sum(w * ((y_target - y_pred) ** 2)) / denom
    return float(np.sqrt(1.0 - np.clip(ratio, 0.0, 1.0)))

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            if col_type == np.float64:
                df[col] = df[col].astype(np.float32)
            elif col_type == np.int64:
                df[col] = df[col].astype(np.int32)
    return df

# =============================
# 2️⃣ LOAD DATA & FREQUENCY ENCODING AN TOÀN
# =============================
train_path = '/kaggle/input/competitions/ts-forecasting/train.parquet'
test_path = '/kaggle/input/competitions/ts-forecasting/test.parquet'
if not os.path.exists(train_path):
    train_path = 'train.parquet'
    test_path = 'test.parquet'

val_threshold = 3500

print("--- BƯỚC 1: TÍNH TOÁN FREQUENCY STATISTICS TỪ TẬP TRAIN ---")
temp = pd.read_parquet(train_path, columns=['sub_code', 'ts_index'])
train_only = temp[temp.ts_index <= val_threshold]

freq_stats = {}
freq_counts = train_only['sub_code'].value_counts()
freq_stats['sub_code_freq']      = freq_counts.to_dict()
freq_stats['sub_code_freq_rank'] = freq_counts.rank(ascending=False, method='dense').to_dict()

del temp, train_only; gc.collect()
print("Tính toán Frequency Encoding hoàn tất! Cỗ máy sạch 100% Data Leakage.")

# =============================
# HYBRID FEATURE ENGINEERING (BẢN TỐI THƯỢNG - MỞ KHÓA SỨC MẠNH XGBOOST)
# =============================
def build_hybrid_features(data, enc_stats, hz):
    x = data.copy()
    group_keys = ['code', 'sub_code', 'sub_category']
    x = x.sort_values(by=group_keys + ['ts_index']).reset_index(drop=True)
    
    for c in group_keys:
        x[c] = x[c].astype('category')

    sub_cat_dummies = pd.get_dummies(x['sub_category'], prefix='subcat', dtype=np.int8)
    x = pd.concat([x, sub_cat_dummies], axis=1)

    if enc_stats is not None:
        x['sub_code_freq'] = x['sub_code'].map(enc_stats['sub_code_freq']).fillna(1).astype(np.float32)
        x['sub_code_log_freq'] = np.log1p(x['sub_code_freq']).astype(np.float32)
        x['sub_code_freq_rank'] = x['sub_code'].map(enc_stats['sub_code_freq_rank']).fillna(x['sub_code'].nunique()).astype(np.float32)
        
        x['sub_code_ts_freq'] = x.groupby(['ts_index', 'sub_code'])['sub_code'].transform('count').astype(np.float32)
        x['sub_code_rel_freq'] = (x['sub_code_ts_freq'] / (x['sub_code_freq'] + 1)).astype(np.float32)
    
    if 'feature_u' in x.columns and 'feature_s' in x.columns: x['d_u_s'] = (x['feature_u'] - x['feature_s']).astype(np.float32)
    if 'feature_ac' in x.columns and 'feature_ab' in x.columns: x['d_ac_ab'] = (x['feature_ac'] - x['feature_ab']).astype(np.float32)
    if 'feature_m' in x.columns and 'feature_l' in x.columns: x['d_m_l'] = (x['feature_m'] - x['feature_l']).astype(np.float32)
    
    if 'feature_al' in x.columns and 'feature_am' in x.columns: 
        x['d_al_am'] = (x['feature_al'] - x['feature_am']).astype(np.float32)
        x['r_al_am'] = (x['feature_al'] / (x['feature_am'] + 1e-7)).astype(np.float32)
        
    if 'feature_cg' in x.columns and 'feature_by' in x.columns: x['d_cg_by'] = (x['feature_cg'] - x['feature_by']).astype(np.float32)

    norm_cols = [c for c in ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am'] if c in x.columns]
    for col in norm_cols:
        g = x.groupby('ts_index')[col]
        x[col + '_cs'] = (x[col] - g.transform('mean')).astype(np.float32)

    cs_cols = [c + '_cs' for c in norm_cols if c + '_cs' in x.columns]
    if cs_cols:
        x['outlier_count'] = (x[cs_cols].abs() > 3.0).sum(axis=1).astype(np.int8)

    top_features = ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'feature_s']
    def calc_slope(y):
        if len(y) < 2: return np.nan
        return np.polyfit(np.arange(len(y)), y, 1)[0]

    for col in top_features:
        if col not in x.columns: continue
        grp = x.groupby(group_keys, observed=True)[col]
        
        for lag in [1, 3, 5, 10, 25]:
            x[f'{col}_lag{lag}'] = grp.shift(lag).astype(np.float32)
            
        x[f'{col}_diff1'] = (x[col] - x[f'{col}_lag1']).astype(np.float32)
        
        x[f'{col}_ema3'] = grp.transform(lambda s: s.ewm(span=3, adjust=False).mean()).astype(np.float32)
        x[f'{col}_ema10'] = grp.transform(lambda s: s.ewm(span=10, adjust=False).mean()).astype(np.float32)
        x[f'{col}_ema25'] = grp.transform(lambda s: s.ewm(span=25, adjust=False).mean()).astype(np.float32)
        
        x[f'{col}_wavelet_short'] = (x[f'{col}_ema3'] - x[f'{col}_ema10']).astype(np.float32)
        x[f'{col}_wavelet_medium'] = (x[f'{col}_ema10'] - x[f'{col}_ema25']).astype(np.float32)

        regime = np.zeros(len(x), dtype=np.int8)
        regime[(x[f'{col}_wavelet_short'] > 0) & (x[f'{col}_wavelet_medium'] > 0)] = 1
        regime[(x[f'{col}_wavelet_short'] < 0) & (x[f'{col}_wavelet_medium'] < 0)] = -1
        x[f'{col}_trend_regime'] = regime

        roll_max_25 = grp.transform(lambda s: s.rolling(25, min_periods=1).max()).astype(np.float32)
        roll_min_25 = grp.transform(lambda s: s.rolling(25, min_periods=1).min()).astype(np.float32)
        x[f'{col}_stochastic25'] = ((x[col] - roll_min_25) / (roll_max_25 - roll_min_25 + 1e-7)).astype(np.float32)
        del roll_max_25, roll_min_25 
        
        x[f'{col}_roll_std_10'] = grp.transform(lambda s: s.rolling(10, min_periods=1).std()).astype(np.float32)
        roll_std_3 = grp.transform(lambda s: s.rolling(3, min_periods=1).std()).astype(np.float32)
        
        x[f'{col}_reversion_3d'] = ((x[col] - x[f'{col}_ema3']) / (roll_std_3 + 1e-7)).astype(np.float32)

        x[f'{col}_vol_squeeze'] = (roll_std_3 / (x[f'{col}_roll_std_10'] + 1e-7)).astype(np.float32)
        
        mean_subcode = x.groupby(['ts_index', 'sub_code'], observed=True)[col].transform('mean')
        x[f'{col}_vs_subcode'] = (x[col] - mean_subcode).astype(np.float32)
        
        if col in ['feature_al', 'feature_am']:
            x[f'{col}_trend_10'] = grp.transform(lambda s: s.rolling(10, min_periods=2).apply(calc_slope, raw=True)).astype(np.float32)

        if hz == 25:
            x[f'{col}_lag50'] = grp.shift(50).astype(np.float32)
            x[f'{col}_ema50'] = grp.transform(lambda s: s.ewm(span=50, adjust=False).mean()).astype(np.float32)
            x[f'{col}_wavelet_long'] = (x[f'{col}_ema25'] - x[f'{col}_ema50']).astype(np.float32)
            if col in ['feature_al', 'feature_am']:
                x[f'{col}_trend_25'] = grp.transform(lambda s: s.rolling(25, min_periods=2).apply(calc_slope, raw=True)).astype(np.float32)

    if hz in [3, 10]:
        mom_cols = [f'{col}_diff1' for col in top_features] + \
                   [f'{col}_wavelet_short' for col in top_features] + \
                   [f'{col}_wavelet_medium' for col in top_features]
                   
        mom_cols = [c for c in mom_cols if c in x.columns]
        for col in mom_cols:
            g = x.groupby('ts_index')[col]
            x[col + '_cs'] = (x[col] - g.transform('mean')).astype(np.float32)
            
        if hz == 10:
            all_cs_cols = [c for c in x.columns if c.endswith('_cs')]
            if all_cs_cols:
                x['outlier_count'] = (x[all_cs_cols].abs() > 3.0).sum(axis=1).astype(np.int8)

    if hz in [1, 3]:
        if 'feature_cg_diff1' in x.columns:
            x['feature_cg_diff1_rank_cs'] = x.groupby('ts_index')['feature_cg_diff1'].rank(pct=True).astype(np.float32)
            x['cg_momentum_decile'] = (x['feature_cg_diff1_rank_cs'] * 9.99).fillna(-1).astype(np.int8)
            
            if 'feature_al_cs' in x.columns:
                x['cg_al_divergence'] = (x['feature_cg_diff1'] * x['feature_al_cs']).astype(np.float32)
                
            if 'feature_cg_reversion_3d' in x.columns:
                x['cg_reversion_rank_cs'] = x.groupby('ts_index')['feature_cg_reversion_3d'].rank(pct=True).astype(np.float32)
                
    if hz in [1]:
        num_cols = x.select_dtypes(exclude=['category']).columns
        x['missing_info_count'] = x[num_cols].isna().sum(axis=1).astype(np.int8)

    return reduce_mem_usage(x)

# =============================
# 4️⃣ TRAINING LOOP (STACKING RIDGE & DIAGNOSTICS)
# =============================
print("\n--- BƯỚC 2: TRAIN & CHẤM ĐIỂM (ENSEMBLE LGBM + XGB + RIDGE NO SHUFFLE) ---")
IS_RD_MODE = False  
forecast_windows = [1]
if IS_RD_MODE:
    print("⚠️ ĐANG CHẠY Ở CHẾ ĐỘ R&D ")
    forecast_windows = [1, 25]          
    seeds = [42]                        
else:
    print(" ĐANG CHẠY Ở CHẾ ĐỘ FULL (SUBMIT KAGGLE)")
    forecast_windows = [1, 3, 10, 25]
    seeds = [42, 2024, 12345, 666, 7777]
    
test_outputs = []
cv_cache = {'y': [], 'pred': [], 'wt': []}

lgb_cfg = {
    'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.005,
    'n_estimators': 6000, 'num_leaves': 1024, 'max_depth': 14,
    'min_child_samples': 400, 'feature_fraction': 0.6, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 10.0, 'verbosity': -1, 'n_jobs': 4
}

xgb_cfg = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse','learning_rate': 0.015,'n_estimators': 6000, 'max_leaves': 1024,'max_depth': 10, 
    'colsample_bytree': 0.6, 'subsample': 0.8,'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_weight': 5.0, 
    'gamma': 1e-4, 'max_cat_to_onehot': 1, 'grow_policy': 'lossguide','tree_method': 'hist','n_jobs': 4,'early_stopping_rounds': 300
}

for hz in forecast_windows:
    print(f"\n{'='*60}")
    print(f">>> 🚀 XỬ LÝ HORIZON = {hz} <<<")
    print(f"{'='*60}")
    
    raw_train = pd.read_parquet(train_path, filters=[('horizon', '==', hz)])
    raw_test = pd.read_parquet(test_path, filters=[('horizon', '==', hz)])
    
    max_train_ts = raw_train['ts_index'].max()
    tail_train = raw_train[raw_train['ts_index'] > max_train_ts - 50]
    raw_test_with_history = pd.concat([tail_train, raw_test], ignore_index=True)

    tr_df = build_hybrid_features(raw_train, freq_stats, hz)
    te_df_full = build_hybrid_features(raw_test_with_history, freq_stats, hz)
    te_df = te_df_full[te_df_full['ts_index'] > max_train_ts].reset_index(drop=True)

    features = [c for c in tr_df.columns if c not in {'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'weight', 'y_target', 'weight_adjusted'}]

    if hz in [1, 10]:
        max_lag = hz + 25 
        fit_mask = tr_df.ts_index <= (val_threshold - max_lag)
    elif hz in [3]:
        max_lag = 25 
        fit_mask = tr_df.ts_index <= (val_threshold - max_lag)
    else:
        fit_mask = tr_df.ts_index <= val_threshold
        
    val_mask = tr_df.ts_index > val_threshold

    X_fit = tr_df.loc[fit_mask, features]
    y_fit = tr_df.loc[fit_mask, 'y_target'].values.astype(np.float32)
    w_fit_original = tr_df.loc[fit_mask, 'weight'].values.astype(np.float32)
    
    ts_train = tr_df.loc[fit_mask, 'ts_index'].values.astype(np.float32)
    ts_min, ts_max = float(ts_train.min()), float(ts_train.max())
    rec = (ts_train - ts_min) / (ts_max - ts_min + 1e-6)
    w_fit = w_fit_original * (0.5 + 40.0 * rec).astype(np.float32)

    X_hold = tr_df.loc[val_mask, features]
    y_hold = tr_df.loc[val_mask, 'y_target'].values.astype(np.float32)
    w_hold = tr_df.loc[val_mask, 'weight'].values.astype(np.float32)

    lgb_val_pred = np.zeros(len(y_hold))
    xgb_val_pred = np.zeros(len(y_hold))
    lgb_tst_pred = np.zeros(len(te_df))
    xgb_tst_pred = np.zeros(len(te_df))
    
    for seed in seeds:
        print(f"\n [SEED {seed}] is running...")
        
        # --- 🟢 LIGHTGBM ---
        lgb_cfg['random_state'] = seed
        mdl_lgb = lgb.LGBMRegressor(**lgb_cfg)
        mdl_lgb.fit(X_fit, y_fit, sample_weight=w_fit,
                    eval_set=[(X_hold, y_hold)], eval_sample_weight=[w_hold],
                    callbacks=[lgb.early_stopping(100, verbose=False),lgb_dynamic_log(period=50)]
                   )
        print("\r" + " " * 80 + "\r", end="", flush=True)
        lgb_val_pred += mdl_lgb.predict(X_hold) / len(seeds)
        lgb_tst_pred += mdl_lgb.predict(te_df[features]) / len(seeds)
        
        # --- 🔴 XGBOOST ---
        xgb_cfg['random_state'] = seed
        mdl_xgb = xgb.XGBRegressor(**xgb_cfg, enable_categorical=True)
        mdl_xgb.fit(X_fit, y_fit, sample_weight=w_fit,
                    eval_set=[(X_hold, y_hold)], sample_weight_eval_set=[w_hold],
                    verbose=False)
        print("\r" + " " * 80 + "\r", end="", flush=True)
        xgb_val_pred += mdl_xgb.predict(X_hold) / len(seeds)
        xgb_tst_pred += mdl_xgb.predict(te_df[features]) / len(seeds)

    # =======================================================
    # 🧠 META-MODEL: RIDGE STACKING (K-FOLD KHÔNG XÁO TRỘN ĐỂ CHỐNG LEAKAGE)
    # =======================================================
    print("\n   🧠 [STACKING] Đang huấn luyện Ridge Meta-Model (OOF KFold No Shuffle)...")
    meta_X_train = np.column_stack([lgb_val_pred, xgb_val_pred])
    meta_X_test = np.column_stack([lgb_tst_pred, xgb_tst_pred])
    
    # 🌟 PHÁT HIỆN 1: TẮT SHUFFLE ĐỂ BẢO TOÀN TRỤC THỜI GIAN, CHỐNG OVERFIT ẢO 🌟
    kf = KFold(n_splits=5, shuffle=False)
    stack_val_pred = np.zeros_like(y_hold)
    
    for train_idx, val_idx in kf.split(meta_X_train):
        fold_model = Ridge(alpha=1.0, positive=True) 
        fold_model.fit(meta_X_train[train_idx], y_hold[train_idx], sample_weight=w_hold[train_idx])
        stack_val_pred[val_idx] = fold_model.predict(meta_X_train[val_idx])
    
    # Khớp mô hình Ridge cuối cùng để Predict tập Test
    final_ridge = Ridge(alpha=1.0, positive=True)
    final_ridge.fit(meta_X_train, y_hold, sample_weight=w_hold)
    stack_tst_pred = final_ridge.predict(meta_X_test)
    
    # =======================================================
    # 📊 BẢNG ĐIỀU KHIỂN DIAGNOSTICS (CHẨN ĐOÁN LÂM SÀNG)
    # =======================================================
    print(f"\n   📊 [DIAGNOSTICS] PHÂN TÍCH PREDICT VS ACTUAL (H{hz}):")
    print(f"      🔹 Thực tế (y)   | Min: {np.min(y_hold):.4f}  | Max: {np.max(y_hold):.4f}   | Mean: {np.mean(y_hold):.4f} | Std: {np.std(y_hold):.4f}")
    print(f"      🔸 Dự đoán (LGB) | Min: {np.min(lgb_val_pred):.4f}  | Max: {np.max(lgb_val_pred):.4f}   | Mean: {np.mean(lgb_val_pred):.4f} | Std: {np.std(lgb_val_pred):.4f}")
    print(f"      🔸 Dự đoán (XGB) | Min: {np.min(xgb_val_pred):.4f}  | Max: {np.max(xgb_val_pred):.4f}   | Mean: {np.mean(xgb_val_pred):.4f} | Std: {np.std(xgb_val_pred):.4f}")
    print(f"      🏆 Dự đoán(Meta) | Min: {np.min(stack_val_pred):.4f}  | Max: {np.max(stack_val_pred):.4f}   | Mean: {np.mean(stack_val_pred):.4f} | Std: {np.std(stack_val_pred):.4f}")
    
    w1, w2 = final_ridge.coef_
    intercept = final_ridge.intercept_
    total_w = w1 + w2 + 1e-7
    print(f"      ⚖️ Tỷ lệ Stacking| LGBM: {(w1/total_w)*100:.1f}% | XGB: {(w2/total_w)*100:.1f}% | Sai số nền (Intercept): {intercept:.6f}")

    # =======================================================
    # 🏆 CHẤM ĐIỂM
    # =======================================================
    score_lgb = weighted_rmse_score(y_hold, lgb_val_pred, w_hold)
    score_xgb = weighted_rmse_score(y_hold, xgb_val_pred, w_hold)
    score_stack = weighted_rmse_score(y_hold, stack_val_pred, w_hold)
    
    print(f"\n   ✅ Điểm LGBM Độc lập: {score_lgb:.5f}")
    print(f"   ✅ Điểm XGBoost Độc lập: {score_xgb:.5f}")
    print(f"   🔥 ĐIỂM STACKING META (CHÍNH): {score_stack:.5f}")
    
    cv_cache['y'].extend(y_hold.tolist())
    cv_cache['pred'].extend(stack_val_pred.tolist())
    cv_cache['wt'].extend(w_hold.tolist())
    
    test_outputs.append(pd.DataFrame({'id': te_df['id'], 'prediction': stack_tst_pred}))
    del tr_df, te_df_full, te_df, X_fit, X_hold; gc.collect()

print(f"\n{'='*60}")
print(f"🏆 ĐIỂM VALIDATION TỔNG HỢP (LGBM + XGB + RIDGE): {weighted_rmse_score(cv_cache['y'], cv_cache['pred'], cv_cache['wt']):.5f} 🏆")
print(f"{'='*60}")

pd.concat(test_outputs).to_csv('submission_kfold_noshuffle.csv', index=False)

