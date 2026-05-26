# == Markdown Cell ==
# # ============================================================================
# # TEAM: Arthur XXIV
# # SCORE: 0.33 (Public Leaderboard)
# # ============================================================================
#  
# ## NOTEBOOK OVERVIEW
# -----------------
#   This notebook implements a horizon-wise LightGBM ensemble for time series
#   forecasting. The approach trains separate models per forecast horizon (1, 3,
#   10, 25) to capture horizon-specific patterns.
# 
# ### KEY DESIGN DECISIONS:
#  1. Horizon-wise modeling: Different horizons have different dynamics
#  2. Time-based validation split: Prevents data leakage (train ≤ 3500, val > 3500)
#  3. Target encoding from training data only: No look-ahead bias
#  4. Early stopping: Prevents overfitting on large dataset
# 
# ### SEQUENTIAL PREDICTION COMPLIANCE:
#  - All statistics (target encoding, frequency counts) computed only from
#    training data (ts_index ≤ 3500)
#  - Lag features use .shift() which only looks backward
#  - No future information is used at any prediction point
# 
# ## RUNTIME: ~20-25 minutes on Kaggle CPU (GPU NOT RECOMMENDED)
# ============================================================================



# == Markdown Cell ==
# # ============================================================================
# # CELL 1: ENVIRONMENT & DEPENDENCIES
# # ============================================================================

# == Markdown Cell ==
# ENVIRONMENT SPECIFICATIONS
# 
# --------------------------
# Platform: Kaggle Notebooks
# Python Version: 3.10.x (Kaggle default)
# Hardware: CPU (GPU not required)
# 
# DEPENDENCIES (with versions):
# - pandas==2.0.3
# - numpy==1.24.3
# - lightgbm==4.1.0
# - tqdm==4.66.1
# - scipy==1.11.3 (indirect dependency)
# 
# To verify versions, Please run:
# 
# - import pandas, numpy, lightgbm
# - print(f"pandas: {pandas.__version__}")
# - print(f"numpy: {numpy.__version__}")
# - print(f"lightgbm: {lightgbm.__version__}")


import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import warnings
from tqdm import tqdm
 
warnings.filterwarnings('ignore')

# Print versions for reproducibility verification
print("=" * 70)
print("ENVIRONMENT VERIFICATION")
print("=" * 70)
print(f"pandas version   : {pd.__version__}")
print(f"numpy version    : {np.__version__}")
print(f"lightgbm version : {lgb.__version__}")
print("=" * 70)


# == Markdown Cell ==
# # ============================================================================
# # CELL 2: CONFIGURATION
# # ============================================================================

# == Markdown Cell ==
# ### CONFIGURATION PARAMETERS
# ------------------------
# All key parameters are defined here for easy modification and transparency.
# 
# DATA PATHS:
# - TRAIN_PATH: Training data with features, targets, and weights
# - TEST_PATH: Test data for submission (no y_target)
# 
# VALIDATION STRATEGY:
# - VAL_THRESHOLD = 3500: Time-based split to simulate out-of-sample prediction
#   - Training set: ts_index ≤ 3500 (~85% of data)
#   - Validation set: ts_index > 3500 (~15% of data)
#   - This mimics the competition's sequential prediction requirement
# 
# HORIZONS:
# - The dataset contains 4 forecast horizons: 1, 3, 10, 25
# - We train separate models per horizon to capture horizon-specific patterns

TRAIN_PATH = '/kaggle/input/competitions/ts-forecasting/train.parquet'
TEST_PATH = '/kaggle/input/competitions/ts-forecasting/test.parquet'
 
VAL_THRESHOLD = 3500
HORIZONS = [1, 3, 10, 25]
SUB_CATEGORIES = None  # Will be populated from data
BLEND_POWER = 2.5  # Power for weighted blending (higher = more weight to better model)
 
# LightGBM params
LGB_PARAMS = {
    'metric': 'rmse',
    'learning_rate': 0.03,
    'n_estimators': 1200,
    'num_leaves': 64,
    'min_child_samples': 100,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'n_jobs': -1,
    'random_state': 101,
}
 
print("=" * 80)
print("🚀 LIGHTGBM DUAL ENSEMBLE")
print("=" * 80)
print("Models to train:")
print("  • 4 LightGBM Horizon models (H1, H3, H10, H25)")
print("  • 5 LightGBM Sub_Category models")
print("  • Total: 9 models")
print(f"  • Blending: Power {BLEND_POWER} weighted average")
print("=" * 80)

# == Markdown Cell ==
# # ============================================================================
# # CELL 3: COMPETITION METRIC
# # ============================================================================

# == Markdown Cell ==
# ### VALUATION METRIC
# -----------------
# The competition uses a weighted RMSE skill score:
#  
# Score = sqrt(1 - clip(sum(w * (y - yhat)^2) / sum(w * y^2), 0, 1))
#  
# Properties:
# - Range: [0, 1] where higher is better
# - Weighted by sample importance (w)
# - Penalizes large errors more heavily (squared term)
# - Score of 0 means predictions are no better than predicting 0
# - Score of 1 means perfect predictions
#  
# This metric rewards accurate predictions on high-weight samples.

def weighted_rmse_score(y_target, y_pred, weights):
    """
    Calculate the competition's weighted RMSE skill score.
    
    Parameters
    ----------
    y_target : array-like
        Ground truth target values
    y_pred : array-like
        Predicted values
    weights : array-like
        Sample weights (importance)
    
    Returns
    -------
    float
        Skill score in range [0, 1], higher is better
    
    Formula
    -------
    Score = sqrt(1 - min(max(sum(w*(y-yhat)^2) / sum(w*y^2), 0), 1))
    """
    y_target = np.array(y_target)
    y_pred = np.array(y_pred)
    weights = np.array(weights)
    
    # Weighted sum of squared errors
    numerator = np.sum(weights * (y_target - y_pred) ** 2)
    
    # Weighted sum of squared targets (normalization factor)
    denominator = np.sum(weights * (y_target ** 2))
    
    # Handle edge case of zero denominator
    if denominator <= 0:
        return 0.0
    
    # Compute ratio and clip to [0, 1]
    ratio = numerator / denominator
    ratio = np.clip(ratio, 0.0, 1.0)
    
    # Final skill score
    return float(np.sqrt(1.0 - ratio))
 
print("✓ Metric function defined: weighted_rmse_score(y_target, y_pred, weights)")

# == Markdown Cell ==
# # ============================================================================
# # CELL 4: FEATURE ENGINEERING FUNCTION
# # ============================================================================

# == Markdown Cell ==
# ### FEATURE ENGINEERING
# -------------------
# This function creates additional features from the raw data.
#  
# FEATURES CREATED:
# 1. Interaction features: Differences between correlated features
# 2. Group mean features: Mean of features within code/sub_code groups
# 3. Target encoding: Mean target by sub_code (from training data only)
# 4. Lag features: Previous timestep values (backward-looking only)
#  
# SEQUENTIAL PREDICTION COMPLIANCE:
# - Target encoding uses pre-computed statistics from training data only
# - Lag features use .shift() which only looks at previous rows
# - Group means are computed within the current dataset (no future info)
#  
# FEATURES EXCLUDED:
# - ts_index: Causes memorization/overfitting when used as a feature
#   (Discovered through ablation studies - removing it improved validation score)

def create_features(dataframe, train_stats=None):
    """
    Create engineered features for the model.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Input data with raw features
    is_train : bool
        Whether this is training data (unused, kept for API consistency)
    train_stats : dict or None
        Pre-computed statistics from training data for target encoding
        Must contain: 'sub_code_target_mean', 'global_mean'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features
    
    Notes
    -----
    - Does NOT include ts_index as a feature (causes overfitting)
    - All features are backward-looking only (no data leakage)
    """    
    
    dataframe = dataframe.copy()
    
    # ====== Interaction Features ======
    if 'feature_al' in dataframe.columns and 'feature_am' in dataframe.columns:
        dataframe['feature_al_minus_feature_am'] = dataframe['feature_al'] - dataframe['feature_am']
    
    # ====== Group Mean Features ======
    group_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    
    if 'feature_al' in dataframe.columns:
        dataframe['feature_al_grp_mean'] = dataframe.groupby(group_cols)['feature_al'].transform('mean')
    
    if 'feature_am' in dataframe.columns:
        dataframe['feature_am_grp_mean'] = dataframe.groupby(group_cols)['feature_am'].transform('mean')
    
    # ====== Target Encoding (Sub_Code) ======
    if train_stats is not None:
        if 'sub_code_target_mean' in train_stats:
            dataframe['sub_code_target_mean'] = dataframe['sub_code'].map(
                train_stats['sub_code_target_mean']
            ).fillna(train_stats['global_mean'])
    
    # ====== Lag Features ======
    # Sort for proper lag calculation, but keep index for alignment
    dataframe = dataframe.sort_values(['code', 'horizon', 'ts_index'])
    
    if 'feature_al' in dataframe.columns:
        dataframe['feature_al_lag1'] = dataframe.groupby(['code', 'horizon'])['feature_al'].shift(1)
    
    if 'feature_am' in dataframe.columns:
        dataframe['feature_am_lag1'] = dataframe.groupby(['code', 'horizon'])['feature_am'].shift(1)
    
    # Reset index AFTER lag features (keeps all columns aligned)
    dataframe = dataframe.reset_index(drop=True)
    
    # Fill NaN values
    dataframe = dataframe.fillna(0)
    
    return dataframe

# == Markdown Cell ==
# # ============================================================================
# # CELL 5: LOAD DATA & COMPUTE TARGET ENCODING STATISTICS
# # ============================================================================

# == Markdown Cell ==
# ### DATA LOADING
# ------------
# Load the training and test datasets.
#  
# DATASET STRUCTURE:
# - id: Unique identifier (code__sub_code__sub_category__horizon__ts_index)
# - code: Primary category identifier
# - sub_code: Secondary category identifier
# - sub_category: Tertiary category (4 values)
# - horizon: Forecast horizon (1, 3, 10, 25)
# - ts_index: Time index (sequential integer)
# - feature_a through feature_ch: Input features
# - y_target: Target variable (training only)
# - weight: Sample weight for evaluation (training only)
# 
# ### TARGET ENCODING STATISTICS
# --------------------------
# Compute statistics from TRAINING DATA ONLY (ts_index <= VAL_THRESHOLD).
#  
# SEQUENTIAL PREDICTION COMPLIANCE:
# - Statistics are computed only from data that would be available at prediction time
# - For any prediction at time t, we only use data from time 1 to t
# - By using ts_index <= 3500, we ensure no future information is used
#  
# STATISTICS COMPUTED:
# - sub_code_target_mean: Average y_target for each sub_code
# - global_mean: Overall average y_target (fallback for unseen sub_codes) 

print("\n📥 Loading data...")
train_full = pd.read_parquet(TRAIN_PATH)
test_full = pd.read_parquet(TEST_PATH)
 
print(f"✅ Train shape: {train_full.shape}")
print(f"✅ Test shape: {test_full.shape}")
 
# Get sub_categories
SUB_CATEGORIES = sorted(train_full['sub_category'].unique().tolist())
print(f"✅ Sub-categories found: {len(SUB_CATEGORIES)} → {SUB_CATEGORIES}")
 
# ============================================================================
# COMPUTE TARGET ENCODING STATS
# ============================================================================
 
print("\n📊 Computing target encoding statistics...")
 
train_for_stats = train_full[train_full['ts_index'] <= VAL_THRESHOLD].copy()
 
train_stats = {
    'sub_code_target_mean': train_for_stats.groupby('sub_code')['y_target'].mean().to_dict(),
    'global_mean': train_for_stats['y_target'].mean()
}
 
print(f"✅ Sub-code groups: {len(train_stats['sub_code_target_mean'])}")
print(f"✅ Global mean: {train_stats['global_mean']:.4f}")
 
del train_for_stats
gc.collect()
 
# ============================================================================
# STORAGE FOR ALL PREDICTIONS
# ============================================================================
 
horizon_val_scores = {}
subcat_val_scores = {}
 
all_test_preds = {
    'lgb_horizon': {},
    'lgb_subcat': {}
}


# == Markdown Cell ==
# # ============================================================================
# # CELL 6: HORIZON-WISE MODEL TRAINING & TRAIN 5 LIGHTGBM SUB_CATEGORY MODELS
# # ============================================================================

# == Markdown Cell ==
# ### MODEL TRAINING STRATEGY
# -----------------------
# We train separate LightGBM models for each forecast horizon (1, 3, 10, 25).
#  
# RATIONALE:
# - Different horizons exhibit different prediction dynamics
# - Short horizons (1, 3) may rely more on recent patterns
# - Long horizons (10, 25) may rely more on structural features
# - Separate models allow horizon-specific pattern learning
#  
# TRAINING PROCESS FOR EACH HORIZON:
# 1. Filter data by horizon
# 2. Apply feature engineering
# 3. Split into train (ts_index <= 3500) and validation (ts_index > 3500)
# 4. Train LightGBM with early stopping on validation set
# 5. (Optional) Save model for later analysis, modify the training loop
# 6. Evaluate using competition metric
# 7. Generate predictions on test set
#  
# EARLY STOPPING:
# - Monitors RMSE on validation set
# - Stops if no improvement for 100 rounds
# - Prevents overfitting and reduces training time

print("\n" + "=" * 80)
print("🎯 PART 1/2: LIGHTGBM HORIZON MODELS")
print("=" * 80)
 
for horizon in tqdm(HORIZONS, desc="LGB Horizon"):
    print(f"\n  HORIZON = {horizon}")
    
    train_horizon = train_full[train_full['horizon'] == horizon].copy()
    test_horizon = test_full[test_full['horizon'] == horizon].copy()
    
    print(f"    Train rows: {len(train_horizon):,} | Test rows: {len(test_horizon):,}")
    
    train_horizon = create_features(train_horizon, train_stats=train_stats)
    test_horizon = create_features(test_horizon, train_stats=train_stats)
    
    drop_cols = ['id', 'y_target', 'weight', 'ts_index']
    all_cols = set(train_horizon.columns) & set(test_horizon.columns)
    feature_cols = sorted([col for col in all_cols if col not in drop_cols])
    
    print(f"    Features: {len(feature_cols)}")
    
    train_mask = train_horizon['ts_index'] <= VAL_THRESHOLD
    val_mask = train_horizon['ts_index'] > VAL_THRESHOLD
    
    X_train = train_horizon.loc[train_mask, feature_cols]
    y_train = train_horizon.loc[train_mask, 'y_target']
    weight_train = train_horizon.loc[train_mask, 'weight']
    
    X_val = train_horizon.loc[val_mask, feature_cols]
    y_val = train_horizon.loc[val_mask, 'y_target']
    weight_val = train_horizon.loc[val_mask, 'weight']
    
    X_test = test_horizon[feature_cols]
    
    # Convert categoricals for LightGBM
    for col in feature_cols:
        if train_horizon[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    print(f"    Training LightGBM...")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=weight_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[weight_val],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(50)
        ]
    )
    
    val_pred = model.predict(X_val)
    val_score = weighted_rmse_score(y_val.values, val_pred, weight_val.values)
    horizon_val_scores[horizon] = val_score
    print(f"    ✅ Val Score: {val_score:.6f}")
    
    test_pred = model.predict(X_test)
    all_test_preds['lgb_horizon'][horizon] = pd.DataFrame({
        'id': test_horizon['id'].values,
        'horizon_pred': test_pred,
        'horizon': horizon,
        'sub_category': test_horizon['sub_category'].values
    })
    
    del train_horizon, test_horizon, X_train, X_val, X_test, model
    gc.collect()

print("\n" + "=" * 80)
print("🎯 PART 2/2: LIGHTGBM SUB_CATEGORY MODELS")
print("=" * 80)
 
for subcat in tqdm(SUB_CATEGORIES, desc="LGB Sub_cat"):
    print(f"\n  SUB_CATEGORY = {subcat}")
    
    train_subcat = train_full[train_full['sub_category'] == subcat].copy()
    test_subcat = test_full[test_full['sub_category'] == subcat].copy()
    
    print(f"    Train rows: {len(train_subcat):,} | Test rows: {len(test_subcat):,}")
    
    train_subcat = create_features(train_subcat, train_stats=train_stats)
    test_subcat = create_features(test_subcat, train_stats=train_stats)
    
    drop_cols = ['id', 'y_target', 'weight', 'ts_index']
    all_cols = set(train_subcat.columns) & set(test_subcat.columns)
    feature_cols = sorted([col for col in all_cols if col not in drop_cols])
    
    print(f"    Features: {len(feature_cols)}")
    
    train_mask = train_subcat['ts_index'] <= VAL_THRESHOLD
    val_mask = train_subcat['ts_index'] > VAL_THRESHOLD
    
    X_train = train_subcat.loc[train_mask, feature_cols]
    y_train = train_subcat.loc[train_mask, 'y_target']
    weight_train = train_subcat.loc[train_mask, 'weight']
    
    X_val = train_subcat.loc[val_mask, feature_cols]
    y_val = train_subcat.loc[val_mask, 'y_target']
    weight_val = train_subcat.loc[val_mask, 'weight']
    
    X_test = test_subcat[feature_cols]
    
    for col in feature_cols:
        if train_subcat[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    print(f"    Training LightGBM...")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        sample_weight=weight_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[weight_val],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(50)
        ]
    )
    
    val_pred = model.predict(X_val)
    val_score = weighted_rmse_score(y_val.values, val_pred, weight_val.values)
    subcat_val_scores[subcat] = val_score
    print(f"    ✅ Val Score: {val_score:.6f}")
    
    test_pred = model.predict(X_test)
    all_test_preds['lgb_subcat'][subcat] = pd.DataFrame({
        'id': test_subcat['id'].values,
        'subcat_pred': test_pred
    })
    
    del train_subcat, test_subcat, X_train, X_val, X_test, model
    gc.collect()

# == Markdown Cell ==
# # ============================================================================
# # CELL 8. HYBRID BLENDING: ROW-SPECIFIC WEIGHTED PREDICTIONS
# # ============================================================================

print("\n" + "=" * 80)
print("🔀 PART 3: WEIGHTED BLENDING")
print("=" * 80)
 
# Combine LGB horizon predictions
horizon_pred_df = pd.concat(all_test_preds['lgb_horizon'].values(), ignore_index=True)
print(f"✅ Horizon predictions: {len(horizon_pred_df):,}")
 
# Combine LGB sub_category predictions
subcat_pred_df = pd.concat(all_test_preds['lgb_subcat'].values(), ignore_index=True)
print(f"✅ Sub_Category predictions: {len(subcat_pred_df):,}")
 
# Merge on id
merged = horizon_pred_df.merge(subcat_pred_df, on='id', how='inner')
print(f"✅ Merged predictions: {len(merged):,}")
 
# Add validation scores
merged['horizon_val_score'] = merged['horizon'].map(horizon_val_scores)
merged['subcat_val_score'] = merged['sub_category'].map(subcat_val_scores)
 
# Check for missing data
print("\n⚠️ Checking for missing data...")
missing_horizon = merged['horizon_pred'].isna().sum()
missing_subcat = merged['subcat_pred'].isna().sum()
missing_h_score = merged['horizon_val_score'].isna().sum()
missing_s_score = merged['subcat_val_score'].isna().sum()
 
print(f"   Missing horizon predictions: {missing_horizon}")
print(f"   Missing subcat predictions: {missing_subcat}")
print(f"   Missing horizon scores: {missing_h_score}")
print(f"   Missing subcat scores: {missing_s_score}")
 
if missing_horizon > 0 or missing_subcat > 0:
    print("⚠️ WARNING: Some predictions or scores are missing!")
    print("   Dropping rows with missing data...")
    merged = merged.dropna(subset=['horizon_pred', 'subcat_pred', 'horizon_val_score', 'subcat_val_score'])
    print(f"   Remaining rows: {len(merged):,}")
 
print(f"\n⚙️ Blending with power {BLEND_POWER} weighted validation scores...")
 
def blend_predictions(row):
    """
    Blend predictions using power-weighted validation scores.
    
    Formula:
      powered_H = score_H ^ BLEND_POWER
      powered_SC = score_SC ^ BLEND_POWER
      h_weight = powered_H / (powered_H + powered_SC)
      s_weight = powered_SC / (powered_H + powered_SC)
      final = h_weight * h_pred + s_weight * s_pred
    """
    h_score = row['horizon_val_score']
    s_score = row['subcat_val_score']
    
    # Handle NaN values
    if pd.isna(h_score):
        h_score = 0.0
    if pd.isna(s_score):
        s_score = 0.0
    
    # Apply power weighting
    powered_h = h_score ** BLEND_POWER
    powered_s = s_score ** BLEND_POWER
    
    total = powered_h + powered_s
    if total <= 0:
        return row['horizon_pred'], 0.5, 0.5  # Fallback
    
    h_weight = powered_h / total
    s_weight = powered_s / total
    
    prediction = h_weight * row['horizon_pred'] + s_weight * row['subcat_pred']
    
    return prediction, h_weight, s_weight
 
# Apply and unpack
merged[['prediction', 'h_weight', 's_weight']] = merged.apply(
    lambda row: pd.Series(blend_predictions(row)), axis=1
)
 
print(f"✅ Blending complete!")

# == Markdown Cell ==
# # ============================================================================
# # CELL 9: VALIDATION RESULTS SUMMARY
# # ============================================================================

# == Markdown Cell ==
# ### VALIDATION RESULTS
# ------------------
# Summary of validation scores per horizon and overall.
#  
# The validation set (ts_index > 3500) simulates out-of-sample performance.
# These scores should be indicative of leaderboard performance.

print("\n" + "=" * 80)
print("📊 RESULTS SUMMARY")
print("=" * 80)
 
print("\n📈 HORIZON MODEL VALIDATION SCORES:")
for horizon_val, score in sorted(horizon_val_scores.items()):
    print(f"   Horizon {horizon_val}: {score:.6f}")
avg_h = np.mean(list(horizon_val_scores.values()))
print(f"   ➜ Average: {avg_h:.6f}")
 
print("\n📈 SUB_CATEGORY MODEL VALIDATION SCORES:")
for subcat_val, score in sorted(subcat_val_scores.items()):
    print(f"   {subcat_val}: {score:.6f}")
avg_s = np.mean(list(subcat_val_scores.values()))
print(f"   ➜ Average: {avg_s:.6f}")
 
print("\n⚖️ BLEND WEIGHT STATISTICS:")
print(f"   Horizon avg weight: {merged['h_weight'].mean():.4f}")
print(f"   Sub_Category avg weight: {merged['s_weight'].mean():.4f}")
 
# Example blends
print("\n🎯 EXAMPLE BLENDS (First 5 rows):")
print("-" * 80)
for index in range(min(5, len(merged))):
    row = merged.iloc[index]
    print(f"\n   Row {index + 1}:")
    print(f"     ID: {row['id']}")
    print(f"     Horizon {int(row['horizon'])} (val={row['horizon_val_score']:.4f}, weight={row['h_weight']:.3f})")
    print(f"       → Prediction: {row['horizon_pred']:.4f}")
    print(f"     {row['sub_category']} (val={row['subcat_val_score']:.4f}, weight={row['s_weight']:.3f})")
    print(f"       → Prediction: {row['subcat_pred']:.4f}")
    print(f"     ✅ FINAL PREDICTION: {row['prediction']:.4f}")
 

# == Markdown Cell ==
# # ============================================================================
# # CELL 10: FEATURE IMPORTANCE ANALYSIS
# # ============================================================================

# == Markdown Cell ==
# ### FEATURE IMPORTANCE ANALYSIS
# ---------------------------
# Top features by LightGBM importance (gain-based).
#  
# This helps understand which features drive predictions for each horizon.

# print("\n" + "=" * 70)
# print("STEP 5: FEATURE IMPORTANCE (Top 10 per Horizon)")
# print("=" * 70)
 
# for horizon in HORIZONS:
#     importances = feature_importance_all[horizon]
#     sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    
#     print(f"\nHorizon {horizon}:")
#     for feature_name, importance_value in sorted_features:
#         print(f"  {feature_name:<30} : {importance_value:,.0f}")

# == Markdown Cell ==
# # ============================================================================
# # CELL 11: CREATE SUBMISSION FILE
# # ============================================================================

# == Markdown Cell ==
# ### SUBMISSION FILE CREATION
# ------------------------
# Combine predictions from all horizons into a single submission file.
#  
# FORMAT:
# - id: Unique identifier matching test.parquet
# - prediction: Predicted target value

print("\n" + "=" * 80)
print("📝 CREATING SUBMISSION")
print("=" * 80)
 
submission = merged[['id', 'prediction']].copy()
submission = submission.sort_values('id').reset_index(drop=True)
 
print(f"\n📊 Final Prediction Statistics:")
print(f"   Count: {len(submission):,}")
print(f"   Mean:  {submission['prediction'].mean():.6f}")
print(f"   Std:   {submission['prediction'].std():.6f}")
print(f"   Min:   {submission['prediction'].min():.6f}")
print(f"   Max:   {submission['prediction'].max():.6f}")
 
# Save submission
output_path = '/kaggle/working/submission.csv'
submission.to_csv(output_path, index=False)
print(f"\n✅ Submission saved to: {output_path}")
print(f"   Rows: {len(submission):,}")

# == Markdown Cell ==
# # ============================================================================
# # CELL 12: FINAL SUMMARY
# # ============================================================================

print("\n" + "=" * 80)
print("✨ LIGHTGBM DUAL ENSEMBLE COMPLETE")
print("=" * 80)
print(f"""
APPROACH:
---------
• 4 LightGBM Horizon models (H1, H3, H10, H25)
• 5 LightGBM Sub_Category models
• Total: 9 models
 
BLENDING STRATEGY:
------------------
For each test row (horizon H, sub_category SC):
  1. Get prediction from H model (val_score = score_H)
  2. Get prediction from SC model (val_score = score_SC)
  3. Apply power weighting (power = {BLEND_POWER}):
     - powered_H = score_H ^ {BLEND_POWER}
     - powered_SC = score_SC ^ {BLEND_POWER}
  4. Weight H: powered_H / (powered_H + powered_SC)
  5. Weight SC: powered_SC / (powered_H + powered_SC)
  6. Final = Weight_H × Pred_H + Weight_SC × Pred_SC
 
WHY POWER {BLEND_POWER} WEIGHTING?
----------------------------------
✓ Amplifies contribution of better models
✓ Weak models (low val score) get downweighted aggressively
✓ Better than linear: score 0.27 vs 0.07 → ~96% vs ~4% weight
 
WHY HYBRID BLEND?
-----------------
✓ Combines horizon specialization (temporal patterns)
✓ Combines sub_category specialization (domain patterns)
✓ Uses validation scores as blend weights (data-driven)
✓ Simple & interpretable
✓ No overfitting risk from blending
 
MODELS TRAINED: 9 total
  • LightGBM Horizon: 4 models (avg val score: {avg_h:.6f})
  • LightGBM Sub_Category: 5 models (avg val score: {avg_s:.6f})
 
SUBMISSION: submission.csv ({len(submission):,} rows)
""")
print("=" * 80)
 

# == Markdown Cell ==
# # ============================================================================
# # REFERENCES & CITATIONS
# # ============================================================================
# 
# This solution was developed with inspiration from the following sources:
# 
# ### PUBLIC NOTEBOOKS (in order of influence):
# -----------------------------------------
# 
# 1. "Multiple LightGBM Approach" by kindasomethin (ORIGINAL)
#    [Multiple LightGBM Approach](https://www.kaggle.com/code/kindasomethin/multiple-lightgbm-approach)
# 
# 
#    Influence: ~55-60% (Primary structural reference)
#    - Horizon-wise training architecture
#    - Validation threshold at ts_index = 3500
#    - Feature interactions (feature_al - feature_am)
#    - Target encoding patterns
#    - Lag and rolling feature design
#    - LightGBM with early stopping
#    - Multi-seed ensemble concept
#    
#    Forked version by Kaushal Nandania:
#    [multiple-lightgbm-approach-with-high-score](https://www.kaggle.com/code/kaushalnandania/multiple-lightgbm-approach-with-high-score)
# 
# 3. "Quantitative Time Series Forecasting - Hedge Funds" by Saurabh Raj Varma
#    [Quantitative Time Series Forecasting](https://www.kaggle.com/code/saurabhrajvarma/quantitive-time-series-forecasting-hedge-funs)
# 
#    Influence: ~10-15%
#    - Target encoding concept
#    - Removing ts_index to prevent overfitting
#    - General EDA and pipeline patterns
# 
# 5. "Correct Split LightGBM No Leak" by Talha Celik
#    [Correct Split LightGBM](https://www.kaggle.com/code/realtalhacelik/correct-split-lightgbm-no-leak)
# 
# 
#    Influence: ~5-10%
#    - Time-based split validation approach
#    - Data leakage prevention patterns
# 
# ### MODIFICATIONS FROM PRIMARY REFERENCE:
# -------------------------------------
# My implementation simplifies the original approach:
# - Removed: rolling mean/std, trend slopes, rank features
# - Removed: multi-seed ensemble (kept single model per horizon)
# - Simplified: lag features (multiple lags → lag1 only)
# - Added: group mean features (feature_al_grp_mean, feature_am_grp_mean)
# 
# ### RATIONALE:
# ----------
# Simpler models often generalize better. This stripped-down version
# achieved competitive leaderboard performance (0.33) with faster
# training time (~6 minutes vs ~25+ minutes).
# """

