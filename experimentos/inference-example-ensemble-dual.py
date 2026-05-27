# =========================================================
# INFERENCE EXAMPLE
# Predict using the SAME dataset structure
# used during training
# =========================================================

# pip install -q lightgbm joblib pandas numpy huggingface_hub


from huggingface_hub import hf_hub_download
import joblib
import pandas as pd
import numpy as np


# =========================================================
# 1. DOWNLOAD MODEL
# =========================================================

REPO_ID = "andrewmos/lightbm-ts-forecasting-kaggle"

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="full_pipeline.pkl",
    repo_type="model"
)


# =========================================================
# 2. LOAD PIPELINE
# =========================================================

pipeline = joblib.load(model_path)

print("Pipeline loaded successfully")


# =========================================================
# 3. EXTRACT COMPONENTS
# =========================================================

horizon_models = pipeline["horizon_models"]
subcat_models = pipeline["subcat_models"]

train_stats = pipeline["train_stats"]

blend_scores = pipeline["blend_scores"]

params = pipeline["params"]

blend_power = pipeline["blend_power"]


print("Number of horizon models:", len(horizon_models))
print("Number of subcategory models:", len(subcat_models))


# =========================================================
# 4. LOAD TEST DATA
# =========================================================

# Replace with your real dataset path

test_df = pd.read_csv("test.csv")

print("Test shape:", test_df.shape)


# =========================================================
# 5. FEATURE ENGINEERING
# IMPORTANT:
# COPY THE EXACT SAME FUNCTION
# FROM THE TRAINING NOTEBOOK
# =========================================================

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


# =========================================================
# 6. CREATE FEATURES
# =========================================================

test_df = create_features(
    test_df,
    train_stats=train_stats
)

print("Features created")


# =========================================================
# 7. PREDICT USING ALL MODELS
# =========================================================

all_predictions = {}

for horizon in horizon_models.keys():

    print(f"\nPredicting horizon {horizon}")

    model_info = horizon_models[horizon]

    model = model_info["model"]

    features = model_info["features"]

    # =====================================================
    # ENSURE SAME FEATURE ORDER
    # =====================================================

    X_test = test_df[features]

    # =====================================================
    # PREDICT
    # =====================================================

    preds = model.predict(X_test)

    all_predictions[horizon] = preds

    print(f"Done horizon {horizon}")


# =========================================================
# 8. CONVERT TO DATAFRAME
# =========================================================

predictions_df = pd.DataFrame(all_predictions)

print("\nPredictions shape:")
print(predictions_df.shape)


# =========================================================
# 9. SAVE SUBMISSION
# =========================================================

predictions_df.to_csv("predictions.csv", index=False)

print("\nSaved predictions.csv")


# =========================================================
# 10. PREVIEW
# =========================================================

print("\nPreview:")
print(predictions_df.head())