import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

np.random.seed(42)

train_path = "/kaggle/input/competitions/ts-forecasting/train.parquet"
test_path = "/kaggle/input/competitions/ts-forecasting/test.parquet"

train_full = pd.read_parquet(train_path)
test_full = pd.read_parquet(test_path)

TARGET = "y_target"
VAL_THRESHOLD = 3503
FORECAST_WINDOWS = [1, 3, 10, 25]

lgb_cfg = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'n_estimators': 1200,
    'num_leaves': 64,
    'min_child_samples': 100,
    'feature_fraction': 0.7,
    'verbosity': -1,
    #'n_jobs': -1,
    #'random_state': 5,
    'seed': 5,
    'deterministic': True,
    'force_row_wise': True,
    #'force_col_wise': True,
    'num_threads': 1
}

def build_context_features(df):
    df = df.copy()

    for col1, col2 in [('feature_al', 'feature_am')]:
        if col1 in df.columns and col2 in df.columns:
            df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]

    group_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    for col in ['feature_al', 'feature_am']:
        if col in df.columns:
            df[f'{col}_grp_mean'] = df.groupby(group_cols)[col].transform('mean')

    return df

def weighted_rmse_score(y_true, y_pred, w):
    num = np.sum(w * (y_true - y_pred) ** 2)
    den = np.sum(w * (y_true ** 2)) + 1e-8
    return float(np.sqrt(num / den))

test_outputs = []
cv_cache = {'y': [], 'pred': [], 'wt': []}

for horizon in FORECAST_WINDOWS:
    tr_df = train_full[train_full['horizon'] == horizon].copy()
    te_df = test_full[test_full['horizon'] == horizon].copy()

    tr_df = build_context_features(tr_df)
    te_df = build_context_features(te_df)

    train_df = tr_df[tr_df['ts_index'] <= VAL_THRESHOLD]
    val_df = tr_df[tr_df['ts_index'] > VAL_THRESHOLD]

    drop_cols = [TARGET, 'ts_index']

    common_cols = list(set(tr_df.columns) & set(te_df.columns))
    features = [c for c in common_cols if c not in drop_cols]

    X_train = train_df[features].copy()
    y_train = train_df[TARGET]

    X_val = val_df[features].copy()
    y_val = val_df[TARGET]

    X_test = te_df[features].copy()

    for col in features:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')

    train_weights = train_df['weight'].values if 'weight' in train_df.columns else None
    val_weights = val_df['weight'].values if 'weight' in val_df.columns else None

    model = lgb.LGBMRegressor(**lgb_cfg)

    model.fit(
        X_train, y_train,
        sample_weight=train_weights,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[val_weights] if val_weights is not None else None,
        eval_metric='rmse',
        categorical_feature='auto',
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )

    val_pred = model.predict(X_val)

    weights = val_weights if val_weights is not None else np.ones_like(y_val)
    weights = weights[:len(y_val)]

    score = weighted_rmse_score(y_val.values, val_pred, weights)

    cv_cache['y'].extend(y_val.values)
    cv_cache['pred'].extend(val_pred)
    cv_cache['wt'].extend(weights)

    test_pred = model.predict(X_test)

    te_df['prediction'] = test_pred
    test_outputs.append(te_df[['id', 'prediction']])  # FIXED

    del tr_df, te_df, train_df, val_df
    del X_train, X_val, X_test, y_train, y_val
    gc.collect()

final_score = weighted_rmse_score(
    np.array(cv_cache['y']),
    np.array(cv_cache['pred']),
    np.array(cv_cache['wt'])
)

submission = pd.concat(test_outputs, ignore_index=True)
submission = submission.sort_values('id').reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

