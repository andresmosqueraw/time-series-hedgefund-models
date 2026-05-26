import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tensorflow.keras.layers import (
    Input, Embedding, Flatten, LSTM, Dense,
    Concatenate, Dropout,
    Reshape
)
from tensorflow.keras.models import Model

import gc

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set TensorFlow to use only the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        print("GPU is available and configured.")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU devices found. Please ensure your Colab runtime type is set to 'GPU'.")

train_df = pd.read_parquet('/kaggle/input/competitions/ts-forecasting/train.parquet')
test_df = pd.read_parquet('/kaggle/input/competitions/ts-forecasting/test.parquet')

# Drop weight column early to save memory (not used as feature)
train_weights = train_df['weight'].values  # keep if needed for evaluation
train_df.drop(columns=['weight'], inplace=True)

print(f"Full Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Subsample: keep complete entities (code + sub_code combos) to preserve full time evolution
# Get unique entity keys
entity_keys = train_df[['code', 'sub_code']].drop_duplicates()
print(f"Total unique (code, sub_code) entities: {len(entity_keys)}")

# Sample ~25% of entities (ensures manageable memory while keeping full temporal data)
np.random.seed(42)
sample_frac = 0.25
sampled_entities = entity_keys.sample(frac=sample_frac, random_state=42)
print(f"Sampled entities: {len(sampled_entities)} ({sample_frac*100:.0f}%)")

# Filter train_df to only include sampled entities
train_df = train_df.merge(sampled_entities, on=['code', 'sub_code'], how='inner').reset_index(drop=True)

# Verify all sub_categories and horizons are represented
print(f"\nSubsampled Train shape: {train_df.shape}")
print(f"Sub-categories in sample: {train_df['sub_category'].nunique()} / {test_df['sub_category'].nunique()}")
print(f"Horizons in sample: {sorted(train_df['horizon'].unique())}")
print(f"ts_index range: {train_df['ts_index'].min()} - {train_df['ts_index'].max()}")
print(f"\nTarget stats:\n{train_df['y_target'].describe()}")
print(f"\nMemory usage: {train_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

gc.collect()

# Define feature columns
feature_cols = [c for c in train_df.columns if c.startswith('feature_')]
cat_cols = ['code', 'sub_code', 'sub_category', 'horizon']

print(f"Number of features: {len(feature_cols)}")
print(f"Categorical columns: {cat_cols}")

# Encode categorical variables (fit on train+test to handle all categories)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(combined)
    train_df[col + '_enc'] = le.transform(train_df[col]).astype(np.int32)
    test_df[col + '_enc'] = le.transform(test_df[col]).astype(np.int32)
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} unique values")

cat_enc_cols = [c + '_enc' for c in cat_cols]

# Drop original categorical string columns to save memory
train_df.drop(columns=cat_cols + ['id'], inplace=True)
test_ids = test_df['id'].values  # save for submission
test_df.drop(columns=cat_cols + ['id'], inplace=True)
gc.collect()
print(f"\nMemory after dropping strings: {train_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

unique_counts = {col: len(label_encoders[col].classes_) for col in cat_cols}
unique_counts

# Convert features to float32 and fill NaN in-place to save memory
for col in feature_cols:
    train_df[col] = train_df[col].fillna(0).astype(np.float32)
for col in feature_cols:
    test_df[col] = test_df[col].fillna(0).astype(np.float32)

gc.collect()

# Normalize numerical features (fit on train only, transform both)
# Use partial_fit approach to avoid creating full copy
scaler = StandardScaler()

# Fit scaler on train data
scaler.fit(train_df[feature_cols])

# Transform in chunks to reduce peak memory
chunk_size = 20  # process 20 columns at a time
for i in range(0, len(feature_cols), chunk_size):
    cols_chunk = feature_cols[i:i+chunk_size]
    idx = list(range(i, min(i+chunk_size, len(feature_cols))))
    train_df[cols_chunk] = ((train_df[cols_chunk].values - scaler.mean_[idx]) / scaler.scale_[idx]).astype(np.float32)
    test_df[cols_chunk] = ((test_df[cols_chunk].values - scaler.mean_[idx]) / scaler.scale_[idx]).astype(np.float32)

gc.collect()
print("Preprocessing complete.")
print(f"Train NaN count: {train_df[feature_cols].isnull().sum().sum()}")
print(f"Memory usage train: {train_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print(f"Memory usage test: {test_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Split training data into train/validation using time-based split
# Use last portion of ts_index as validation (respects temporal order)
val_cutoff = int(train_df['ts_index'].max() * 0.85)  # ~85% for training
print(f"Validation cutoff ts_index: {val_cutoff}")

train_mask = train_df['ts_index'] <= val_cutoff
val_mask = train_df['ts_index'] > val_cutoff

train_split = train_df[train_mask].reset_index(drop=True)
val_split = train_df[val_mask].reset_index(drop=True)

print(f"Train split: {len(train_split)} rows (ts_index 1-{val_cutoff})")
print(f"Val split: {len(val_split)} rows (ts_index {val_cutoff+1}-{train_df['ts_index'].max()})")

train_feat_split = train_split[feature_cols].values.reshape((train_split.shape[0], 1, len(feature_cols)))
val_feat_split = val_split[feature_cols].values.reshape((val_split.shape[0], 1, len(feature_cols)))

train_feat_split.shape, val_feat_split.shape

train_dataset = tf.data.Dataset.from_tensor_slices(train_feat_split)
val_dataset = tf.data.Dataset.from_tensor_slices(val_feat_split)

cat_dataset = tf.data.Dataset.from_tensor_slices((
    tf.cast(train_split['code_enc'].values, tf.int32),
    tf.cast(train_split['sub_code_enc'].values, tf.int32),
    tf.cast(train_split['sub_category_enc'].values, tf.int32),
    tf.cast(train_split['horizon_enc'].values, tf.int32),
))

targets = tf.data.Dataset.from_tensor_slices(
    tf.cast(train_split['y_target'].values, tf.float32)
)

full_dataset = tf.data.Dataset.zip(((train_dataset, cat_dataset), targets)) \
    .shuffle(1024) \
    .batch(256) \
    .prefetch(tf.data.AUTOTUNE)

val_cat_dataset = tf.data.Dataset.from_tensor_slices((
    tf.cast(val_split['code_enc'].values, tf.int32),
    tf.cast(val_split['sub_code_enc'].values, tf.int32),
    tf.cast(val_split['sub_category_enc'].values, tf.int32),
    tf.cast(val_split['horizon_enc'].values, tf.int32),
))

val_targets = tf.data.Dataset.from_tensor_slices(
    tf.cast(val_split['y_target'].values, tf.float32)
)

val_full_dataset = tf.data.Dataset.zip(((val_dataset, val_cat_dataset), val_targets)) \
    .shuffle(1024) \
    .batch(256) \
    .prefetch(tf.data.AUTOTUNE)

print(f'Objetos eliminados: {gc.collect()}')

# == Markdown Cell ==
# # Modelo

# Sequence input
seq_input = Input(shape=train_dataset.element_spec.shape, name="seq_input")
# seq_input = Flatten()(seq_input)

# code
code_input = Input(shape=(1,), name="code_input")
code_emb = Embedding(input_dim=unique_counts['code'], output_dim=len(feature_cols), name='code_embedding')(code_input)
# code_emb = Flatten()(code_emb)

# sub_code
sub_code_input = Input(shape=(1,), name="sub_code_input")
sub_code_emb = Embedding(input_dim=unique_counts['sub_code'], output_dim=len(feature_cols), name='sub_code_embedding')(sub_code_input)
# sub_code_emb = Flatten()(sub_code_emb)

# sub_category
sub_cat_input = Input(shape=(1,), name="sub_cat_input")
sub_cat_emb = Embedding(input_dim=unique_counts['sub_category'], output_dim=len(feature_cols), name='sub_category_embedding')(sub_cat_input)
# sub_cat_emb = Flatten()(sub_cat_emb)

# horizon
horizon_input = Input(shape=(1,), name="horizon_input")
horizon_emb = Embedding(input_dim=unique_counts['horizon'], output_dim=len(feature_cols),name='horizon_embedding')(horizon_input)
# horizon_emb = Flatten()(horizon_emb)

# Combine
x = Concatenate(axis=1, name='concat')([
    code_emb,
    sub_code_emb,
    sub_cat_emb,
    horizon_emb,
    seq_input
])

# x = Reshape((-1, 1))(x)
x = LSTM(256)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu")(x)

output = Dense(1)(x)

model = Model(
    inputs=[
        seq_input,
        code_input,
        sub_code_input,
        sub_cat_input,
        horizon_input
    ],
    outputs=output
)

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="mae",
    metrics=['mse']
)

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

callbacks = [
    # early_stop,
    reduce_lr
]

history = model.fit(
    full_dataset,
  # [
  #   training_dataset,
  #   code_dataset,
  #   sub_code_dataset,
  #   sub_category_dataset,
  #   horizon_dataset
  # ],
  # y_train,
  validation_data=val_full_dataset,
  epochs=100,
  callbacks=callbacks
)

import matplotlib.pyplot as plt

# grafica de funcion de perdida y metricas
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.title('Loss (MAE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# grafica de funcion de perdida y metricas
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['mse'], label='Training')
plt.plot(history.history['val_mse'], label='Validation')
plt.legend()
plt.title('MSE')
plt.xlabel('Epochs')
plt.ylabel('')
plt.show()

model.save("model.keras")

import json
history_dict = history.history
# Save it to a file
with open("training_history.json", "w") as f:
    json.dump(history_dict, f)

