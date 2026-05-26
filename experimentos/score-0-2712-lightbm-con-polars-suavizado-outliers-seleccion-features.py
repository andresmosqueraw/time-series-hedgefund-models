pip install pyclustering


import warnings
import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import polars.selectors as cs
import math
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.datasets import make_blobs
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from sklearn.datasets import make_blobs
import itertools
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
warnings.filterwarnings('ignore')


TRAIN_PATH = '/kaggle/input/competitions/ts-forecasting/train.parquet'
TEST_PATH = '/kaggle/input/competitions/ts-forecasting/test.parquet'

VAL_THRESHOLD = 3500
FORECAST_WINDOWS = [1, 3, 10, 25]


LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.015,
    'n_estimators': 4200,
    'num_leaves': 31,
    'min_child_samples': 200,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 10.0,
    'verbosity': -1
}

# == Markdown Cell ==
# ## FEATURES


def fill_data(df_train, df_test):
    """
    Gộp chung Train và Test để Forward Fill, giúp tập Test mượn được 
    giá trị cuối cùng của tập Train để lấp đầy những ngày đầu tiên bị khuyết.
    """
    print("🔗 Đang gộp Train và Test để Forward Fill (Mượn đà lịch sử)...")
    
    # --- 1. ĐÁNH DẤU VÀ GỘP BẢNG ---
    df_train = df_train.with_columns(pl.lit(False).alias("_is_test"))
    df_test = df_test.with_columns(pl.lit(True).alias("_is_test"))
    
    # Lưu lại danh sách cột gốc của tập Test để lát sau tách ra không bị dư cột lạ
    original_test_cols = [c for c in df_test.columns if c != "_is_test"]
    
    # Nối chéo (Diagonal) vì tập Test không có y_target và weight
    combined_df = pl.concat([df_train, df_test], how="diagonal")
    
    # --- 2. SẮP XẾP CHUẨN XÁC (Sửa lỗi Random "Bóng ma") ---
    # BẮT BUỘC: Phải sort theo Cụm (code, sub_code...) trước rồi mới đến Thời gian
    group_cols = ["code", "sub_code", "sub_category", "horizon"]
    combined_df = combined_df.sort(group_cols + ["ts_index"])
    
    # --- 3. XÁC ĐỊNH CỘT CẦN FILL VÀ THỰC THI ---
    # ⚠️ CẢNH BÁO SỐNG CÒN: Tuyệt đối không được fill cột y_target và weight, 
    # nếu không giá trị y_target của Train sẽ chảy tràn sang Test gây Data Leakage!
    exclude_cols = group_cols + ["id", "ts_index", "y_target", "weight", "_is_test"]
    
    print("🔄 Đang xử lý Forward Fill xuyên suốt 2 tập dữ liệu...")
    combined_df = combined_df.with_columns(
        cs.numeric().exclude(exclude_cols) # Chỉ chọn các cột số là Feature
        .forward_fill()                    # Fill từ trên xuống dưới
        .fill_null(0)                      # Nếu đoạn đầu cùng vẫn Null thì điền 0
        .over(group_cols)                  # Chạy độc lập trong từng mã cổ phiếu/sản phẩm
    )
    
    # --- 4. TÁCH TRẢ LẠI ---
    print("✂️ Đang tách trả lại tập Train và Test nguyên vẹn...")
    
    # Lấy lại tập Train
    df_train_new = combined_df.filter(~pl.col("_is_test")).drop("_is_test")
    
    # Lấy lại tập Test (Chỉ lấy đúng những cột mà Test vốn có ban đầu)
    df_test_new = combined_df.filter(pl.col("_is_test")).select(original_test_cols)
    
    print(f"✅ Hoàn tất Fill Data!")
    print(f"   📉 Train shape: {df_train_new.shape}")
    print(f"   📈 Test shape:  {df_test_new.shape}")
    del combined_df
    return df_train_new, df_test_new

def reduce_memory_usage(df: pl.DataFrame, name="Data"):
    """
    Tự động chuyển đổi các cột sang kiểu dữ liệu nhỏ nhất có thể.
    - Float64 -> Float32
    - Int64 -> Int8, Int16, Int32 (tùy giá trị max/min)
    - String -> Categorical (Nếu số lượng giá trị duy nhất thấp)
    """
    start_mem = df.estimated_size("mb")
    print(f"[{name}] Dung lượng ban đầu: {start_mem:.2f} MB")
    
    # 1. Xử lý số thực (Float): Mặc định chuyển hết về Float32 (Đủ cho Deep Learning)
    # Float32 nhẹ bằng 1/2 so với Float64
    cols_float = df.select(pl.col(pl.Float64)).columns
    if cols_float:
        df = df.with_columns(pl.col(cols_float).cast(pl.Float32))

    # 2. Xử lý số nguyên (Int): Kiểm tra min/max để chọn loại Int bé nhất
    cols_int = df.select(pl.col(pl.Int64)).columns
    for col in cols_int:
        # Lấy min/max để quyết định kiểu
        c_min = df[col].min()
        c_max = df[col].max()
        
        # Chọn kiểu dữ liệu phù hợp
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            new_type = pl.Int8
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            new_type = pl.Int16
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            new_type = pl.Int32
        else:
            new_type = pl.Int64 # Giữ nguyên nếu số quá lớn
            
        if new_type != pl.Int64:
            df = df.with_columns(pl.col(col).cast(new_type))

    # 3. Xử lý String -> Categorical (Quan trọng nhất)
    # Nếu một cột String có ít giá trị lặp lại (ví dụ 'horizon', 'sub_category'), chuyển sang Categorical cực nhẹ
    cols_str = df.select(pl.col(pl.Utf8)).columns
    for col in cols_str:
        n_unique = df[col].n_unique()
        n_rows = df.height
        
        # Ngưỡng: Nếu số giá trị duy nhất < 50% tổng số dòng -> Chuyển sang Categorical
        # Ví dụ: 'sub_category' chỉ có 10 loại trên 5 triệu dòng -> Rất nên chuyển
        if n_unique / n_rows < 0.5:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))

    end_mem = df.estimated_size("mb")
    print(f"[{name}] Dung lượng sau tối ưu: {end_mem:.2f} MB (Giảm {(start_mem - end_mem)/start_mem*100:.1f}%)")
    
    return df

def smooth_outliers_hampel(df_train, df_test, features, window_size=50, threshold=5.0):
    """
    Làm mượt Outlier bằng thuật toán Pseudo-Hampel Filter.
    GHI ĐÈ TRỰC TIẾP lên các feature gốc để tiết kiệm tối đa RAM.
    """
    print(f"🌊 Bắt đầu quy trình SMOOTH OUTLIER (Ghi đè {len(features)} features gốc)...")
    
    TEMPORAL_KEYS = ["code", "sub_code", "sub_category", "horizon"]
    
    # 1. Gộp bảng chống đứt gãy timeline
    df_train = df_train.with_columns(pl.lit(False).alias("_is_test"))
    df_test  = df_test.with_columns(pl.lit(True).alias("_is_test"))
    combined_df = pl.concat([df_train, df_test], how="diagonal").sort(TEMPORAL_KEYS + ["ts_index"])

    # 2. XÂY DỰNG BỘ LỌC
    smooth_exprs = []
    
    for feat in features:
        # Tính Median và Std cục bộ (Rolling)
        r_median = pl.col(feat).rolling_median(window_size=window_size, min_periods=1, center=True).over(TEMPORAL_KEYS)
        r_std = pl.col(feat).rolling_std(window_size=window_size, min_periods=1, center=True).over(TEMPORAL_KEYS).fill_null(0)
        
        # Công thức nhận diện Outlier: |X - Median| > Threshold * Std
        is_outlier = (pl.col(feat) - r_median).abs() > (threshold * r_std)
        
        # ÉP MƯỢT VÀ GHI ĐÈ: Thay bằng .alias(feat) để đè đúng tên cột cũ
        expr_smooth = pl.when(is_outlier)\
                        .then(r_median)\
                        .otherwise(pl.col(feat))\
                        .alias(feat)
        
        smooth_exprs.append(expr_smooth)

    # 3. THỰC THI POLARS
    print(f"   * Kích hoạt máy chà nhám (Window={window_size}, Lực ép={threshold} Sigma)...")
    combined_df = combined_df.with_columns(smooth_exprs)

    # 4. TÁCH TRẢ BẢNG
    print("✂️ Tách trả lại tập Train và Test...")
    df_train_new = combined_df.filter(~pl.col("_is_test")).drop("_is_test")
    
    # Lấy lại đúng cấu trúc cột của tập Test ban đầu (đã được làm mượt bên trong)
    original_test_cols = [c for c in df_test.columns if c != "_is_test"]
    df_test_new = combined_df.filter(pl.col("_is_test")).select(original_test_cols)
    
    # Xóa sạch dấu vết của bảng gộp để cứu RAM ngay lập tức
    del combined_df
    gc.collect()
    
    print("✅ HOÀN TẤT! Dữ liệu gốc đã được chà mượt mà như lụa.")
    return df_train_new, df_test_new

import gc
def create_sign_categorical_features(df_train, df_test, feature_list):
    """
    Nhận vào df_train, df_test và một list các features.
    Tạo ra các cột Categorical mới dựa trên DẤU (-1, 0, 1) của các features đó.
    """
    print(f"\n⚙️ [2/2] Đang tạo Categorical features từ dấu của {len(feature_list)} features...")
    
    new_cat_cols = []
    exprs = []
    
    for f in feature_list:
        new_col_name = f"{f}_sign_cat"
        new_cat_cols.append(new_col_name)
        
        # Biểu thức Polars: Lấy dấu -> Ép sang chuỗi -> Ép sang Categorical
        exprs.append(
            pl.col(f).sign().cast(pl.Utf8).cast(pl.Categorical).alias(new_col_name)
        )
        
    # Apply cùng 1 lúc (rất nhanh) cho cả 2 tập dữ liệu
    df_train = df_train.with_columns(exprs)
    df_test = df_test.with_columns(exprs)
    
    print(f"   ✅ Đã tạo thành công {len(new_cat_cols)} cột mới: {new_cat_cols}")
    gc.collect()
    return df_train, df_test

def add_sign_magnitude_to_both(df, df_test, col_a, col_b, new_col_name=None):
    """
    Tạo feature: sign(col_a) * abs(col_b)
    Áp dụng đồng thời bằng 1 biểu thức duy nhất cho cả df (Train) và df_test (Test).
    """
    if new_col_name is None:
        new_col_name = f"{col_a}_sign_x_{col_b}_abs"
        
    print(f"⚡ Đang tạo feature '{new_col_name}' trên cả 2 tập dữ liệu...")
    
    # 1. Khai báo biểu thức tính toán (Expression)
    expr = (pl.col(col_a).sign() * pl.col(col_b).abs()).alias(new_col_name)
    
    # 2. Áp dụng biểu thức lên cả 2 DataFrames
    df = df.with_columns(expr)
    df_test = df_test.with_columns(expr)
    
    print("   ✅ Hoàn tất!")
    
    return df, df_test

import gc
def add_interactive_features(df_train, df_test, features_to_interact):
    """
    Tạo các feature tương tác (Cộng, Trừ, Nhân, Chia) giữa các cặp feature.
    - Truyền vào 2 feature: Tạo ra 1 cặp (5 cột mới).
    - Truyền vào 3 feature: Tạo ra 3 cặp (15 cột mới).
    """
    print(f"🧬 Bắt đầu lai tạo Interactive Features cho {len(features_to_interact)} cột...")
    
    # Tạo tất cả các cặp kết hợp có thể có (Không trùng lặp)
    # Ví dụ: [A, B, C] -> (A,B), (A,C), (B,C)
    pairs = list(itertools.combinations(features_to_interact, 2))
    print(f"🔗 Đã tìm ra {len(pairs)} cặp để lai tạo.")

    interaction_exprs = []
    new_cols = []
    
    for f1, f2 in pairs:
        # 1. Phép CỘNG (Tổng sức mạnh)
        add_name = f"inter_{f1}_plus_{f2}"
        expr_add = (pl.col(f1) + pl.col(f2)).alias(add_name)
        
        # 2. Phép TRỪ (Khoảng cách / Chênh lệch)
        # Thường dùng thêm .abs() để lấy độ lệch tuyệt đối
        sub_name = f"inter_{f1}_minus_{f2}"
        expr_sub = (pl.col(f1) - pl.col(f2)).alias(sub_name)
        
        # 3. Phép NHÂN (Khuếch đại tín hiệu - Rất quan trọng nếu 2 biến có ý nghĩa tỷ lệ thuận)
        mul_name = f"inter_{f1}_mul_{f2}"
        expr_mul = (pl.col(f1) * pl.col(f2)).alias(mul_name)
        
        # 4. Phép CHIA (Tỷ lệ tương quan)
        # Phải cộng thêm 1e-9 để tránh lỗi chia cho 0 (ZeroDivisionError)
        div1_name = f"inter_{f1}_div_{f2}"
        expr_div1 = (pl.col(f1) / (pl.col(f2) + 1e-9)).fill_null(0).alias(div1_name)
        
        div2_name = f"inter_{f2}_div_{f1}"
        expr_div2 = (pl.col(f2) / (pl.col(f1) + 1e-9)).fill_null(0).alias(div2_name)
        
        # Đưa vào danh sách chờ thực thi
        interaction_exprs.extend([expr_add, expr_sub, expr_mul, expr_div1, expr_div2])
        new_cols.extend([add_name, sub_name, mul_name, div1_name, div2_name])

    # Thực thi tính toán chớp nhoáng trên cả Train và Test
    print("⚙️ Đang ép xung xử lý bằng Polars...")
    df_train_new = df_train.with_columns(interaction_exprs)
    df_test_new  = df_test.with_columns(interaction_exprs)
    
    print(f"✅ HOÀN TẤT! Đã đúc thêm {len(new_cols)} Interactive Features mới.")
    gc.collect()
    return df_train_new, df_test_new

# =====================================================================
# CÁCH GỌI HÀM:
# =====================================================================
# Bác chọn ra 2 hoặc 3 feature xịn nhất từ bước quét Correlation để cho chúng lai tạo
# top_features = ["feature_v", "feature_am", "feature_u"]
# df, df_test = add_interactive_features(df, df_test, top_features)

def dispel_unrelated_data(df, df_test, THRESHOLD):
    """
    Lọc bỏ các feature có độ tương quan thấp với Target trên tập Train.
    Sau đó áp dụng danh sách feature giữ lại này để gọt giũa cả tập Train và tập Test.
    """
    TARGET_COL = "y_target"
    # Các cột "Bất khả xâm phạm" (ID, Target, Weight...)
    # Lưu ý: Tập Test sẽ không có y_target và weight, ta cần xử lý khéo ở đoạn cuối
    compulsory_cols = ["id", "code", "sub_code", "sub_category", "horizon", "ts_index", "y_target", "weight"]
    
    # --- 1. TÍNH TOÁN TƯƠNG QUAN TRÊN TẬP TRAIN ---
    print(f"🔄 Đang tính toán tương quan Spearman trên tập Train (Threshold >= {THRESHOLD})...")

    # Lấy các cột số thực sự là feature (loại bỏ các cột ID/Target)
    feature_cols = df.select(cs.numeric().exclude(compulsory_cols)).columns

    # Tính correlation từng cột với Target
    # Mẹo: Dùng list comprehension của Python kết hợp Polars expression sẽ nhanh hơn vòng lặp thường
    correlations = []
    
    # Nếu data quá lớn, có thể lấy mẫu (sample) để tính correlation cho nhanh
    # df_sample = df.sample(n=500000) if len(df) > 500000 else df
    
    for col in feature_cols:
        # Tính Spearman (quan hệ phi tuyến) hoặc Pearson (tuyến tính)
        corr_val = df.select(pl.corr(col, TARGET_COL, method='spearman')).item()
        
        # Chỉ giữ lại nếu giá trị hợp lệ (không phải NaN/None)
        if corr_val is not None:
            correlations.append((col, corr_val))

    # --- 2. CHỌN LỌC FEATURE ---
    # Sắp xếp theo trị tuyệt đối giảm dần (Quan trọng nhất lên đầu)
    top_features = sorted(
        [x for x in correlations if abs(x[1]) >= THRESHOLD], 
        key=lambda x: abs(x[1]), 
        reverse=True
    )

    # In ra danh sách các feature đạt chuẩn
    print(f"\n📊 Tìm thấy {len(top_features)} features đạt chuẩn:")
    for feat, cor in top_features[:10]: # In mẫu 10 cái đầu
        print(f"  - {feat}: {abs(cor):.4f}")
    if len(top_features) > 10: print("  ... và các feature khác.")

    selected_feature_names = [item[0] for item in top_features]

    # --- 3. ÁP DỤNG LỌC ĐỒNG BỘ CHO CẢ TRAIN VÀ TEST ---
    print(f"\n✂️ Đang cắt gọt dữ liệu...")

    # A. Xử lý tập TRAIN: Giữ lại cột bắt buộc + Feature được chọn
    final_cols_train = compulsory_cols + selected_feature_names
    # (Dùng intersection để tránh lỗi nếu lỡ tay xóa mất cột nào đó trước đó)
    final_cols_train = [c for c in final_cols_train if c in df.columns]
    
    df_reduced = df.select(final_cols_train)

    # B. Xử lý tập TEST: Giữ lại cột bắt buộc (trừ Target/Weight) + Feature được chọn
    # Tập test thường chỉ cần ID + Feature để chạy model
    test_compulsory = ["id", "code", "sub_code", "sub_category", "horizon", "ts_index", "y_target", "weight"]
    final_cols_test = test_compulsory + selected_feature_names
    
    # Kiểm tra an toàn: Chỉ select những cột thực sự có trong df_test
    # (Đề phòng trường hợp feature engineer bị thiếu trên tập test)
    final_cols_test = [c for c in final_cols_test if c in df_test.columns]

    df_test_reduced = df_test.select(final_cols_test)

    print(f"✅ Hoàn tất! Kích thước sau khi lọc:")
    print(f"   - Train: {df_reduced.shape}")
    print(f"   - Test : {df_test_reduced.shape}")

    return df_reduced, df_test_reduced

import gc

def add_custom_feature(df_train, df_test, custom_features, windows=[1000]):
    """
    Tạo tính năng Tích lũy/Rolling: Median, Max, Min, Robust Z-Score.
    Phiên bản dùng Median thay cho Mean để chống nhiễu (Outliers) tuyệt đối!
    """
    # --- 1. CẤU HÌNH VÀ KIỂM TRA ---
    KEY_COLS = ["code", "sub_code", "sub_category", "horizon"]
    
    missing_cols = [f for f in custom_features if f not in df_train.columns]
    if missing_cols:
        raise ValueError(f"❌ LỖI: Các cột sau không tồn tại trong df_train: {missing_cols}")
        
    print(f"🔥 Bắt đầu xử lý cho {len(custom_features)} features (Median, Max, Min, Robust Z-Score)...")

    # --- 2. GHÉP NỐI ĐỂ MƯỢN ĐÀ LỊCH SỬ ---
    print("🔗 Đang nối Train và Test để mượn dữ liệu lịch sử...")
    df_train = df_train.with_columns(pl.lit(False).alias("_is_test"))
    df_test  = df_test.with_columns(pl.lit(True).alias("_is_test"))
    
    combined_df = pl.concat([df_train, df_test], how="diagonal")
    
    # Sắp xếp chuẩn để hàm Rolling chạy mượt theo trục thời gian của từng nhóm
    combined_df = combined_df.sort(KEY_COLS + ["ts_index"])

    # --- 3. TẠO SIÊU FEATURES ---
    print(f"🚀 Đang tạo Rolling/Tích lũy với cửa sổ {windows}...")
    
    feature_exprs = []
    new_feature_names = []
    
    for feat in custom_features:
        for w in windows:
            # 1. MEDIAN TÍCH LŨY (Kháng nhiễu cực mạnh)
            rmedian_name = f"{feat}_median_{w}"
            expr_rmedian = pl.col(feat).rolling_median(window_size=w, min_periods=1).over(KEY_COLS)
            
            # 2. MAX TÍCH LŨY
            rmax_name = f"{feat}_max_{w}"
            expr_rmax = pl.col(feat).rolling_max(window_size=w, min_periods=1).over(KEY_COLS)
            
            # 3. MIN TÍCH LŨY
            rmin_name = f"{feat}_min_{w}"
            expr_rmin = pl.col(feat).rolling_min(window_size=w, min_periods=1).over(KEY_COLS)

            # 4. ĐỘ LỆCH CHUẨN TÍCH LŨY (Làm mẫu số cho Z-Score)
            expr_rstd = pl.col(feat).rolling_std(window_size=w, min_periods=1).over(KEY_COLS).fill_null(0)
            
            # 5. ROBUST Z-SCORE TÍCH LŨY (Lấy Median làm tâm)
            zscore_name = f"{feat}_robust_zscore_{w}"
            expr_zscore = ((pl.col(feat) - expr_rmedian) / (expr_rstd + 1e-9)).fill_null(0)
            
            # Gói ghém lại để Polars chạy song song bằng C++ SIMD
            feature_exprs.extend([
                expr_rmedian.alias(rmedian_name), 
                expr_rmax.alias(rmax_name), 
                expr_rmin.alias(rmin_name), 
                expr_zscore.alias(zscore_name)
            ])
            
            # Lưu tên cột mới để lát nữa lọc cho tập Test
            new_feature_names.extend([
                rmedian_name, rmax_name, rmin_name, zscore_name
            ])

    # Kích hoạt thực thi toàn bộ công thức trong 1 nhịp thở
    print(f"   * Đang ép xung xử lý {len(new_feature_names)} cột mới...")
    combined_df = combined_df.with_columns(feature_exprs)

    # --- 4. TÁCH TRẢ LẠI TẬP TRAIN VÀ TEST ---
    print("✂️ Đang tách trả lại tập Train và Test...")
    df_train_new = combined_df.filter(~pl.col("_is_test")).drop("_is_test")
    
    original_test_cols = [c for c in df_test.columns if c != "_is_test"]
    df_test_new = combined_df.filter(pl.col("_is_test")).select(original_test_cols + new_feature_names)
    
    print(f"✅ HOÀN TẤT! Đã đắp thêm {len(new_feature_names)} features Tích lũy/Rolling siêu xịn.")
    
    # Ép anh lao công dọn dẹp biến tạm giải phóng RAM ngay lập tức
    del combined_df
    gc.collect()
    
    return df_train_new, df_test_new

import gc

def add_cumulative(df_train, df_test, features, spans=[10, 30]):
    print(f"🔥 Bắt đầu tính Combo Tích Lũy Kháng Nhiễu (Expanding Median & EMA) cho {len(features)} features...")
    
    TEMPORAL_KEYS = ["code", "sub_code", "sub_category", "horizon"]
    
    df_train = df_train.with_columns(pl.lit(False).alias("_is_test"))
    df_test  = df_test.with_columns(pl.lit(True).alias("_is_test"))
    
    combined_df = pl.concat([df_train, df_test], how="diagonal")
    combined_df = combined_df.sort(TEMPORAL_KEYS + ["ts_index"])

    # Lấy tổng số dòng để làm "Cửa sổ vô cực" cho hàm Expanding Median
    max_window = combined_df.height

    # CHIA LÀM 2 NHỊP ĐỂ TRÁNH LỖI SONG SONG CỦA POLARS
    base_exprs = []     # Nhịp 1: Chứa các cột gốc (EMA, Median)
    derived_exprs = []  # Nhịp 2: Chứa các cột phái sinh (MACD, Conflict)
    new_cols = []
    
    for feat in features:
        # =====================================================================
        # NHỊP 1: TÍNH TOÁN CÁC CỘT NỀN TẢNG (BASE)
        # =====================================================================
        
        # 1. EXPANDING MEDIAN (Tuyệt chiêu ép Rolling thành Expanding)
        expanding_median_name = f"{feat}_expanding_median"
        expr_expanding_median = pl.col(feat).rolling_median(
            window_size=max_window, 
            min_periods=1
        ).over(TEMPORAL_KEYS).alias(expanding_median_name)
        
        expanding_diff_name = f"{feat}_diff_expanding_median"
        expr_expanding_diff = (pl.col(feat) - expr_expanding_median).alias(expanding_diff_name)
        
        base_exprs.extend([expr_expanding_median, expr_expanding_diff])
        new_cols.extend([expanding_median_name, expanding_diff_name])

        # 2. EXPONENTIAL MOMENTUM (EMA - Giữ nguyên để tính MACD chuẩn)
        for span in spans:
            ema_name = f"{feat}_ema_{span}"
            expr_ema = pl.col(feat).ewm_mean(span=span, min_periods=1, ignore_nulls=True).over(TEMPORAL_KEYS).alias(ema_name)
            
            ema_diff_name = f"{feat}_diff_ema_{span}"
            expr_ema_diff = (pl.col(feat) - expr_ema).alias(ema_diff_name)
            
            ema_ratio_name = f"{feat}_ratio_ema_{span}"
            expr_ema_ratio = (pl.col(feat) / (expr_ema + 1e-9)).fill_null(1).alias(ema_ratio_name)
            
            base_exprs.extend([expr_ema, expr_ema_diff, expr_ema_ratio])
            new_cols.extend([ema_name, ema_diff_name, ema_ratio_name])

        # =====================================================================
        # NHỊP 2: TÍNH TOÁN CÁC CỘT LAI TẠO (DERIVED)
        # =====================================================================
        
        # 1. MACD (Dựa trên EMA)
        if len(spans) >= 2:
            short_span, long_span = spans[0], spans[-1]
            macd_name = f"{feat}_macd_{short_span}_{long_span}"
            
            expr_macd = (pl.col(f"{feat}_ema_{short_span}") - pl.col(f"{feat}_ema_{long_span}")).alias(macd_name)
            derived_exprs.append(expr_macd)
            new_cols.append(macd_name)

        # 2. XUNG ĐỘT NGẮN - DÀI (Trend Conflict: Tín hiệu nhanh EMA vs Mỏ neo Median)
        trend_conflict_name = f"{feat}_trend_conflict_{spans[0]}_vs_expanding_median"
        expr_trend_conflict = (pl.col(f"{feat}_ema_{spans[0]}") - pl.col(f"{feat}_expanding_median")).alias(trend_conflict_name)
        
        derived_exprs.append(expr_trend_conflict)
        new_cols.append(trend_conflict_name)

    # --- THỰC THI POLARS CỰC TỐC ---
    print("🚀 Đang chạy Polars xử lý Tích lũy (Nhịp 1 -> Nhịp 2)...")
    combined_df = combined_df.with_columns(base_exprs).with_columns(derived_exprs)

    # --- TÁCH TRẢ LẠI ---
    print("✂️ Đang tách trả lại tập Train và Test...")
    df_train_new = combined_df.filter(~pl.col("_is_test")).drop("_is_test")
    
    original_test_cols = [c for c in df_test.columns if c != "_is_test"]
    df_test_new = combined_df.filter(pl.col("_is_test")).select(original_test_cols + new_cols)
    
    del combined_df
    gc.collect()
    
    print(f"✅ HOÀN TẤT! Đã đắp thêm {len(new_cols)} features Tích lũy (Hệ Median) an toàn 100%.")
    return df_train_new, df_test_new


def add_rowwise_max_min_features(df_train, df_test, feature_group):
    """
    Tính Max và Min theo chiều ngang (từng dòng) cho một nhóm các features.
    Thường dùng để tìm "Ngưỡng giới hạn trên/dưới" của một cụm 3 feature có cùng tính chất.
    """
    if len(feature_group) < 2:
        raise ValueError("❌ LỖI: Cần truyền vào ít nhất 2 features để so sánh!")
        
    print(f"📐 Đang tính Max/Min ngang cho bộ tính năng: {feature_group}...")
    
    # Rút gọn tên các cột để làm tên feature mới (cho đỡ dài dòng)
    # Ví dụ: ['feature_am', 'feature_cf', 'feature_u'] -> 'am_cf_u'
    short_names = [f.replace("feature_", "") for f in feature_group]
    group_suffix = "_".join(short_names)
    
    max_name = f"row_max_{group_suffix}"
    min_name = f"row_min_{group_suffix}"
    
    # 1. BIỂU THỨC TÍNH TOÁN NGANG (HORIZONTAL)
    expr_max = pl.max_horizontal(feature_group).alias(max_name)
    expr_min = pl.min_horizontal(feature_group).alias(min_name)
    
    # 2. VŨ KHÍ BÍ MẬT: ĐỘ RỘNG BIÊN ĐỘ (Spread/Range ngang)
    # Lấy Max ngang trừ Min ngang xem 3 thằng này đang "đồng thuận" hay đang "rẽ nhánh"
    spread_name = f"row_spread_{group_suffix}"
    expr_spread = (pl.max_horizontal(feature_group) - pl.min_horizontal(feature_group)).alias(spread_name)
    
    # Thực thi ép xung
    print("⚙️ Đang ép xung xử lý bằng Polars...")
    df_train_new = df_train.with_columns([expr_max, expr_min, expr_spread])
    df_test_new  = df_test.with_columns([expr_max, expr_min, expr_spread])
    
    print(f"✅ HOÀN TẤT! Đã đắp thêm 3 siêu tính năng:")
    print(f"   1. {max_name}")
    print(f"   2. {min_name}")
    print(f"   3. {spread_name}")
    
    return df_train_new, df_test_new

# =====================================================================
# CÁCH GỌI HÀM:
# =====================================================================
# Bác chọn ra bộ 3 feature đưa vào list:
# bo_3_feature = ["feature_am", "feature_cf", "feature_u"]
# df, df_test = add_rowwise_max_min_features(df, df_test, bo_3_feature)

def drop_useless_features_both(df_train, df_test, features_to_drop):
    """
    Xóa hàng loạt Feature khỏi CẢ 2 tập Train và Test cùng một lúc.
    Đảm bảo schema (cấu trúc cột) của 2 tập luôn đồng bộ hoàn hảo.
    """
    print(f"⚔️ ĐANG KÍCH HOẠT MÁY CHÉM KÉP: Mục tiêu {len(features_to_drop)} features...")
    
    # --- Hàm xử lý chém an toàn cho từng bảng ---
    def _drop_safe(df, df_name):
        if df is None:
            return None
            
        is_polars = isinstance(df, pl.DataFrame)
        existing_cols = df.columns
        
        # Chỉ lấy những cột thực sự tồn tại trong bảng này
        cols_to_act_drop = [col for col in features_to_drop if col in existing_cols]
        
        if len(cols_to_act_drop) == 0:
            print(f"   [{df_name}] êm ru, không có feature rác nào để xóa.")
            return df
            
        # Ra tay chém
        if is_polars:
            df_cleaned = df.drop(cols_to_act_drop)
        else:
            df_cleaned = df.drop(columns=cols_to_act_drop)
            
        print(f"   [{df_name}] Đã chém bay {len(cols_to_act_drop)} cột | Còn lại: {len(df_cleaned.columns)} cột.")
        return df_cleaned

    # --- Thực thi trên cả 2 tập ---
    df_train_clean = _drop_safe(df_train, "Train Set")
    df_test_clean  = _drop_safe(df_test, "Test Set")
    
    print("✅ ĐÃ ĐỒNG BỘ XONG CẤU TRÚC 2 TẬP!")
    
    return df_train_clean, df_test_clean

def process_data(df, df_test):
    df, df_test = fill_data(df, df_test)
    
    df = reduce_memory_usage(df)
    df_test = reduce_memory_usage(df_test)

    df, df_test = smooth_outliers_hampel(df, df_test, ['feature_al', 'feature_x', 'feature_n','feature_y','feature_z', 
                                            'feature_bo', 'feature_bm', 'feature_bz', 'feature_ag', 'feature_ao',
                                              'feature_af','feature_u'])

    df, df_test = create_sign_categorical_features(df, df_test, ['feature_cd', 'feature_bz'])
    df, df_test = add_sign_magnitude_to_both(df, df_test, 'feature_cd', 'feature_af')
    
    df, df_test = add_sign_magnitude_to_both(df, df_test, 'feature_v', 'feature_i')


    df, df_test = add_interactive_features(df, df_test, ['feature_af', 'feature_bz', 'feature_u'])
    df, df_test = add_interactive_features(df, df_test, ["feature_al", 'feature_am'])
    df, df_test = add_interactive_features(df, df_test, ["feature_al", 'feature_cc'])
    df, df_test = add_interactive_features(df, df_test, ["feature_az", 'feature_bz'])
    df, df_test = add_interactive_features(df, df_test, ["feature_az", 'feature_a'])
    df, df_test = add_interactive_features(df, df_test, ["feature_am", 'feature_cc'])
    df, df_test = add_interactive_features(df, df_test, ["feature_cg", 'feature_by'])
    df, df_test = add_interactive_features(df, df_test, ["feature_v", 'feature_al'])
    df, df_test = add_interactive_features(df, df_test, ["feature_v", 'feature_i'])
    
    df, df_test = add_custom_feature(df, df_test, ['feature_bz' ,'feature_bo', 'feature_v', 'feature_i'])

    df, df_test = add_cumulative(df, df_test,['inter_feature_am_mul_feature_cc'],[20] )
    
    df, df_test = add_cumulative(df, df_test, ['feature_al' ] )
    df,df_test = add_rowwise_max_min_features(df, df_test, ['feature_af', 'feature_u', 'feature_bm'] )
    df,df_test = add_rowwise_max_min_features(df, df_test, ['feature_bz', 'feature_cc', 'feature_by'] )

    
    df, df_test = drop_useless_features_both(df, df_test, ['feature_bb', 'feature_t', 'feature_o', 
                                                           'feature_s', 'feature_ba', 'feature_bc', 'feature_p', 
                                                           'feature_aw', 'feature_bm', 'feature_cc', 'feature_an', 
                                                           'feature_bo', 'feature_u', 'feature_bz', 'feature_ap', 'feature_ae', 'feature_q'])

    return df, df_test

def add_lag_features(df, value_cols=['feature_al', 'feature_am', 'feature_cg', 'feature_by'], lags=[1, 3, 5, 10, 25]):
    df = df.sort_values(['code', 'sub_code', 'sub_category', 'horizon', 'ts_index'])
    for col in value_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby(['code', 'sub_code', 'sub_category', 'horizon'])[col].shift(lag)
    return df

def add_rolling_features(df, value_cols=['feature_al', 'feature_am'], windows=[5, 10, 20]):
    df = df.sort_values(['code', 'sub_code', 'sub_category', 'horizon', 'ts_index'])
    for col in value_cols:
        for window in windows:
            df[f'{col}_roll_mean_{window}'] = df.groupby(['code', 'sub_code', 'sub_category', 'horizon'])[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'{col}_roll_std_{window}'] = df.groupby(['code', 'sub_code', 'sub_category', 'horizon'])[col].transform(lambda x: x.rolling(window, min_periods=1).std())
    return df

def add_trend_features(df, value_cols=['feature_al', 'feature_am'], windows=[10, 20]):
    df = df.sort_values(['code', 'sub_code', 'sub_category', 'horizon', 'ts_index'])
    def rolling_slope(series, window):
        def calc_slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
        return series.rolling(window, min_periods=2).apply(calc_slope, raw=True)
    for col in value_cols:
        for window in windows:
            df[f'{col}_trend_{window}'] = df.groupby(['code', 'sub_code', 'sub_category', 'horizon'])[col].transform(lambda x: rolling_slope(x, window))
    return df

def build_enhanced_features(data, enc_stats=None):
    df = data.copy()
    if enc_stats is not None:
        for c in ['sub_category', 'sub_code']:
            # Ép kiểu sang float ngay sau khi map để thoát khỏi ràng buộc Categorical
            df[c + '_enc'] = df[c].map(enc_stats[c]).astype(float).fillna(enc_stats['global_mean'])
    df['d_al_am'] = df['feature_al'] - df['feature_am']
    df['r_al_am'] = df['feature_al'] / (df['feature_am'] + 1e-7)
    df['d_cg_by'] = df['feature_cg'] - df['feature_by']
    norm_cols = ['feature_al', 'feature_am', 'feature_cg', 'feature_by', 'd_al_am']
    for col in norm_cols:
        g = df.groupby('ts_index')[col]
        df[col + '_cs'] = (df[col] - g.transform('mean')) / (g.transform('std') + 1e-7)
    df['t_cycle'] = np.sin(2 * np.pi * df['ts_index'] / 100)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    for col in ['feature_al', 'feature_am']:
        df[f'{col}_diff_1'] = df.groupby(['code', 'sub_code', 'sub_category', 'horizon'])[col].diff(1)
        df[f'{col}_rank'] = df.groupby('ts_index')[col].rank(pct=True)
    # df = df.fillna(0)
    return df

def get_feature_columns(df):
    exclude_cols = {'id', 'code', 'sub_code', 'sub_category', 'horizon', 'ts_index', 'weight', 'y_target'}
    return [c for c in df.columns if c not in exclude_cols]


def weighted_rmse_score(y_target, y_pred, w):
    y_target = np.array(y_target)
    y_pred = np.array(y_pred)
    w = np.array(w)
    denom = np.sum(w * (y_target ** 2))
    if denom <= 0:
        return 0.0
    numerator = np.sum(w * ((y_target - y_pred) ** 2))
    ratio = numerator / denom
    return float(np.sqrt(1.0 - np.clip(ratio, 0.0, 1.0)))

print('Metric ready')

print('Computing statistics...')
temp = pd.read_parquet(TRAIN_PATH, columns=['sub_category', 'sub_code', 'y_target', 'ts_index'])
train_only = temp[temp.ts_index <= VAL_THRESHOLD]
train_stats = {
    'sub_category': train_only.groupby('sub_category')['y_target'].mean().to_dict(),
    'sub_code': train_only.groupby('sub_code')['y_target'].mean().to_dict(),
    'global_mean': train_only['y_target'].mean()
}
del temp, train_only
gc.collect()
print('Statistics computed')

# == Markdown Cell ==
# ## KEY CHANGE: 5-Seed Ensemble (was 2)

# Cấu hình mục tiêu Bảo đã đưa ra
TARGET_FEATURES = {1: 197, 3: 177, 10: 197, 25: 217} 

def train_single_horizon(horizon):
    print(f'\n🚀 Horizon {horizon} | Fast Mode (Target: {TARGET_FEATURES.get(horizon, "Full")})')
    
    # 1. Load data & Downcast ngay lập tức
    tr_pl = pl.read_parquet(TRAIN_PATH).filter(pl.col("horizon") == horizon)
    te_pl = pl.read_parquet(TEST_PATH).filter(pl.col("horizon") == horizon)
    tr_pl, te_pl = process_data(tr_pl, te_pl)
    tr_df, te_df = tr_pl.to_pandas(), te_pl.to_pandas()
    del tr_pl, te_pl; gc.collect()
    
    tr_df = build_enhanced_features(tr_df, train_stats)
    te_df = build_enhanced_features(te_df, train_stats)
    initial_features = get_feature_columns(tr_df)
    
    # Chia mask cố định
    fit_mask = tr_df.ts_index <= VAL_THRESHOLD
    val_mask = tr_df.ts_index > VAL_THRESHOLD
    
    target_n = TARGET_FEATURES.get(horizon, len(initial_features))
    selected_features = initial_features

    # --- PASS 1: RANKING (Nếu cần loại bỏ feature) ---
    if target_n < len(initial_features):
        print(f"🔍 Đang xếp hạng {len(initial_features)} features...")
        rank_mdl = lgb.LGBMRegressor(**LGB_PARAMS, random_state=42) # Train nhanh
        rank_mdl.fit(
            tr_df.loc[fit_mask, initial_features], tr_df.loc[fit_mask, 'y_target'],
            eval_set=[(tr_df.loc[val_mask, initial_features], tr_df.loc[val_mask, 'y_target'])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        # Lấy Top N features mạnh nhất
        imp_df = pd.DataFrame({'f': initial_features, 'g': rank_mdl.booster_.feature_importance(importance_type='gain')})
        selected_features = imp_df.sort_values('g', ascending=False).head(target_n)['f'].tolist()
        print(f"✅ Đã chọn được {len(selected_features)} features tinh túy.")
        del rank_mdl; gc.collect()

    # --- PASS 2: FINAL ENSEMBLE (5 SEEDS) ---
    val_pred = np.zeros(len(tr_df.loc[val_mask]))
    tst_pred = np.zeros(len(te_df))
    seeds = [42, 2026, 1232006, 3107, 2709]
    
    for seed in seeds:
        mdl = lgb.LGBMRegressor(**LGB_PARAMS, random_state=seed)
        mdl.fit(
            tr_df.loc[fit_mask, selected_features], tr_df.loc[fit_mask, 'y_target'],
            sample_weight=tr_df.loc[fit_mask, 'weight'],
            eval_set=[(tr_df.loc[val_mask, selected_features], tr_df.loc[val_mask, 'y_target'])],
            eval_sample_weight=[tr_df.loc[val_mask, 'weight']],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        )
        val_pred += mdl.predict(tr_df.loc[val_mask, selected_features]) / len(seeds)
        tst_pred += mdl.predict(te_df[selected_features]) / len(seeds)

    final_score = weighted_rmse_score(tr_df.loc[val_mask, 'y_target'], val_pred, tr_df.loc[val_mask, 'weight'])
    test_ids = te_df['id'].values
    
    print(f"💎 Horizon {horizon} Final Score: {final_score:.5f}")
    return tst_pred, test_ids, final_score

from joblib import Parallel, delayed


results = Parallel(n_jobs=2, backend="loky")(
    delayed(train_single_horizon)(hz) for hz in FORECAST_WINDOWS
)

# 2. Thu thập kết quả
test_outputs = []
validation_scores = {}

for i, hz in enumerate(FORECAST_WINDOWS):
    tst_pred, test_ids, val_score = results[i]
    test_outputs.append(pd.DataFrame({'id': test_ids, 'prediction': tst_pred}))
    validation_scores[hz] = val_score

print('\n✅ All horizons complete with Parallel Processing!')
# test_outputs = []
# validation_scores = {}

# for hz in FORECAST_WINDOWS:
#     tst_pred, test_ids, val_score = train_single_horizon(hz)
#     test_outputs.append(pd.DataFrame({'id': test_ids, 'prediction': tst_pred}))
#     validation_scores[hz] = val_score

# print('All horizons complete')

submission = pd.concat(test_outputs, ignore_index=True)
submission.to_csv('submission.csv', index=False)
print(f'Saved {len(submission):,} predictions')
print(submission.head(10))

print(f"{'Horizon':<10} {'Val Score':<12}")
print('-'*25)
for hz in sorted(validation_scores.keys()):
    print(f'{hz:<10} {validation_scores[hz]:<12.5f}')

avg_val = np.mean(list(validation_scores.values()))
print('-'*25)
print('SUBMISSION done')

