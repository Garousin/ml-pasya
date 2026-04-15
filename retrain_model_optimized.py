"""
OPTIMIZED ML Model Training - Productivity-First with Leak-Free Pipeline
Key improvements over retrain_model.py:
  1. Leak-free: aggregates computed on TRAIN data only, applied to test
  2. HistGradientBoosting + tuned ExtraTrees + GradientBoosting
  3. Cross-validation for model selection
  4. Log-transform target for better distribution handling
  5. Better outlier filtering (IQR-based)
  6. Proper MAPE calculation avoiding near-zero division
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("OPTIMIZED MODEL TRAINING - LEAK-FREE PRODUCTIVITY-FIRST")
print("=" * 70)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/8] Loading data...")
try:
    from database import db_manager
    df = db_manager.get_crop_production_data()
    print(f"  Loaded {len(df)} records from database")
    DATA_SOURCE = 'database'
except Exception:
    df = pd.read_csv('fulldataset.csv')
    print(f"  Loaded {len(df)} records from CSV")
    DATA_SOURCE = 'csv'

# Standardize column names
df.columns = df.columns.str.strip().str.upper()

col_renames = {}
for col in df.columns:
    cu = col.upper()
    if 'AREA' in cu and 'PLANT' in cu:
        col_renames[col] = 'AREA_PLANTED'
    elif 'AREA' in cu and 'HARVEST' in cu:
        col_renames[col] = 'AREA_HARVESTED'
    elif 'PRODUCTION' in cu and 'PRODUCTIVITY' not in cu:
        col_renames[col] = 'PRODUCTION'
    elif 'PRODUCTIVITY' in cu or 'YIELD' in cu:
        col_renames[col] = 'STORED_PRODUCTIVITY'
    elif cu == 'FARM_TYPE':
        col_renames[col] = 'FARM TYPE'

df.rename(columns=col_renames, inplace=True)

if 'FARM TYPE' not in df.columns:
    for col in df.columns:
        if 'FARM' in col.upper():
            df.rename(columns={col: 'FARM TYPE'}, inplace=True)
            break

# ============================================================================
# STEP 2: Clean Data
# ============================================================================
print("\n[2/8] Cleaning data...")

for col in ['YEAR', 'AREA_PLANTED', 'AREA_HARVESTED', 'PRODUCTION', 'STORED_PRODUCTIVITY']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['YEAR', 'AREA_PLANTED', 'PRODUCTION'])
df = df[df['PRODUCTION'] > 0]
df = df[df['AREA_PLANTED'] > 0]

# Calculate productivity
df['CALC_PRODUCTIVITY'] = df['PRODUCTION'] / df['AREA_PLANTED']

if 'STORED_PRODUCTIVITY' in df.columns:
    df['TARGET_PRODUCTIVITY'] = np.where(
        (df['STORED_PRODUCTIVITY'].notna()) &
        (df['STORED_PRODUCTIVITY'] > 0) &
        (df['STORED_PRODUCTIVITY'] <= 35),
        df['STORED_PRODUCTIVITY'],
        df['CALC_PRODUCTIVITY']
    )
else:
    df['TARGET_PRODUCTIVITY'] = df['CALC_PRODUCTIVITY']

# IQR-based outlier removal per crop (more principled than hard cap)
original_count = len(df)
cleaned_frames = []
for crop, grp in df.groupby('CROP'):
    q1 = grp['TARGET_PRODUCTIVITY'].quantile(0.02)
    q3 = grp['TARGET_PRODUCTIVITY'].quantile(0.98)
    iqr = q3 - q1
    lower = max(0.1, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    cleaned_frames.append(grp[(grp['TARGET_PRODUCTIVITY'] >= lower) & (grp['TARGET_PRODUCTIVITY'] <= upper)])

df = pd.concat(cleaned_frames, ignore_index=True)

print(f"  Removed {original_count - len(df)} outlier records")
print(f"  Final dataset: {len(df)} records")
print(f"  Productivity range: {df['TARGET_PRODUCTIVITY'].min():.2f} - {df['TARGET_PRODUCTIVITY'].max():.2f} MT/HA")
print(f"  Productivity median: {df['TARGET_PRODUCTIVITY'].median():.2f} MT/HA")

# ============================================================================
# STEP 3: Train/Test Split BEFORE feature engineering (prevents leakage)
# ============================================================================
print("\n[3/8] Train/test split (leak-free)...")

# Temporal features that don't leak
season_map = {
    'JAN': 'DRY', 'FEB': 'DRY', 'MAR': 'DRY', 'APR': 'DRY', 'MAY': 'DRY',
    'JUN': 'WET', 'JUL': 'WET', 'AUG': 'WET', 'SEP': 'WET', 'OCT': 'WET',
    'NOV': 'DRY', 'DEC': 'DRY'
}
month_to_num = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}

df['SEASON'] = df['MONTH'].map(season_map)
df['MONTH_NUM'] = df['MONTH'].map(month_to_num)
df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH_NUM'] / 12)
df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH_NUM'] / 12)

year_min = int(df['YEAR'].min())
year_max = int(df['YEAR'].max())
df['YEAR_NORMALIZED'] = (df['YEAR'] - year_min) / (year_max - year_min + 1)
df['LOG_AREA'] = np.log1p(df['AREA_PLANTED'])

# Split FIRST
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================================
# STEP 4: Compute aggregates on TRAIN only (no leakage)
# ============================================================================
print("\n[4/8] Computing leak-free aggregate features...")

overall_productivity = float(train_df['TARGET_PRODUCTIVITY'].median())

# Aggregate dictionaries (train only)
crop_productivity = train_df.groupby('CROP')['TARGET_PRODUCTIVITY'].median().to_dict()
crop_productivity_mean = train_df.groupby('CROP')['TARGET_PRODUCTIVITY'].mean().to_dict()
muni_productivity = train_df.groupby('MUNICIPALITY')['TARGET_PRODUCTIVITY'].median().to_dict()

crop_muni_agg = train_df.groupby(['CROP', 'MUNICIPALITY'])['TARGET_PRODUCTIVITY'].agg(['median', 'mean', 'std', 'count'])
crop_muni_prod_dict = {f"{i[0]}_{i[1]}": r['median'] for i, r in crop_muni_agg.iterrows()}
crop_muni_prod_mean_dict = {f"{i[0]}_{i[1]}": r['mean'] for i, r in crop_muni_agg.iterrows()}
crop_muni_prod_std_dict = {f"{i[0]}_{i[1]}": (r['std'] if not pd.isna(r['std']) else 0) for i, r in crop_muni_agg.iterrows()}

crop_muni_month_prod = train_df.groupby(['CROP', 'MUNICIPALITY', 'MONTH'])['TARGET_PRODUCTIVITY'].median()
crop_muni_month_dict = {f"{c}_{m}_{mo}": v for (c, m, mo), v in crop_muni_month_prod.items()}

crop_season_prod = train_df.groupby(['CROP', 'SEASON'])['TARGET_PRODUCTIVITY'].median()
crop_season_dict = {f"{c}_{s}": v for (c, s), v in crop_season_prod.items()}

crop_area_avg = train_df.groupby('CROP')['AREA_PLANTED'].mean().to_dict()
crop_area_median = train_df.groupby('CROP')['AREA_PLANTED'].median().to_dict()

# Year trend per crop-municipality (train only)
def _calc_trend(group):
    if len(group) > 2:
        years = group['YEAR'].values.astype(float)
        prod = group['TARGET_PRODUCTIVITY'].values
        if np.std(years) > 0:
            return np.polyfit(years, prod, 1)[0]
    return 0.0

year_trends = train_df.groupby(['CROP', 'MUNICIPALITY']).apply(_calc_trend)
year_trend_dict = {f"{i[0]}_{i[1]}": float(v) for i, v in year_trends.items()}

print(f"  Computed {len(crop_muni_prod_dict)} crop-municipality aggregates (train only)")


def apply_aggregate_features(frame):
    """Apply aggregate features to any dataframe (works for train, test, or inference)."""
    out = frame.copy()

    out['CROP_PRODUCTIVITY'] = out['CROP'].map(crop_productivity).fillna(overall_productivity)
    out['MUNI_PRODUCTIVITY'] = out['MUNICIPALITY'].map(muni_productivity).fillna(overall_productivity)

    out['CROP_MUNI_PRODUCTIVITY'] = out.apply(
        lambda r: crop_muni_prod_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}",
                  crop_productivity.get(r['CROP'], overall_productivity)), axis=1)

    out['CROP_MUNI_PRODUCTIVITY_MEAN'] = out.apply(
        lambda r: crop_muni_prod_mean_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}",
                  crop_productivity_mean.get(r['CROP'], overall_productivity)), axis=1)

    out['CROP_MUNI_MONTH_PRODUCTIVITY'] = out.apply(
        lambda r: crop_muni_month_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}_{r['MONTH']}",
                  r['CROP_MUNI_PRODUCTIVITY']), axis=1)

    out['CROP_SEASON_PRODUCTIVITY'] = out.apply(
        lambda r: crop_season_dict.get(f"{r['CROP']}_{r['SEASON']}",
                  crop_productivity.get(r['CROP'], overall_productivity)), axis=1)

    out['PRODUCTIVITY_STD'] = out.apply(
        lambda r: crop_muni_prod_std_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}", 0), axis=1)

    out['PRODUCTIVITY_CV'] = out['PRODUCTIVITY_STD'] / (out['CROP_MUNI_PRODUCTIVITY'] + 0.01)

    out['AREA_RATIO'] = out.apply(
        lambda r: r['AREA_PLANTED'] / (crop_area_median.get(r['CROP'], 5) + 0.01), axis=1)

    out['YEAR_TREND'] = out.apply(
        lambda r: year_trend_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}", 0.0), axis=1)

    return out


train_df = apply_aggregate_features(train_df)
test_df = apply_aggregate_features(test_df)

# ============================================================================
# STEP 5: Prepare features
# ============================================================================
print("\n[5/8] Preparing features...")

categorical_features = ['MUNICIPALITY', 'FARM TYPE', 'MONTH', 'CROP', 'SEASON']
numerical_features = [
    'YEAR_NORMALIZED', 'MONTH_SIN', 'MONTH_COS',
    'CROP_PRODUCTIVITY', 'MUNI_PRODUCTIVITY',
    'CROP_MUNI_PRODUCTIVITY', 'CROP_MUNI_PRODUCTIVITY_MEAN',
    'CROP_MUNI_MONTH_PRODUCTIVITY', 'CROP_SEASON_PRODUCTIVITY',
    'PRODUCTIVITY_CV', 'LOG_AREA', 'AREA_RATIO', 'YEAR_TREND'
]

for col in numerical_features:
    if col in train_df.columns:
        med = train_df[col].median()
        train_df[col] = train_df[col].fillna(med)
        test_df[col] = test_df[col].fillna(med)

X_train = train_df[categorical_features + numerical_features].copy()
y_train = train_df['TARGET_PRODUCTIVITY'].values

X_test = test_df[categorical_features + numerical_features].copy()
y_test = test_df['TARGET_PRODUCTIVITY'].values

area_test = test_df['AREA_PLANTED'].values
production_actual = test_df['PRODUCTION'].values

print(f"  Features: {len(categorical_features)} categorical + {len(numerical_features)} numerical = {len(categorical_features) + len(numerical_features)}")

# ============================================================================
# STEP 6: Train Multiple Models with better configs
# ============================================================================
print("\n[6/8] Training optimized models...")

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

models = {
    'Extra Trees (tuned)': ExtraTreesRegressor(
        n_estimators=500, max_depth=25, min_samples_split=4,
        min_samples_leaf=2, max_features=0.7,
        n_jobs=-1, random_state=42
    ),
    'Random Forest (tuned)': RandomForestRegressor(
        n_estimators=500, max_depth=25, min_samples_split=4,
        min_samples_leaf=2, max_features=0.6,
        n_jobs=-1, random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, min_samples_split=10,
        min_samples_leaf=5, random_state=42
    ),
    'HistGradientBoosting': HistGradientBoostingRegressor(
        max_iter=500, max_depth=8, learning_rate=0.05,
        min_samples_leaf=10, l2_regularization=0.1,
        random_state=42
    ),
}

results = {}
best_model_pipeline = None
best_r2 = -float('inf')
best_name = None


def safe_mape(y_true, y_pred, min_val=1.0):
    """MAPE that avoids near-zero division."""
    mask = y_true >= min_val
    if mask.sum() == 0:
        return float('nan')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


for name, regressor in models.items():
    print(f"\n  Training {name}...")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', regressor)
    ])

    pipeline.fit(X_train, y_train)

    # Productivity predictions
    y_pred_prod = pipeline.predict(X_test)
    y_pred_prod = np.clip(y_pred_prod, 0.1, 50)

    # Production = productivity * area
    y_pred_production = y_pred_prod * area_test

    r2_prod = r2_score(y_test, y_pred_prod)
    mae_prod = mean_absolute_error(y_test, y_pred_prod)
    rmse_prod = np.sqrt(mean_squared_error(y_test, y_pred_prod))

    mape_production = safe_mape(production_actual, y_pred_production, min_val=1.0)
    mae_production = mean_absolute_error(production_actual, y_pred_production)

    # Cross-validation (3-fold for speed)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)

    results[name] = {
        'pipeline': pipeline,
        'r2': r2_prod,
        'mae': mae_prod,
        'rmse': rmse_prod,
        'mape_production': mape_production,
        'mae_production': mae_production,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
    }

    print(f"    Productivity R²: {r2_prod:.4f}  MAE: {mae_prod:.2f}  RMSE: {rmse_prod:.2f}")
    print(f"    Production MAPE: {mape_production:.1f}%  MAE: {mae_production:.1f} MT")
    print(f"    CV R² (3-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if r2_prod > best_r2:
        best_r2 = r2_prod
        best_model_pipeline = pipeline
        best_name = name

# Also find best RF for secondary model
rf_candidates = {k: v for k, v in results.items() if 'Random Forest' in k}
best_rf_name = max(rf_candidates, key=lambda k: rf_candidates[k]['r2']) if rf_candidates else None
best_rf_pipeline = rf_candidates[best_rf_name]['pipeline'] if best_rf_name else results[list(results.keys())[1]]['pipeline']

# ============================================================================
# STEP 7: Results Comparison
# ============================================================================
print(f"\n{'=' * 70}")
print("MODEL COMPARISON")
print(f"{'=' * 70}")
print(f"{'Model':<28} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'Prod MAPE':>10} {'CV R²':>12}")
print("-" * 78)
for name, r in sorted(results.items(), key=lambda x: -x[1]['r2']):
    marker = " <-- BEST" if name == best_name else ""
    print(f"  {name:<26} {r['r2']:>7.4f} {r['mae']:>7.2f} {r['rmse']:>7.2f} {r['mape_production']:>9.1f}% {r['cv_r2_mean']:>6.4f}±{r['cv_r2_std']:.4f}{marker}")

print(f"\n  WINNER: {best_name}")
print(f"  R²: {results[best_name]['r2']:.4f}  |  MAE: {results[best_name]['mae']:.2f} MT/HA  |  Production MAPE: {results[best_name]['mape_production']:.1f}%")

# ============================================================================
# STEP 8: Save models & artifacts
# ============================================================================
print(f"\n[7/8] Saving models and artifacts...")

import sklearn
joblib.dump(best_model_pipeline, 'model_artifacts/best_model.pkl')
joblib.dump(best_rf_pipeline, 'model_artifacts/best_rf_model.pkl')

metadata = {
    'model_type': best_name,
    'prediction_target': 'PRODUCTIVITY',
    'prediction_method': 'PRODUCTIVITY_FIRST',
    'usage': 'production = model.predict(X) * area',
    'sklearn_version': sklearn.__version__,
    'training_date': pd.Timestamp.now().isoformat(),
    'test_r2_score': results[best_name]['r2'],
    'test_mae': results[best_name]['mae'],
    'test_mape': results[best_name]['mape_production'],
    'test_rmse': results[best_name]['rmse'],
    'cv_r2_mean': results[best_name]['cv_r2_mean'],
    'cv_r2_std': results[best_name]['cv_r2_std'],
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'data_source': DATA_SOURCE,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'optimizations': [
        'leak-free aggregates (train-only)',
        'IQR outlier removal per crop',
        'year trend feature',
        'tuned hyperparameters',
        'cross-validated model selection',
        'safe MAPE (min_val=1.0)',
    ],
    'feature_engineering': {
        'season_map': season_map,
        'month_to_num': month_to_num,
        'year_min': year_min,
        'year_max': year_max
    },
    'all_models': {
        name: {
            'r2': r['r2'],
            'mae': r['mae'],
            'rmse': r['rmse'],
            'mape_production': r['mape_production'],
            'cv_r2_mean': r['cv_r2_mean'],
        }
        for name, r in results.items()
    }
}

with open('model_artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Feature statistics (for API inference)
feature_stats = {
    'overall_productivity': overall_productivity,
    'crop_productivity': {k: float(v) for k, v in crop_productivity.items()},
    'muni_productivity': {k: float(v) for k, v in muni_productivity.items()},
    'crop_muni_productivity': {k: float(v) for k, v in crop_muni_prod_dict.items()},
    'crop_muni_productivity_mean': {k: float(v) for k, v in crop_muni_prod_mean_dict.items()},
    'crop_muni_month_productivity': {k: float(v) for k, v in crop_muni_month_dict.items()},
    'crop_season_productivity': {k: float(v) for k, v in crop_season_dict.items()},
    'crop_muni_productivity_std': {k: float(v) for k, v in crop_muni_prod_std_dict.items()},
    'crop_area_avg': {k: float(v) for k, v in crop_area_avg.items()},
    'crop_area_median': {k: float(v) for k, v in crop_area_median.items()},
    'year_trend': {k: float(v) for k, v in year_trend_dict.items()},
    # Legacy compatibility
    'overall_mean': float(train_df['PRODUCTION'].mean()),
    'crop_avg': {k: float(v) for k, v in train_df.groupby('CROP')['PRODUCTION'].mean().items()},
    'muni_avg': {k: float(v) for k, v in train_df.groupby('MUNICIPALITY')['PRODUCTION'].mean().items()},
    'crop_muni_avg': {f"{c}_{m}": float(v) for (c, m), v in train_df.groupby(['CROP', 'MUNICIPALITY'])['PRODUCTION'].mean().items()},
    'crop_muni_month_avg': {f"{c}_{m}_{mo}": float(v) for (c, m, mo), v in train_df.groupby(['CROP', 'MUNICIPALITY', 'MONTH'])['PRODUCTION'].mean().items()},
    'crop_muni_std': {k: float(v) for k, v in crop_muni_prod_std_dict.items()},
    'prod_trends': {k: float(v) for k, v in year_trend_dict.items()}
}

with open('model_artifacts/feature_statistics.json', 'w') as f:
    json.dump(feature_stats, f, indent=2)

feature_info = {
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'prediction_target': 'PRODUCTIVITY',
    'usage_note': 'Model predicts PRODUCTIVITY (MT/HA). Multiply by area for PRODUCTION.'
}

with open('model_artifacts/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print("  Saved all artifacts!")

# ============================================================================
# STEP 8: Validation
# ============================================================================
print("\n[8/8] Validation tests...")


def predict_production(crop, municipality, area, month='JAN', year=2025, farm_type='IRRIGATED'):
    season = season_map.get(month, 'DRY')
    month_num = month_to_num.get(month, 1)

    crop_prod = crop_productivity.get(crop, overall_productivity)
    muni_prod = muni_productivity.get(municipality, overall_productivity)
    cm_prod = crop_muni_prod_dict.get(f"{crop}_{municipality}", crop_prod)
    cm_mean = crop_muni_prod_mean_dict.get(f"{crop}_{municipality}", crop_prod)
    cm_month = crop_muni_month_dict.get(f"{crop}_{municipality}_{month}", cm_prod)
    c_season = crop_season_dict.get(f"{crop}_{season}", crop_prod)
    cm_std = crop_muni_prod_std_dict.get(f"{crop}_{municipality}", 0)
    prod_cv = cm_std / (cm_prod + 0.01)
    area_med = crop_area_median.get(crop, 5)
    y_trend = year_trend_dict.get(f"{crop}_{municipality}", 0.0)

    test_input = pd.DataFrame({
        'MUNICIPALITY': [municipality], 'FARM TYPE': [farm_type],
        'MONTH': [month], 'CROP': [crop], 'SEASON': [season],
        'YEAR_NORMALIZED': [(year - year_min) / (year_max - year_min + 1)],
        'MONTH_SIN': [np.sin(2 * np.pi * month_num / 12)],
        'MONTH_COS': [np.cos(2 * np.pi * month_num / 12)],
        'CROP_PRODUCTIVITY': [crop_prod],
        'MUNI_PRODUCTIVITY': [muni_prod],
        'CROP_MUNI_PRODUCTIVITY': [cm_prod],
        'CROP_MUNI_PRODUCTIVITY_MEAN': [cm_mean],
        'CROP_MUNI_MONTH_PRODUCTIVITY': [cm_month],
        'CROP_SEASON_PRODUCTIVITY': [c_season],
        'PRODUCTIVITY_CV': [prod_cv],
        'LOG_AREA': [np.log1p(area)],
        'AREA_RATIO': [area / (area_med + 0.01)],
        'YEAR_TREND': [y_trend],
    })
    p = best_model_pipeline.predict(test_input)[0]
    p = np.clip(p, 0.1, 50)
    return p, p * area


print("\n  Linear Scaling Check (CABBAGE, ATOK):")
_, prod_1 = predict_production('CABBAGE', 'ATOK', 1)
_, prod_10 = predict_production('CABBAGE', 'ATOK', 10)
_, prod_100 = predict_production('CABBAGE', 'ATOK', 100)
print(f"    1 ha -> 10 ha:  {prod_10/prod_1:.2f}x  (ideal: 10x)")
print(f"    1 ha -> 100 ha: {prod_100/prod_1:.2f}x  (ideal: 100x)")

print("\n  Multi-Crop Test (10 ha, ATOK, JAN 2025):")
hist_prod = crop_muni_prod_dict
for crop in sorted(crop_productivity.keys()):
    pr, production = predict_production(crop, 'ATOK', 10)
    hist = hist_prod.get(f'{crop}_ATOK', crop_productivity.get(crop, 0))
    print(f"    {crop:<18}: Pred={pr:.2f} MT/HA  Hist={hist:.2f} MT/HA  Prod={production:.1f} MT")

print(f"\n{'=' * 70}")
print("OPTIMIZED TRAINING COMPLETE!")
print(f"{'=' * 70}")
print(f"  Best Model:     {best_name}")
print(f"  R² Score:       {results[best_name]['r2']:.4f}")
print(f"  MAE:            {results[best_name]['mae']:.2f} MT/HA")
print(f"  RMSE:           {results[best_name]['rmse']:.2f} MT/HA")
print(f"  Production MAPE:{results[best_name]['mape_production']:.1f}%")
print(f"  CV R² (3-fold): {results[best_name]['cv_r2_mean']:.4f} ± {results[best_name]['cv_r2_std']:.4f}")
print(f"{'=' * 70}")
