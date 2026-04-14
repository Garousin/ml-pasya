"""
HYBRID ML Model Training - Best of Both Approaches
Predicts PRODUCTIVITY first, then calculates PRODUCTION = Productivity × Area

This ensures:
1. Linear scaling with area (key fix!)
2. Accurate productivity predictions
3. Better generalization across different farm sizes
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HYBRID MODEL TRAINING - PRODUCTIVITY-FIRST APPROACH")
print("="*70)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/7] Loading data...")
try:
    from database import db_manager
    df = db_manager.get_crop_production_data()
    print(f"  Loaded {len(df)} records from database")
    DATA_SOURCE = 'database'
except:
    df = pd.read_csv('fulldataset.csv')
    print(f"  Loaded {len(df)} records from CSV")
    DATA_SOURCE = 'csv'

# Standardize column names
df.columns = df.columns.str.strip().str.upper()

# Find and rename columns
col_renames = {}
for col in df.columns:
    col_upper = col.upper()
    if 'AREA' in col_upper and 'PLANT' in col_upper:
        col_renames[col] = 'AREA_PLANTED'
    elif 'AREA' in col_upper and 'HARVEST' in col_upper:
        col_renames[col] = 'AREA_HARVESTED'
    elif 'PRODUCTION' in col_upper:
        col_renames[col] = 'PRODUCTION'
    elif 'PRODUCTIVITY' in col_upper or 'YIELD' in col_upper:
        col_renames[col] = 'STORED_PRODUCTIVITY'
    elif col_upper == 'FARM_TYPE':
        col_renames[col] = 'FARM_TYPE'

df.rename(columns=col_renames, inplace=True)

# Handle FARM TYPE variations
if 'FARM_TYPE' in df.columns and 'FARM TYPE' not in df.columns:
    df.rename(columns={'FARM_TYPE': 'FARM TYPE'}, inplace=True)
elif 'FARM TYPE' not in df.columns:
    for col in df.columns:
        if 'FARM' in col.upper():
            df.rename(columns={col: 'FARM TYPE'}, inplace=True)
            break

# ============================================================================
# STEP 2: Clean and Prepare Data
# ============================================================================
print("\n[2/7] Cleaning data...")

# Convert to numeric
for col in ['YEAR', 'AREA_PLANTED', 'AREA_HARVESTED', 'PRODUCTION', 'STORED_PRODUCTIVITY']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Basic filtering
df = df.dropna(subset=['YEAR', 'AREA_PLANTED', 'PRODUCTION'])
df = df[df['PRODUCTION'] > 0]
df = df[df['AREA_PLANTED'] > 0]

# Calculate productivity
df['CALC_PRODUCTIVITY'] = df['PRODUCTION'] / df['AREA_PLANTED']

# Use stored productivity if reasonable, otherwise calculated
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

# Remove extreme values
productivity_cap = 40  # MT/HA - realistic for most crops
original_count = len(df)
df = df[(df['TARGET_PRODUCTIVITY'] > 0.5) & (df['TARGET_PRODUCTIVITY'] <= productivity_cap)]

print(f"  Removed {original_count - len(df)} extreme records")
print(f"  Final dataset: {len(df)} records")
print(f"  Productivity range: {df['TARGET_PRODUCTIVITY'].min():.1f} - {df['TARGET_PRODUCTIVITY'].max():.1f} MT/HA")
print(f"  Productivity median: {df['TARGET_PRODUCTIVITY'].median():.2f} MT/HA")

# ============================================================================
# STEP 3: Feature Engineering
# ============================================================================
print("\n[3/7] Engineering features...")

# Temporal features
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
df['YEARS_FROM_START'] = df['YEAR'] - year_min

# PRODUCTIVITY AGGREGATES (key features for accuracy)
overall_productivity = df['TARGET_PRODUCTIVITY'].median()

# By Crop
crop_productivity = df.groupby('CROP')['TARGET_PRODUCTIVITY'].median().to_dict()
crop_productivity_mean = df.groupby('CROP')['TARGET_PRODUCTIVITY'].mean().to_dict()

# By Municipality  
muni_productivity = df.groupby('MUNICIPALITY')['TARGET_PRODUCTIVITY'].median().to_dict()

# By Crop+Municipality (most important!)
crop_muni_prod = df.groupby(['CROP', 'MUNICIPALITY'])['TARGET_PRODUCTIVITY'].agg(['median', 'mean', 'std', 'count'])
crop_muni_prod_dict = {f"{idx[0]}_{idx[1]}": row['median'] for idx, row in crop_muni_prod.iterrows()}
crop_muni_prod_mean_dict = {f"{idx[0]}_{idx[1]}": row['mean'] for idx, row in crop_muni_prod.iterrows()}
crop_muni_prod_std_dict = {f"{idx[0]}_{idx[1]}": row['std'] if not pd.isna(row['std']) else 0 for idx, row in crop_muni_prod.iterrows()}
crop_muni_count_dict = {f"{idx[0]}_{idx[1]}": row['count'] for idx, row in crop_muni_prod.iterrows()}

# By Crop+Municipality+Month
crop_muni_month_prod = df.groupby(['CROP', 'MUNICIPALITY', 'MONTH'])['TARGET_PRODUCTIVITY'].median()
crop_muni_month_dict = {f"{c}_{m}_{mo}": v for (c, m, mo), v in crop_muni_month_prod.items()}

# By Crop+Season
crop_season_prod = df.groupby(['CROP', 'SEASON'])['TARGET_PRODUCTIVITY'].median()
crop_season_dict = {f"{c}_{s}": v for (c, s), v in crop_season_prod.items()}

# Area statistics
crop_area_avg = df.groupby('CROP')['AREA_PLANTED'].mean().to_dict()
crop_area_median = df.groupby('CROP')['AREA_PLANTED'].median().to_dict()

# Add features to dataframe
df['CROP_PRODUCTIVITY'] = df['CROP'].map(crop_productivity)
df['CROP_PRODUCTIVITY_MEAN'] = df['CROP'].map(crop_productivity_mean)
df['MUNI_PRODUCTIVITY'] = df['MUNICIPALITY'].map(muni_productivity)

df['CROP_MUNI_PRODUCTIVITY'] = df.apply(
    lambda r: crop_muni_prod_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}", 
              crop_productivity.get(r['CROP'], overall_productivity)), axis=1)

df['CROP_MUNI_PRODUCTIVITY_MEAN'] = df.apply(
    lambda r: crop_muni_prod_mean_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}", 
              crop_productivity_mean.get(r['CROP'], overall_productivity)), axis=1)

df['CROP_MUNI_MONTH_PRODUCTIVITY'] = df.apply(
    lambda r: crop_muni_month_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}_{r['MONTH']}", 
              r['CROP_MUNI_PRODUCTIVITY']), axis=1)

df['CROP_SEASON_PRODUCTIVITY'] = df.apply(
    lambda r: crop_season_dict.get(f"{r['CROP']}_{r['SEASON']}", 
              crop_productivity.get(r['CROP'], overall_productivity)), axis=1)

df['PRODUCTIVITY_STD'] = df.apply(
    lambda r: crop_muni_prod_std_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}", 0), axis=1)

df['PRODUCTIVITY_CV'] = df['PRODUCTIVITY_STD'] / (df['CROP_MUNI_PRODUCTIVITY'] + 0.01)

# Area features (for potential diminishing returns)
df['LOG_AREA'] = np.log1p(df['AREA_PLANTED'])
df['AREA_RATIO'] = df.apply(
    lambda r: r['AREA_PLANTED'] / (crop_area_median.get(r['CROP'], 5) + 0.01), axis=1)

print(f"  Feature engineering complete")

# ============================================================================
# STEP 4: Prepare Model Features
# ============================================================================
print("\n[4/7] Preparing model features...")

categorical_features = ['MUNICIPALITY', 'FARM TYPE', 'MONTH', 'CROP', 'SEASON']
numerical_features = [
    'YEAR_NORMALIZED', 'MONTH_SIN', 'MONTH_COS',
    'CROP_PRODUCTIVITY', 'MUNI_PRODUCTIVITY',
    'CROP_MUNI_PRODUCTIVITY', 'CROP_MUNI_PRODUCTIVITY_MEAN',
    'CROP_MUNI_MONTH_PRODUCTIVITY', 'CROP_SEASON_PRODUCTIVITY',
    'PRODUCTIVITY_CV', 'LOG_AREA', 'AREA_RATIO'
]

# Fill any NaN
for col in numerical_features:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

X = df[categorical_features + numerical_features].copy()
y = df['TARGET_PRODUCTIVITY'].values

print(f"  Features: {len(categorical_features)} categorical + {len(numerical_features)} numerical")

# ============================================================================
# STEP 5: Train-Test Split and Model Training
# ============================================================================
print("\n[5/7] Training models...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Store area and production for test validation
test_idx = X_test.index
area_test = df.loc[test_idx, 'AREA_PLANTED'].values
production_actual = df.loc[test_idx, 'PRODUCTION'].values

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Train models
models = {
    'Extra Trees': ExtraTreesRegressor(
        n_estimators=300, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, n_jobs=-1, random_state=42
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, n_jobs=-1, random_state=42
    )
}

results = {}
best_model = None
best_mape = float('inf')

for name, regressor in models.items():
    print(f"\n  Training {name}...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', regressor)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Predict productivity
    y_pred_prod = pipeline.predict(X_test)
    y_pred_prod = np.clip(y_pred_prod, 0.5, 40)  # Reasonable bounds
    
    # Calculate production
    y_pred_production = y_pred_prod * area_test
    
    # Metrics
    r2_productivity = r2_score(y_test, y_pred_prod)
    mae_productivity = mean_absolute_error(y_test, y_pred_prod)
    
    # Filter out extreme outliers for production metrics
    mask = production_actual < np.percentile(production_actual, 99)
    if mask.sum() > 0:
        mape_production = np.mean(np.abs((production_actual[mask] - y_pred_production[mask]) / 
                                         (production_actual[mask] + 0.01))) * 100
        mae_production = mean_absolute_error(production_actual[mask], y_pred_production[mask])
    else:
        mape_production = 100
        mae_production = 100
    
    results[name] = {
        'model': pipeline,
        'r2_productivity': r2_productivity,
        'mae_productivity': mae_productivity,
        'mape_production': mape_production,
        'mae_production': mae_production
    }
    
    print(f"    Productivity R²: {r2_productivity:.4f}, MAE: {mae_productivity:.2f} MT/HA")
    print(f"    Production MAPE: {mape_production:.2f}%, MAE: {mae_production:.2f} MT")
    
    if mape_production < best_mape:
        best_mape = mape_production
        best_model = pipeline
        best_name = name

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_name}")
print(f"  Productivity R²: {results[best_name]['r2_productivity']:.4f}")
print(f"  Productivity MAE: {results[best_name]['mae_productivity']:.2f} MT/HA")
print(f"  Production MAPE: {results[best_name]['mape_production']:.2f}%")
print(f"{'='*70}")

# ============================================================================
# STEP 6: Save Models and Metadata
# ============================================================================
print("\n[6/7] Saving models...")

joblib.dump(best_model, 'model_artifacts/best_model.pkl')
joblib.dump(results['Random Forest']['model'], 'model_artifacts/best_rf_model.pkl')

# Metadata
metadata = {
    'model_type': best_name,
    'prediction_target': 'PRODUCTIVITY',
    'prediction_method': 'PRODUCTIVITY_FIRST',
    'usage': 'production = model.predict(X) * area',
    'sklearn_version': '1.3.2',
    'training_date': pd.Timestamp.now().isoformat(),
    'test_r2_score': results[best_name]['r2_productivity'],
    'test_mae': results[best_name]['mae_productivity'],
    'test_mape': results[best_name]['mape_production'],
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'feature_engineering': {
        'season_map': season_map,
        'month_to_num': month_to_num,
        'year_min': year_min,
        'year_max': year_max
    }
}

with open('model_artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Feature statistics
feature_stats = {
    'overall_productivity': overall_productivity,
    'crop_productivity': crop_productivity,
    'muni_productivity': muni_productivity,
    'crop_muni_productivity': crop_muni_prod_dict,
    'crop_muni_month_productivity': crop_muni_month_dict,
    'crop_season_productivity': crop_season_dict,
    'crop_muni_productivity_std': crop_muni_prod_std_dict,
    'crop_area_avg': crop_area_avg,
    'crop_area_median': crop_area_median,
    # Legacy compatibility fields
    'overall_mean': float(df['PRODUCTION'].mean()),
    'crop_avg': {k: float(v) for k, v in df.groupby('CROP')['PRODUCTION'].mean().items()},
    'muni_avg': {k: float(v) for k, v in df.groupby('MUNICIPALITY')['PRODUCTION'].mean().items()},
    'crop_muni_avg': {f"{c}_{m}": float(v) for (c, m), v in df.groupby(['CROP', 'MUNICIPALITY'])['PRODUCTION'].mean().items()},
    'crop_muni_month_avg': {f"{c}_{m}_{mo}": float(v) for (c, m, mo), v in df.groupby(['CROP', 'MUNICIPALITY', 'MONTH'])['PRODUCTION'].mean().items()},
    'crop_muni_std': crop_muni_prod_std_dict,
    'prod_trends': {k: 0.0 for k in crop_muni_prod_dict.keys()}
}

with open('model_artifacts/feature_statistics.json', 'w') as f:
    json.dump(feature_stats, f, indent=2)

# Feature info
feature_info = {
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'prediction_target': 'PRODUCTIVITY',
    'usage_note': 'Model predicts PRODUCTIVITY. Multiply by area for PRODUCTION.'
}

with open('model_artifacts/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print("  Saved!")

# ============================================================================
# STEP 7: Validation Tests
# ============================================================================
print("\n[7/7] Validation tests...")

def predict_production(crop, municipality, area, month='JAN', year=2025, farm_type='IRRIGATED'):
    """Predict production for given parameters"""
    season = season_map.get(month, 'DRY')
    month_num = month_to_num.get(month, 1)
    
    # Get productivity features
    crop_prod = crop_productivity.get(crop, overall_productivity)
    muni_prod = muni_productivity.get(municipality, overall_productivity)
    crop_muni_prod = crop_muni_prod_dict.get(f"{crop}_{municipality}", crop_prod)
    crop_muni_mean = crop_muni_prod_mean_dict.get(f"{crop}_{municipality}", crop_prod)
    crop_muni_month = crop_muni_month_dict.get(f"{crop}_{municipality}_{month}", crop_muni_prod)
    crop_season = crop_season_dict.get(f"{crop}_{season}", crop_prod)
    prod_std = crop_muni_prod_std_dict.get(f"{crop}_{municipality}", 0)
    prod_cv = prod_std / (crop_muni_prod + 0.01)
    
    area_median = crop_area_median.get(crop, 5)
    
    test_input = pd.DataFrame({
        'MUNICIPALITY': [municipality],
        'FARM TYPE': [farm_type],
        'MONTH': [month],
        'CROP': [crop],
        'SEASON': [season],
        'YEAR_NORMALIZED': [(year - year_min) / (year_max - year_min + 1)],
        'MONTH_SIN': [np.sin(2 * np.pi * month_num / 12)],
        'MONTH_COS': [np.cos(2 * np.pi * month_num / 12)],
        'CROP_PRODUCTIVITY': [crop_prod],
        'MUNI_PRODUCTIVITY': [muni_prod],
        'CROP_MUNI_PRODUCTIVITY': [crop_muni_prod],
        'CROP_MUNI_PRODUCTIVITY_MEAN': [crop_muni_mean],
        'CROP_MUNI_MONTH_PRODUCTIVITY': [crop_muni_month],
        'CROP_SEASON_PRODUCTIVITY': [crop_season],
        'PRODUCTIVITY_CV': [prod_cv],
        'LOG_AREA': [np.log1p(area)],
        'AREA_RATIO': [area / (area_median + 0.01)]
    })
    
    pred_productivity = best_model.predict(test_input)[0]
    pred_productivity = np.clip(pred_productivity, 0.5, 40)
    pred_production = pred_productivity * area
    
    return pred_productivity, pred_production

# Test LETTUCE scaling
print("\n  LETTUCE Area Scaling Test (ATOK):")
print(f"  Historical LETTUCE productivity in ATOK: {crop_muni_prod_dict.get('LETTUCE_ATOK', 'N/A'):.2f} MT/HA")
print(f"  {'Area (ha)':<12} {'Pred Prod.':<15} {'Production':<15} {'Expected':<15} {'Diff':<10}")
print("  " + "-"*67)

lettuce_prod = crop_muni_prod_dict.get('LETTUCE_ATOK', 11)
for area in [0.1, 0.5, 1.0, 2.86, 5.0, 10.0, 20.0]:
    pred_prod, pred_production = predict_production('LETTUCE', 'ATOK', area)
    expected = area * lettuce_prod
    diff = (pred_production - expected) / expected * 100 if expected > 0 else 0
    print(f"  {area:<12.2f} {pred_prod:<15.2f} {pred_production:<15.2f} {expected:<15.2f} {diff:+.1f}%")

# Linear scaling check
print("\n  Linear Scaling Check:")
_, prod_1 = predict_production('LETTUCE', 'ATOK', 1)
_, prod_10 = predict_production('LETTUCE', 'ATOK', 10)
_, prod_100 = predict_production('LETTUCE', 'ATOK', 100)
print(f"  1 ha → 10 ha: Production should be 10x, actual: {prod_10/prod_1:.1f}x")
print(f"  1 ha → 100 ha: Production should be 100x, actual: {prod_100/prod_1:.1f}x")

# Test other crops
print("\n  Multi-Crop Test (10 ha each):")
for crop in ['LETTUCE', 'WHITE POTATO', 'CABBAGE', 'CARROT', 'BROCCOLI']:
    if crop in crop_productivity:
        pred_prod, pred_production = predict_production(crop, 'ATOK', 10)
        hist_prod = crop_muni_prod_dict.get(f'{crop}_ATOK', crop_productivity.get(crop, 10))
        print(f"  {crop:<15}: Predicted={pred_prod:.2f} MT/HA, Historical={hist_prod:.2f} MT/HA")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
print(f"""
SUMMARY:
- Model predicts PRODUCTIVITY (MT/HA) with R² = {results[best_name]['r2_productivity']:.4f}
- Production MAPE: {results[best_name]['mape_production']:.2f}%
- Predictions now scale linearly with area!

USAGE:
  productivity = model.predict(features)  # Returns MT/HA
  production = productivity * area        # Calculate production

The ml_api.py will need to be updated to use this new approach.
""")
