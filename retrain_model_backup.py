"""
Retrain ML Model with Current sklearn Version
Fixes version mismatch and improves prediction accuracy
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("RETRAINING MODEL WITH SKLEARN 1.3.2")
print("="*60)

# Load data from database
try:
    from database import db_manager
    print("\n[1/7] Loading data from database...")
    df = db_manager.get_crop_production_data()
    print(f"Loaded {len(df)} records from database")
except:
    print("\n[1/7] Loading data from CSV...")
    df = pd.read_csv('fulldataset.csv')
    print(f"Loaded {len(df)} records from CSV")

# Standardize column names
df.columns = df.columns.str.strip().str.upper()
column_mapping = {
    'FARM_TYPE': 'FARM TYPE',
    'AREA_PLANTED': 'Area planted(ha)',
    'AREA_HARVESTED': 'Area harvested(ha)',
    'PRODUCTION': 'Production(MT)',
    'PRODUCTIVITY': 'Yield(MT/ha)'
}
for old, new in column_mapping.items():
    if old in df.columns:
        df.rename(columns={old: new}, inplace=True)

# Ensure we have required columns
if 'FARM TYPE' not in df.columns and 'FARM_TYPE' in df.columns:
    df.rename(columns={'FARM_TYPE': 'FARM TYPE'}, inplace=True)
if 'Area planted(ha)' not in df.columns:
    for col in df.columns:
        if 'AREA' in col and 'PLANT' in col:
            df.rename(columns={col: 'Area planted(ha)'}, inplace=True)
            break
if 'Production(MT)' not in df.columns:
    for col in df.columns:
        if 'PRODUCTION' in col:
            df.rename(columns={col: 'Production(MT)'}, inplace=True)
            break

# Also get Area harvested if available
if 'Area harvested(ha)' not in df.columns:
    for col in df.columns:
        if 'AREA' in col and 'HARVEST' in col:
            df.rename(columns={col: 'Area harvested(ha)'}, inplace=True)
            break

print(f"Columns: {list(df.columns)}")

# Clean data
print("\n[2/7] Cleaning data...")
df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
df['Area planted(ha)'] = pd.to_numeric(df['Area planted(ha)'], errors='coerce')
df['Production(MT)'] = pd.to_numeric(df['Production(MT)'], errors='coerce')
df['Area harvested(ha)'] = pd.to_numeric(df['Area harvested(ha)'], errors='coerce')
df = df.dropna(subset=['YEAR', 'Area planted(ha)', 'Production(MT)'])
df = df[df['Production(MT)'] > 0]
df = df[df['Area planted(ha)'] > 0]

# Use the stored PRODUCTIVITY (which is Production/AreaHarvested) for validation
# But calculate based on area planted for training to ensure consistency
df['CALC_YIELD'] = df['Production(MT)'] / df['Area planted(ha)']

# Also check the stored productivity for reasonableness
if 'Yield(MT/ha)' in df.columns:
    df['STORED_YIELD'] = pd.to_numeric(df['Yield(MT/ha)'], errors='coerce')
else:
    df['STORED_YIELD'] = df['Production(MT)'] / df['Area harvested(ha)']

# Cap extreme CALCULATED yields (productivity above 50 MT/ha is unrealistic)
# But keep records where STORED_YIELD is reasonable (< 30) even if CALC_YIELD is high
# This handles cases where area_harvested >> area_planted
yield_cap = 50
stored_yield_cap = 30

# Keep if EITHER calculated yield is ok OR stored yield is reasonable
df_clean = df[(df['CALC_YIELD'] <= yield_cap) | (df['STORED_YIELD'] <= stored_yield_cap)]
outliers_removed = len(df) - len(df_clean)
df = df_clean

# For training, use production that's consistent with area planted
# If stored yield is reasonable, recalculate production = area_planted * stored_yield
# This ensures predictions based on area_planted will be accurate
df['ADJUSTED_PRODUCTION'] = np.where(
    df['STORED_YIELD'] <= stored_yield_cap,
    df['Area planted(ha)'] * df['STORED_YIELD'],  # Use realistic yield
    df['Production(MT)']  # Keep original if stored yield is also extreme
)

# Use adjusted production for training
df['Production(MT)'] = df['ADJUSTED_PRODUCTION']
df['CALC_YIELD'] = df['Production(MT)'] / df['Area planted(ha)']

# Final cleanup - remove any remaining extreme values
df = df[df['CALC_YIELD'] <= yield_cap]

print(f"Removed {outliers_removed} outlier records")
print(f"After cleaning: {len(df)} records")

# Feature engineering
print("\n[3/7] Engineering features...")
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

# Area features
df['LOG_AREA'] = np.log1p(df['Area planted(ha)'])
df['AREA_SQUARED'] = df['Area planted(ha)'] ** 2
df['SQRT_AREA'] = np.sqrt(df['Area planted(ha)'])

# Calculate aggregates for feature engineering
overall_mean = df['Production(MT)'].mean()
crop_avg = df.groupby('CROP')['Production(MT)'].mean().to_dict()
muni_avg = df.groupby('MUNICIPALITY')['Production(MT)'].mean().to_dict()
crop_muni_avg = df.groupby(['CROP', 'MUNICIPALITY'])['Production(MT)'].mean()
crop_muni_avg_dict = {f"{c}_{m}": v for (c, m), v in crop_muni_avg.items()}
crop_muni_month_avg = df.groupby(['CROP', 'MUNICIPALITY', 'MONTH'])['Production(MT)'].mean()
crop_muni_month_dict = {f"{c}_{m}_{mo}": v for (c, m, mo), v in crop_muni_month_avg.items()}

# Productivity by crop
crop_productivity = df.groupby('CROP').apply(
    lambda x: (x['Production(MT)'] / x['Area planted(ha)']).median()
).to_dict()

# Standard deviation
crop_muni_std = df.groupby(['CROP', 'MUNICIPALITY'])['Production(MT)'].std()
crop_muni_std_dict = {f"{c}_{m}": v for (c, m), v in crop_muni_std.items()}

# Area averages
crop_area_avg = df.groupby('CROP')['Area planted(ha)'].mean().to_dict()

# Add features to dataframe
df['CROP_AVG'] = df['CROP'].map(crop_avg)
df['MUNI_AVG'] = df['MUNICIPALITY'].map(muni_avg)
df['CROP_MUNI_AVG'] = df.apply(lambda r: crop_muni_avg_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}", r['CROP_AVG']), axis=1)
df['CROP_MUNI_MONTH_AVG'] = df.apply(lambda r: crop_muni_month_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}_{r['MONTH']}", r['CROP_MUNI_AVG']), axis=1)
df['PRODUCTIVITY_ESTIMATE'] = df['CROP'].map(crop_productivity)
df['PROD_TREND'] = df.apply(lambda r: crop_muni_std_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}", 0), axis=1)
df['CROP_MUNI_STD'] = df.apply(lambda r: crop_muni_std_dict.get(f"{r['CROP']}_{r['MUNICIPALITY']}", 0), axis=1)

df['EXPECTED_PRODUCTION'] = df['Area planted(ha)'] * df['PRODUCTIVITY_ESTIMATE']
df['LOG_EXPECTED_PROD'] = np.log1p(df['EXPECTED_PRODUCTION'])

year_min = df['YEAR'].min()
year_max = df['YEAR'].max()
df['YEAR_NORMALIZED'] = (df['YEAR'] - year_min) / (year_max - year_min)
df['YEARS_FROM_START'] = df['YEAR'] - year_min

df['PRODUCTIVITY_AREA'] = df['PRODUCTIVITY_ESTIMATE'] * df['Area planted(ha)']
df['PROD_CV'] = df['CROP_MUNI_STD'] / (df['CROP_MUNI_AVG'] + 1)
df['AREA_VS_CROP_AVG'] = df.apply(lambda r: r['Area planted(ha)'] / (crop_area_avg.get(r['CROP'], r['Area planted(ha)']) + 0.01), axis=1)
df['PROD_VS_MUNI_AVG'] = df['CROP_MUNI_AVG'] / (df['MUNI_AVG'] + 0.01)
df['CROP_SEASON'] = df['CROP'] + '_' + df['SEASON']

print(f"Features created: {len(df.columns)} columns")

# Define features for model
categorical_features = ['MUNICIPALITY', 'FARM TYPE', 'MONTH', 'CROP']
numerical_features = [
    'YEAR', 'Area planted(ha)', 'LOG_AREA', 'SQRT_AREA',
    'MONTH_SIN', 'MONTH_COS', 'CROP_AVG', 'MUNI_AVG', 
    'CROP_MUNI_AVG', 'CROP_MUNI_MONTH_AVG', 'PRODUCTIVITY_ESTIMATE',
    'EXPECTED_PRODUCTION', 'LOG_EXPECTED_PROD', 'YEAR_NORMALIZED',
    'YEARS_FROM_START', 'PRODUCTIVITY_AREA', 'PROD_CV', 
    'AREA_VS_CROP_AVG', 'PROD_VS_MUNI_AVG'
]

# Remove any features with NaN
for col in numerical_features:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

X = df[categorical_features + numerical_features]
y = df['Production(MT)']

print(f"\nFeatures: {len(categorical_features)} categorical + {len(numerical_features)} numerical")

# Train-test split
print("\n[4/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

# Train multiple models
print("\n[5/7] Training models...")
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=5, 
        min_samples_leaf=2, n_jobs=-1, random_state=42
    ),
    'Extra Trees': ExtraTreesRegressor(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, n_jobs=-1, random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        min_samples_split=5, min_samples_leaf=2, random_state=42
    )
}

results = {}
best_model = None
best_r2 = -999

for name, regressor in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline with log transform on target
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', regressor)
    ])
    
    model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
    
    results[name] = {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape, 'model': model}
    print(f"  R²: {r2:.4f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_name}")
print(f"R²: {results[best_name]['r2']:.4f}")
print(f"MAE: {results[best_name]['mae']:.2f} MT")
print(f"MAPE: {results[best_name]['mape']:.2f}%")
print(f"{'='*60}")

# Save models
print("\n[6/7] Saving models...")
joblib.dump(best_model, 'model_artifacts/best_model.pkl')
joblib.dump(results['Random Forest']['model'], 'model_artifacts/best_rf_model.pkl')
print("Models saved!")

# Save metadata
metadata = {
    'model_type': best_name,
    'sklearn_version': '1.3.2',
    'test_r2_score': results[best_name]['r2'],
    'test_mae': results[best_name]['mae'],
    'test_rmse': results[best_name]['rmse'],
    'test_mape': results[best_name]['mape'],
    'features': categorical_features + numerical_features,
    'feature_engineering': {
        'season_map': season_map,
        'month_to_num': month_to_num,
        'year_min': int(year_min),
        'year_max': int(year_max)
    },
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('model_artifacts/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Save feature statistics
feature_stats = {
    'overall_mean': overall_mean,
    'crop_avg': crop_avg,
    'muni_avg': muni_avg,
    'crop_muni_avg': crop_muni_avg_dict,
    'crop_muni_month_avg': crop_muni_month_dict,
    'crop_productivity': crop_productivity,
    'crop_muni_std': {k: v if not pd.isna(v) else 0 for k, v in crop_muni_std_dict.items()},
    'crop_area_avg': crop_area_avg,
    'prod_trends': {k: 0 for k in crop_muni_avg_dict.keys()}
}

with open('model_artifacts/feature_statistics.json', 'w') as f:
    json.dump(feature_stats, f, indent=2)

# Save feature info
feature_info = {
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'model_features': categorical_features + numerical_features,
    'n_features': len(categorical_features) + len(numerical_features)
}

with open('model_artifacts/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print("Metadata saved!")

# Test prediction
print("\n[7/7] Testing prediction...")
test_input = pd.DataFrame({
    'MUNICIPALITY': ['ATOK'],
    'FARM TYPE': ['IRRIGATED'],
    'YEAR': [2025],
    'MONTH': ['JAN'],
    'CROP': ['WHITE POTATO'],
    'Area planted(ha)': [50],
    'LOG_AREA': [np.log1p(50)],
    'SQRT_AREA': [np.sqrt(50)],
    'MONTH_SIN': [np.sin(2 * np.pi * 1 / 12)],
    'MONTH_COS': [np.cos(2 * np.pi * 1 / 12)],
    'CROP_AVG': [crop_avg.get('WHITE POTATO', overall_mean)],
    'MUNI_AVG': [muni_avg.get('ATOK', overall_mean)],
    'CROP_MUNI_AVG': [crop_muni_avg_dict.get('WHITE POTATO_ATOK', crop_avg.get('WHITE POTATO', overall_mean))],
    'CROP_MUNI_MONTH_AVG': [crop_muni_month_dict.get('WHITE POTATO_ATOK_JAN', crop_muni_avg_dict.get('WHITE POTATO_ATOK', overall_mean))],
    'PRODUCTIVITY_ESTIMATE': [crop_productivity.get('WHITE POTATO', 14)],
    'EXPECTED_PRODUCTION': [50 * crop_productivity.get('WHITE POTATO', 14)],
    'LOG_EXPECTED_PROD': [np.log1p(50 * crop_productivity.get('WHITE POTATO', 14))],
    'YEAR_NORMALIZED': [(2025 - year_min) / (year_max - year_min)],
    'YEARS_FROM_START': [2025 - year_min],
    'PRODUCTIVITY_AREA': [crop_productivity.get('WHITE POTATO', 14) * 50],
    'PROD_CV': [0],
    'AREA_VS_CROP_AVG': [50 / (crop_area_avg.get('WHITE POTATO', 50) + 0.01)],
    'PROD_VS_MUNI_AVG': [crop_muni_avg_dict.get('WHITE POTATO_ATOK', 1) / (muni_avg.get('ATOK', 1) + 0.01)]
})

prediction = best_model.predict(test_input)[0]
expected = 50 * crop_productivity.get('WHITE POTATO', 14)

print(f"\nTest: WHITE POTATO, 50ha, ATOK, JAN 2025")
print(f"Predicted: {prediction:.2f} MT")
print(f"Expected (based on median productivity {crop_productivity.get('WHITE POTATO', 14):.1f} MT/ha): {expected:.2f} MT")
print(f"Difference: {((prediction - expected) / expected * 100):.1f}%")

print("\n" + "="*60)
print("RETRAINING COMPLETE!")
print("="*60)
