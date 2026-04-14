"""
ULTIMATE ACCURACY MODEL TRAINING
Goal: Achieve maximum prediction accuracy using multiple techniques
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import Ridge
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ULTIMATE ACCURACY MODEL - MAXIMUM PRECISION")
print("="*80)

# ===================== STEP 1: LOAD AND CLEAN DATA =====================
print("\n[STEP 1] Loading and cleaning data...")
df = pd.read_csv('fulldataset.csv')

# Convert numeric columns
numeric_cols = ['Production(mt)', 'Area planted(ha)', 'Area harvested(ha)', 'Productivity(mt/ha)']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove invalid records
df = df.dropna(subset=['Production(mt)', 'Area planted(ha)'])
df = df[df['Production(mt)'] > 0]
df = df[df['Area planted(ha)'] > 0]

print(f"   Total samples: {len(df)}")
print(f"   Production range: {df['Production(mt)'].min():.2f} - {df['Production(mt)'].max():.2f} MT")
print(f"   Crops: {df['CROP'].nunique()}")
print(f"   Municipalities: {df['MUNICIPALITY'].nunique()}")

# ===================== STEP 2: ADVANCED FEATURE ENGINEERING =====================
print("\n[STEP 2] Advanced feature engineering...")

# Season mapping
season_map = {
    'DEC': 'DRY', 'JAN': 'DRY', 'FEB': 'DRY', 'MAR': 'DRY', 'APR': 'DRY',
    'MAY': 'WET', 'JUN': 'WET', 'JUL': 'WET', 'AUG': 'WET', 'SEP': 'WET', 'OCT': 'WET', 'NOV': 'WET'
}
df['SEASON'] = df['MONTH'].map(season_map)

# Month to number for cyclical features
month_to_num = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
df['MONTH_NUM'] = df['MONTH'].map(month_to_num)

# Cyclical month encoding (captures that Dec is close to Jan)
df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH_NUM'] / 12)
df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH_NUM'] / 12)

# Historical averages (CRITICAL for accuracy)
df['CROP_AVG'] = df.groupby('CROP')['Production(mt)'].transform('mean')
df['MUNI_AVG'] = df.groupby('MUNICIPALITY')['Production(mt)'].transform('mean')
df['CROP_MUNI_AVG'] = df.groupby(['CROP', 'MUNICIPALITY'])['Production(mt)'].transform('mean')
df['CROP_MUNI_MONTH_AVG'] = df.groupby(['CROP', 'MUNICIPALITY', 'MONTH'])['Production(mt)'].transform('mean')

# Productivity features
df['PRODUCTIVITY_ESTIMATE'] = df.groupby('CROP')['Productivity(mt/ha)'].transform('median')
df['EXPECTED_PRODUCTION'] = df['Area planted(ha)'] * df['PRODUCTIVITY_ESTIMATE']

# Area features
df['LOG_AREA'] = np.log1p(df['Area planted(ha)'])
df['AREA_SQUARED'] = df['Area planted(ha)'] ** 2
df['SQRT_AREA'] = np.sqrt(df['Area planted(ha)'])

# Year features
year_min = df['YEAR'].min()
year_max = df['YEAR'].max()
df['YEAR_NORMALIZED'] = (df['YEAR'] - year_min) / (year_max - year_min)
df['YEARS_FROM_START'] = df['YEAR'] - year_min

# Trend features - production trend over years per crop-municipality
def calc_trend(group):
    if len(group) > 1:
        years = group['YEAR'].values
        prod = group['Production(mt)'].values
        if np.std(years) > 0:
            slope = np.polyfit(years, prod, 1)[0]
            return slope
    return 0

trends = df.groupby(['CROP', 'MUNICIPALITY']).apply(calc_trend).reset_index()
trends.columns = ['CROP', 'MUNICIPALITY', 'PROD_TREND']
df = df.merge(trends, on=['CROP', 'MUNICIPALITY'], how='left')
df['PROD_TREND'] = df['PROD_TREND'].fillna(0)

# Interaction features
df['CROP_SEASON'] = df['CROP'] + '_' + df['SEASON']
df['PRODUCTIVITY_AREA'] = df['PRODUCTIVITY_ESTIMATE'] * df['Area planted(ha)']
df['LOG_EXPECTED_PROD'] = np.log1p(df['EXPECTED_PRODUCTION'])

# Variance/stability features
df['CROP_MUNI_STD'] = df.groupby(['CROP', 'MUNICIPALITY'])['Production(mt)'].transform('std').fillna(0)
df['PROD_CV'] = df['CROP_MUNI_STD'] / (df['CROP_MUNI_AVG'] + 1)  # Coefficient of variation

# Relative features
df['AREA_VS_CROP_AVG'] = df['Area planted(ha)'] / (df.groupby('CROP')['Area planted(ha)'].transform('mean') + 0.01)
df['PROD_VS_MUNI_AVG'] = df['CROP_MUNI_AVG'] / (df['MUNI_AVG'] + 0.01)

print(f"   Created {len(df.columns) - 9} new features")

# ===================== STEP 3: PREPARE FEATURES =====================
print("\n[STEP 3] Preparing features...")

categorical_features = ['MUNICIPALITY', 'FARM TYPE', 'MONTH', 'CROP', 'SEASON', 'CROP_SEASON']
numerical_features = [
    'YEAR', 'Area planted(ha)', 'LOG_AREA', 'AREA_SQUARED', 'SQRT_AREA',
    'MONTH_SIN', 'MONTH_COS',
    'CROP_AVG', 'MUNI_AVG', 'CROP_MUNI_AVG', 'CROP_MUNI_MONTH_AVG',
    'PRODUCTIVITY_ESTIMATE', 'EXPECTED_PRODUCTION', 'LOG_EXPECTED_PROD',
    'YEAR_NORMALIZED', 'YEARS_FROM_START',
    'PROD_TREND', 'PRODUCTIVITY_AREA',
    'CROP_MUNI_STD', 'PROD_CV',
    'AREA_VS_CROP_AVG', 'PROD_VS_MUNI_AVG'
]

X = df[categorical_features + numerical_features].copy()
y = df['Production(mt)'].values

print(f"   Total features: {len(categorical_features) + len(numerical_features)}")
print(f"   Categorical: {len(categorical_features)}")
print(f"   Numerical: {len(numerical_features)}")

# Stratified split by crop
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['CROP']
)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ===================== STEP 4: CREATE PREPROCESSING =====================
print("\n[STEP 4] Creating preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='drop'
)

# ===================== STEP 5: TRAIN MULTIPLE MODELS =====================
print("\n[STEP 5] Training and comparing multiple models...")
print("="*80)

results = {}

# Model 1: Optimized Random Forest with Log Transform
print("\n>>> Training Random Forest (Log Transform)...")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=500,
        max_depth=50,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    ))
])

rf_model = TransformedTargetRegressor(
    regressor=rf_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
results['Random Forest'] = {
    'r2': r2_score(y_test, rf_pred),
    'mae': mean_absolute_error(y_test, rf_pred),
    'mape': mean_absolute_percentage_error(y_test, rf_pred) * 100,
    'model': rf_model
}
print(f"   R2: {results['Random Forest']['r2']:.4f} | MAPE: {results['Random Forest']['mape']:.2f}%")

# Model 2: Gradient Boosting with Log Transform
print("\n>>> Training Gradient Boosting...")
gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    ))
])

gb_model = TransformedTargetRegressor(
    regressor=gb_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
results['Gradient Boosting'] = {
    'r2': r2_score(y_test, gb_pred),
    'mae': mean_absolute_error(y_test, gb_pred),
    'mape': mean_absolute_percentage_error(y_test, gb_pred) * 100,
    'model': gb_model
}
print(f"   R2: {results['Gradient Boosting']['r2']:.4f} | MAPE: {results['Gradient Boosting']['mape']:.2f}%")

# Model 3: Extra Trees with Log Transform
print("\n>>> Training Extra Trees...")
et_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', ExtraTreesRegressor(
        n_estimators=500,
        max_depth=50,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    ))
])

et_model = TransformedTargetRegressor(
    regressor=et_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)
et_model.fit(X_train, y_train)
et_pred = et_model.predict(X_test)
results['Extra Trees'] = {
    'r2': r2_score(y_test, et_pred),
    'mae': mean_absolute_error(y_test, et_pred),
    'mape': mean_absolute_percentage_error(y_test, et_pred) * 100,
    'model': et_model
}
print(f"   R2: {results['Extra Trees']['r2']:.4f} | MAPE: {results['Extra Trees']['mape']:.2f}%")

# Model 4: Ensemble (Average of top models)
print("\n>>> Creating Ensemble (Weighted Average)...")
# Use predictions from all models
ensemble_pred = (0.4 * rf_pred + 0.3 * gb_pred + 0.3 * et_pred)
results['Ensemble'] = {
    'r2': r2_score(y_test, ensemble_pred),
    'mae': mean_absolute_error(y_test, ensemble_pred),
    'mape': mean_absolute_percentage_error(y_test, ensemble_pred) * 100,
    'model': None  # Will save all models
}
print(f"   R2: {results['Ensemble']['r2']:.4f} | MAPE: {results['Ensemble']['mape']:.2f}%")

# ===================== STEP 6: SELECT BEST MODEL =====================
print("\n" + "="*80)
print("[STEP 6] MODEL COMPARISON RESULTS")
print("="*80)
print(f"\n{'Model':<20} {'R2 Score':<12} {'MAE (MT)':<12} {'MAPE (%)':<12}")
print("-"*60)

for name, res in sorted(results.items(), key=lambda x: -x[1]['r2']):
    print(f"{name:<20} {res['r2']:.4f}       {res['mae']:.2f}        {res['mape']:.2f}")

# Find best model
best_name = max(results, key=lambda x: results[x]['r2'])
best_result = results[best_name]
print(f"\n>>> BEST MODEL: {best_name}")
print(f"    R2 Score: {best_result['r2']:.4f} ({best_result['r2']*100:.1f}% variance explained)")
print(f"    MAPE: {best_result['mape']:.2f}% average error")

# ===================== STEP 7: DETAILED ANALYSIS =====================
print("\n" + "="*80)
print("[STEP 7] DETAILED PERFORMANCE ANALYSIS")
print("="*80)

# Use best individual model for detailed analysis
if best_name == 'Ensemble':
    best_pred = ensemble_pred
    best_model = rf_model  # Use RF for prediction intervals
else:
    best_pred = results[best_name]['model'].predict(X_test)
    best_model = results[best_name]['model']

# Performance by production scale
test_df = X_test.copy()
test_df['Actual'] = y_test
test_df['Predicted'] = best_pred

print("\nPerformance by Production Scale:")
print("-"*50)
scales = [
    ('Small (0-50 MT)', (0, 50)),
    ('Medium (50-200 MT)', (50, 200)),
    ('Large (200-1000 MT)', (200, 1000)),
    ('Very Large (>1000 MT)', (1000, float('inf')))
]

for scale_name, (low, high) in scales:
    mask = (test_df['Actual'] >= low) & (test_df['Actual'] < high)
    if mask.sum() > 0:
        scale_r2 = r2_score(test_df.loc[mask, 'Actual'], test_df.loc[mask, 'Predicted'])
        scale_mape = mean_absolute_percentage_error(test_df.loc[mask, 'Actual'], test_df.loc[mask, 'Predicted']) * 100
        print(f"  {scale_name:<25} R2: {scale_r2:.4f}  MAPE: {scale_mape:.2f}%  (n={mask.sum()})")

# Performance by crop
print("\nPerformance by Crop:")
print("-"*50)
for crop in sorted(test_df['CROP'].unique()):
    mask = test_df['CROP'] == crop
    if mask.sum() >= 10:
        crop_r2 = r2_score(test_df.loc[mask, 'Actual'], test_df.loc[mask, 'Predicted'])
        crop_mape = mean_absolute_percentage_error(test_df.loc[mask, 'Actual'], test_df.loc[mask, 'Predicted']) * 100
        print(f"  {crop:<20} R2: {crop_r2:.4f}  MAPE: {crop_mape:.2f}%  (n={mask.sum()})")

# ===================== STEP 8: SAVE BEST MODEL =====================
print("\n" + "="*80)
print("[STEP 8] SAVING BEST MODEL")
print("="*80)

# Save individual models for ensemble
if best_name == 'Ensemble':
    ensemble_models = {
        'rf': rf_model,
        'gb': gb_model,
        'et': et_model,
        'weights': [0.4, 0.3, 0.3]
    }
    joblib.dump(ensemble_models, 'model_artifacts/best_ensemble_model.pkl')
    print("   Saved: model_artifacts/best_ensemble_model.pkl")
else:
    joblib.dump(best_model, 'model_artifacts/best_ultimate_model.pkl')
    print("   Saved: model_artifacts/best_ultimate_model.pkl")

# Also save the RF model for prediction intervals
joblib.dump(rf_model, 'model_artifacts/best_rf_ultimate.pkl')
print("   Saved: model_artifacts/best_rf_ultimate.pkl")

# Save metadata
metadata = {
    'model_type': best_name,
    'training_date': str(datetime.now()),
    'test_r2_score': float(best_result['r2']),
    'test_mae': float(best_result['mae']),
    'test_mape': float(best_result['mape']),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'all_models_performance': {
        name: {'r2': float(res['r2']), 'mape': float(res['mape'])}
        for name, res in results.items()
    },
    'feature_engineering': {
        'season_map': season_map,
        'month_to_num': month_to_num,
        'year_min': int(year_min),
        'year_max': int(year_max)
    }
}

with open('model_artifacts/ultimate_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("   Saved: model_artifacts/ultimate_model_metadata.json")

# Save feature statistics for prediction
feature_stats = {
    'crop_avg': df.groupby('CROP')['Production(mt)'].mean().to_dict(),
    'muni_avg': df.groupby('MUNICIPALITY')['Production(mt)'].mean().to_dict(),
    'crop_muni_avg': {f"{k[0]}_{k[1]}": v for k, v in df.groupby(['CROP', 'MUNICIPALITY'])['Production(mt)'].mean().to_dict().items()},
    'crop_muni_month_avg': {
        f"{k[0]}_{k[1]}_{k[2]}": v for k, v in 
        df.groupby(['CROP', 'MUNICIPALITY', 'MONTH'])['Production(mt)'].mean().to_dict().items()
    },
    'crop_productivity': df.groupby('CROP')['Productivity(mt/ha)'].median().to_dict(),
    'crop_area_avg': df.groupby('CROP')['Area planted(ha)'].mean().to_dict(),
    'prod_trends': {f"{k[0]}_{k[1]}": v for k, v in trends.set_index(['CROP', 'MUNICIPALITY'])['PROD_TREND'].to_dict().items()},
    'crop_muni_std': {f"{k[0]}_{k[1]}": v for k, v in df.groupby(['CROP', 'MUNICIPALITY'])['Production(mt)'].std().fillna(0).to_dict().items()},
    'overall_mean': float(df['Production(mt)'].mean())
}

with open('model_artifacts/feature_statistics.json', 'w') as f:
    json.dump(feature_stats, f, indent=2, default=str)
print("   Saved: model_artifacts/feature_statistics.json")

# Save categorical values
categorical_values = {
    col: sorted(df[col].unique().tolist())
    for col in categorical_features
}

with open('model_artifacts/categorical_values_ultimate.json', 'w') as f:
    json.dump(categorical_values, f, indent=2)
print("   Saved: model_artifacts/categorical_values_ultimate.json")

# ===================== FINAL SUMMARY =====================
print("\n" + "="*80)
print("TRAINING COMPLETE - FINAL SUMMARY")
print("="*80)
print(f"\n   BEST MODEL: {best_name}")
print(f"   R2 Score: {best_result['r2']:.4f} ({best_result['r2']*100:.1f}% variance explained)")
print(f"   MAPE: {best_result['mape']:.2f}% (average prediction error)")
print(f"   MAE: {best_result['mae']:.2f} MT (typical error)")
print(f"\n   Model is ready for production use!")
print("="*80)
