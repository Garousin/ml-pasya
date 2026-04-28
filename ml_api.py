"""
Benguet Crop Production ML API - V2 (Productivity-First)
Uses productivity-first prediction for accurate area scaling
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os
import shutil
from datetime import datetime
from urllib.request import Request, urlopen

# Import database manager
try:
    from database import db_manager, load_data_from_database
    DB_AVAILABLE = db_manager.test_connection()
except ImportError:
    DB_AVAILABLE = False
    db_manager = None
    print("[WARNING] Database module not available, using CSV only")

app = Flask(__name__)
CORS(app)

MODEL_DIR = 'model_artifacts'
DEFAULT_MODEL_ARTIFACT_BASE_URL = os.environ.get(
    'MODEL_ARTIFACT_BASE_URL',
    'https://media.githubusercontent.com/media/Garousin/ML-pasya/main/model_artifacts'
).rstrip('/')


def is_git_lfs_pointer(file_path):
    """Return True when the file is a Git LFS pointer text file, not real binary content."""
    try:
        if not os.path.exists(file_path):
            return False
        with open(file_path, 'rb') as f:
            header = f.read(128)
        return header.startswith(b'version https://git-lfs.github.com/spec/v1')
    except Exception:
        return False


def get_artifact_url(file_name):
    """Resolve artifact URL from explicit env var first, then fallback to base URL."""
    env_map = {
        'best_model.pkl': 'BEST_MODEL_URL',
        'best_rf_model.pkl': 'BEST_RF_MODEL_URL',
    }
    env_var = env_map.get(file_name)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    return f"{DEFAULT_MODEL_ARTIFACT_BASE_URL}/{file_name}"


def ensure_model_artifact(file_name):
    """Download model artifact when missing or when repository contains only an LFS pointer."""
    file_path = os.path.join(MODEL_DIR, file_name)
    missing = not os.path.exists(file_path)
    pointer = is_git_lfs_pointer(file_path)

    if not missing and not pointer:
        return None

    reason = 'missing' if missing else 'git-lfs pointer detected'
    url = get_artifact_url(file_name)
    print(f"[INFO] {file_name}: {reason}, downloading from {url}")

    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        request = Request(url, headers={'User-Agent': 'ml-pasya-api/1.0'})
        with urlopen(request, timeout=180) as response, open(file_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        if is_git_lfs_pointer(file_path):
            return (
                f"{file_name} downloaded as Git LFS pointer. "
                "Use media.githubusercontent.com URL or set BEST_MODEL_URL/BEST_RF_MODEL_URL."
            )

        print(f"[OK] Downloaded artifact: {file_name}")
        return None
    except Exception as e:
        return f"{file_name} download failed: {e}"

print("Loading ML API V2 (Productivity-First)...")

model = None
rf_model = None
metadata = {}
feature_stats = {}
startup_errors = []
OFFICIAL_ML_TASK = 'PRODUCTIVITY_FIRST_PLANNING_ESTIMATE'
OFFICIAL_TRAINING_PIPELINE = 'retrain_model_optimized.py'
HEURISTIC_CONFIDENCE_NOTE = 'Confidence intervals are based on random-forest tree dispersion and are not yet calibrated uncertainty estimates.'

for artifact_name in ('best_model.pkl', 'best_rf_model.pkl'):
    download_error = ensure_model_artifact(artifact_name)
    if download_error:
        startup_errors.append(download_error)

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
except Exception as e:
    startup_errors.append(f"best_model.pkl failed to load: {e}")

try:
    rf_model = joblib.load(os.path.join(MODEL_DIR, 'best_rf_model.pkl'))
except Exception as e:
    startup_errors.append(f"best_rf_model.pkl failed to load: {e}")

try:
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
except Exception as e:
    startup_errors.append(f"model_metadata.json failed to load: {e}")

try:
    with open(os.path.join(MODEL_DIR, 'feature_statistics.json'), 'r') as f:
        feature_stats = json.load(f)
except Exception as e:
    startup_errors.append(f"feature_statistics.json failed to load: {e}")

# Check if using productivity-first model
PRODUCTIVITY_FIRST = metadata.get('prediction_target') == 'PRODUCTIVITY'
if not PRODUCTIVITY_FIRST:
    startup_errors.append(
        'Deployment is frozen to the PRODUCTIVITY_FIRST planning task. Retrain and deploy artifacts from retrain_model_optimized.py only.'
    )
print(f"[OK] Model type: {metadata.get('model_type', 'Unknown')}")
print(f"[OK] Prediction target: {'PRODUCTIVITY' if PRODUCTIVITY_FIRST else 'PRODUCTION'}")
print(f"[OK] Database: {'Connected' if DB_AVAILABLE else 'Not available (using CSV)'}")

if 'test_r2_score' in metadata:
    print(f"[OK] R² Score: {metadata.get('test_r2_score', 0):.4f}")

if startup_errors:
    print("[WARNING] Startup completed with missing artifacts:")
    for err in startup_errors:
        print(f"  - {err}")

# Feature engineering maps
feature_engineering = metadata.get('feature_engineering', {})
season_map = feature_engineering.get('season_map', {})
month_to_num = feature_engineering.get('month_to_num', {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
})
year_min = feature_engineering.get('year_min', 2000)
year_max = feature_engineering.get('year_max', datetime.now().year)
productivity_categorical_features = set(metadata.get('categorical_features', []))
productivity_numerical_features = set(metadata.get('numerical_features', []))
area_context_config = feature_engineering.get('area_context', {})
area_context_small_ratio = area_context_config.get('small_ratio', 0.75)
area_context_large_ratio = area_context_config.get('large_ratio', 1.25)

# Month name to abbreviation mapping
month_full_to_abbr = {
    'JANUARY': 'JAN', 'FEBRUARY': 'FEB', 'MARCH': 'MAR', 'APRIL': 'APR',
    'MAY': 'MAY', 'JUNE': 'JUN', 'JULY': 'JUL', 'AUGUST': 'AUG',
    'SEPTEMBER': 'SEP', 'OCTOBER': 'OCT', 'NOVEMBER': 'NOV', 'DECEMBER': 'DEC',
    'JAN': 'JAN', 'FEB': 'FEB', 'MAR': 'MAR', 'APR': 'APR',
    'JUN': 'JUN', 'JUL': 'JUL', 'AUG': 'AUG', 'SEP': 'SEP', 
    'OCT': 'OCT', 'NOV': 'NOV', 'DEC': 'DEC'
}
month_num_to_abbr = {value: key for key, value in month_to_num.items()}

# Municipality name normalization (handle spaces/variations)
municipality_normalize = {
    'LA TRINIDAD': 'LATRINIDAD',
    'LATRINIDAD': 'LATRINIDAD',
    'LA_TRINIDAD': 'LATRINIDAD',
}

def normalize_municipality(name):
    """Normalize municipality name to match training data format"""
    if name is None:
        return name
    name_upper = name.upper().strip()
    # Check explicit mapping first
    if name_upper in municipality_normalize:
        return municipality_normalize[name_upper]
    # Remove spaces and special chars for matching
    return name_upper.replace(' ', '').replace('_', '')


def get_historical_data_range():
    """Get min and max years from historical data"""
    if DB_AVAILABLE and db_manager:
        min_year, max_year = db_manager.get_historical_data_range()
        if min_year and max_year:
            return min_year, max_year
    df = pd.read_csv('fulldataset.csv')
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df = df.dropna(subset=['YEAR'])
    return int(df['YEAR'].min()), int(df['YEAR'].max())


def get_first_present(payload, keys, default=None):
    """Return the first non-empty value found for a set of possible keys."""
    for key in keys:
        value = payload.get(key)
        if value is not None and value != '':
            return value
    return default


def normalize_month_value(value):
    """Accept month numbers, abbreviations, or full names and return JAN/FEB/..."""
    if value is None:
        return None

    if isinstance(value, (int, np.integer)):
        month_num = int(value)
    elif isinstance(value, float) and value.is_integer():
        month_num = int(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.isdigit():
            month_num = int(cleaned)
        else:
            month_key = month_full_to_abbr.get(cleaned.upper(), cleaned.upper()[:3])
            if month_key not in month_to_num:
                raise ValueError(f'Invalid month value: {value}')
            return month_key
    else:
        raise ValueError(f'Invalid month value: {value}')

    if month_num not in month_num_to_abbr:
        raise ValueError(f'Invalid month number: {month_num}')
    return month_num_to_abbr[month_num]


def normalize_prediction_payload(raw_payload):
    """Normalize old Laravel and new API payload shapes into one planning request."""
    if not isinstance(raw_payload, dict):
        raise ValueError('Request body must be a JSON object')

    normalized = {
        'municipality': get_first_present(raw_payload, ['municipality', 'MUNICIPALITY']),
        'farm_type': get_first_present(raw_payload, ['farm_type', 'FARM_TYPE', 'farmType']),
        'year': get_first_present(raw_payload, ['year', 'YEAR']),
        'month': get_first_present(raw_payload, ['month', 'MONTH']),
        'crop': get_first_present(raw_payload, ['crop', 'CROP']),
        'area_planted': get_first_present(
            raw_payload,
            ['area_planted', 'area_planted_ha', 'Area_planted_ha', 'AREA_PLANTED_HA', 'Area planted(ha)']
        ),
    }

    required = ['municipality', 'farm_type', 'year', 'month', 'crop', 'area_planted']
    for field in required:
        if normalized[field] is None:
            raise KeyError(field)

    normalized['municipality'] = str(normalized['municipality']).strip()
    normalized['farm_type'] = str(normalized['farm_type']).strip().upper()
    normalized['year'] = int(normalized['year'])
    normalized['month'] = normalize_month_value(normalized['month'])
    normalized['crop'] = str(normalized['crop']).strip().upper()
    normalized['area_planted'] = float(normalized['area_planted'])
    return normalized


def build_available_options_response():
    """Return Laravel-friendly dropdown options."""
    if DB_AVAILABLE and db_manager:
        crops = sorted(db_manager.get_available_crops())
        municipalities = sorted(db_manager.get_available_municipalities())
        source = 'database'
    else:
        crops = sorted(feature_stats.get('crop_productivity', {}).keys())
        municipalities = sorted(feature_stats.get('muni_productivity', {}).keys())
        source = 'model'

    return {
        'source': source,
        'municipalities': municipalities,
        'farm_types': ['IRRIGATED', 'RAINFED'],
        'crops': crops,
        'months': [
            {'value': month_num, 'label': datetime(2000, month_num, 1).strftime('%B'), 'abbr': month_num_to_abbr[month_num]}
            for month_num in range(1, 13)
        ],
    }


def build_legacy_statistics_response():
    """Return summary data for older Laravel integrations."""
    if DB_AVAILABLE and db_manager:
        min_year, max_year = db_manager.get_historical_data_range()
        crops = db_manager.get_available_crops()
        municipalities = db_manager.get_available_municipalities()
        source = 'database'
    else:
        min_year, max_year = get_historical_data_range()
        crops = list(feature_stats.get('crop_productivity', {}).keys())
        municipalities = list(feature_stats.get('muni_productivity', {}).keys())
        source = 'csv'

    return {
        'year_range': {'min': min_year, 'max': max_year},
        'forecast_range': {'min': max_year + 1, 'max': max_year + 5},
        'crops_count': len(crops),
        'municipalities_count': len(municipalities),
        'crops': sorted(crops),
        'municipalities': sorted(municipalities),
        'data_source': source,
        'official_ml_task': OFFICIAL_ML_TASK,
    }


def build_legacy_history_response(filters, page, limit):
    """Return historical production records in a Laravel-friendly format."""
    query_filters = {}
    if filters.get('municipality'):
        query_filters['municipality'] = normalize_municipality(filters['municipality'])
    if filters.get('crop'):
        query_filters['crop'] = str(filters['crop']).upper()
    if filters.get('year'):
        query_filters['year_from'] = int(filters['year'])
        query_filters['year_to'] = int(filters['year'])

    if DB_AVAILABLE and db_manager:
        history_df = db_manager.get_crop_production_data(query_filters)
        source = 'database'
    else:
        history_df = load_data_from_database()
        source = 'csv'
        if query_filters.get('municipality'):
            history_df = history_df[history_df['MUNICIPALITY'] == query_filters['municipality']]
        if query_filters.get('crop'):
            history_df = history_df[history_df['CROP'] == query_filters['crop']]
        if query_filters.get('year_from'):
            history_df = history_df[history_df['YEAR'] >= query_filters['year_from']]
        if query_filters.get('year_to'):
            history_df = history_df[history_df['YEAR'] <= query_filters['year_to']]

    history_df = history_df.sort_values(['YEAR', 'MONTH']).reset_index(drop=True)
    total = int(len(history_df))
    start = max(0, (page - 1) * limit)
    end = start + limit
    page_df = history_df.iloc[start:end].copy()

    records = []
    for _, row in page_df.iterrows():
        records.append({
            'municipality': row['MUNICIPALITY'],
            'farm_type': row.get('FARM_TYPE'),
            'year': int(row['YEAR']),
            'month': row['MONTH'],
            'crop': row['CROP'],
            'area_planted_ha': float(row['AREA_PLANTED']) if pd.notna(row['AREA_PLANTED']) else None,
            'area_harvested_ha': float(row['AREA_HARVESTED']) if 'AREA_HARVESTED' in row and pd.notna(row['AREA_HARVESTED']) else None,
            'production_mt': float(row['PRODUCTION']) if pd.notna(row['PRODUCTION']) else None,
            'productivity_mt_ha': float(row['PRODUCTIVITY']) if 'PRODUCTIVITY' in row and pd.notna(row['PRODUCTIVITY']) else None,
        })

    return {
        'success': True,
        'source': source,
        'total': total,
        'page': page,
        'limit': limit,
        'records': records,
        'data': records,
    }


def get_crop_area_reference(crop):
    """Return the typical planted area for a crop."""
    crop_area_lookup = feature_stats.get('crop_area_median', feature_stats.get('crop_area_avg', {}))
    return crop_area_lookup.get(crop, 5)


def derive_area_context(area, reference_area):
    """Bucket planted area as weak context instead of a direct productivity driver."""
    if pd.isna(area) or area <= 0:
        return 'UNKNOWN'

    baseline = max(reference_area, 0.01)
    ratio = area / baseline

    if ratio < area_context_small_ratio:
        return 'SMALL'
    if ratio > area_context_large_ratio:
        return 'LARGE'
    return 'TYPICAL'


def calculate_features_v2(input_df):
    """Calculate features for productivity-first model"""
    # Convert month names
    input_df['MONTH'] = input_df['MONTH'].map(
        lambda x: month_full_to_abbr.get(x.upper(), x.upper()[:3])
    )
    
    # Season
    input_df['SEASON'] = input_df['MONTH'].map(season_map)
    
    # Month features
    input_df['MONTH_NUM'] = input_df['MONTH'].map(month_to_num)
    input_df['MONTH_SIN'] = np.sin(2 * np.pi * input_df['MONTH_NUM'] / 12)
    input_df['MONTH_COS'] = np.cos(2 * np.pi * input_df['MONTH_NUM'] / 12)
    
    # Year features
    input_df['YEAR_NORMALIZED'] = (input_df['YEAR'] - year_min) / (year_max - year_min + 1)
    
    # Get overall productivity
    overall_productivity = feature_stats.get('overall_productivity', 
                          feature_stats.get('overall_productivity_median', 11.0))
    
    # Productivity features (use new fields if available, fallback to old)
    for idx in input_df.index:
        crop = input_df.loc[idx, 'CROP']
        muni = input_df.loc[idx, 'MUNICIPALITY']
        month = input_df.loc[idx, 'MONTH']
        season = input_df.loc[idx, 'SEASON']
        area = input_df.loc[idx, 'Area planted(ha)']
        key = f"{crop}_{muni}"
        key_month = f"{crop}_{muni}_{month}"
        key_season = f"{crop}_{season}"
        
        # Crop productivity
        crop_prod = feature_stats.get('crop_productivity', {}).get(crop, overall_productivity)
        input_df.loc[idx, 'CROP_PRODUCTIVITY'] = crop_prod
        
        # Municipality productivity
        muni_prod = feature_stats.get('muni_productivity', {}).get(muni, overall_productivity)
        input_df.loc[idx, 'MUNI_PRODUCTIVITY'] = muni_prod
        
        # Crop-Municipality productivity
        crop_muni_prod = feature_stats.get('crop_muni_productivity', {}).get(key, crop_prod)
        input_df.loc[idx, 'CROP_MUNI_PRODUCTIVITY'] = crop_muni_prod
        
        # Crop-Municipality mean (if available)
        crop_muni_mean = feature_stats.get('crop_muni_productivity', {}).get(key, crop_prod)
        input_df.loc[idx, 'CROP_MUNI_PRODUCTIVITY_MEAN'] = crop_muni_mean
        
        # Crop-Municipality-Month productivity
        crop_muni_month = feature_stats.get('crop_muni_month_productivity', {}).get(key_month, crop_muni_prod)
        input_df.loc[idx, 'CROP_MUNI_MONTH_PRODUCTIVITY'] = crop_muni_month
        
        # Crop-Season productivity
        crop_season = feature_stats.get('crop_season_productivity', {}).get(key_season, crop_prod)
        input_df.loc[idx, 'CROP_SEASON_PRODUCTIVITY'] = crop_season
        
        # Productivity CV
        prod_std = feature_stats.get('crop_muni_productivity_std', {}).get(key, 0)
        input_df.loc[idx, 'PRODUCTIVITY_CV'] = prod_std / (crop_muni_prod + 0.01)

        # Area is weak context for productivity, not a direct yield driver.
        area_reference = get_crop_area_reference(crop)
        input_df.loc[idx, 'AREA_CONTEXT'] = derive_area_context(area, area_reference)
        if 'LOG_AREA' in productivity_numerical_features:
            input_df.loc[idx, 'LOG_AREA'] = np.log1p(area)
        if 'AREA_RATIO' in productivity_numerical_features:
            input_df.loc[idx, 'AREA_RATIO'] = area / (area_reference + 0.01)
        
        # Year trend feature
        year_trend = feature_stats.get('year_trend', feature_stats.get('prod_trends', {})).get(key, 0.0)
        input_df.loc[idx, 'YEAR_TREND'] = year_trend
    
    return input_df


def calculate_features_legacy(input_df):
    """Calculate features for legacy production-based model"""
    input_df['MONTH'] = input_df['MONTH'].map(
        lambda x: month_full_to_abbr.get(x.upper(), x.upper()[:3])
    )
    input_df['SEASON'] = input_df['MONTH'].map(season_map)
    input_df['MONTH_NUM'] = input_df['MONTH'].map(month_to_num)
    input_df['MONTH_SIN'] = np.sin(2 * np.pi * input_df['MONTH_NUM'] / 12)
    input_df['MONTH_COS'] = np.cos(2 * np.pi * input_df['MONTH_NUM'] / 12)
    
    input_df['LOG_AREA'] = np.log1p(input_df['Area planted(ha)'])
    input_df['AREA_SQUARED'] = input_df['Area planted(ha)'] ** 2
    input_df['SQRT_AREA'] = np.sqrt(input_df['Area planted(ha)'])
    
    overall_mean = feature_stats.get('overall_mean', 100)
    
    for idx in input_df.index:
        crop = input_df.loc[idx, 'CROP']
        muni = input_df.loc[idx, 'MUNICIPALITY']
        month = input_df.loc[idx, 'MONTH']
        key = f"{crop}_{muni}"
        key_month = f"{crop}_{muni}_{month}"
        
        input_df.loc[idx, 'CROP_AVG'] = feature_stats['crop_avg'].get(crop, overall_mean)
        input_df.loc[idx, 'MUNI_AVG'] = feature_stats['muni_avg'].get(muni, overall_mean)
        input_df.loc[idx, 'CROP_MUNI_AVG'] = feature_stats['crop_muni_avg'].get(key, input_df.loc[idx, 'CROP_AVG'])
        input_df.loc[idx, 'CROP_MUNI_MONTH_AVG'] = feature_stats['crop_muni_month_avg'].get(key_month, input_df.loc[idx, 'CROP_MUNI_AVG'])
        input_df.loc[idx, 'PRODUCTIVITY_ESTIMATE'] = feature_stats.get('crop_productivity', {}).get(crop, 10)
        input_df.loc[idx, 'PROD_TREND'] = feature_stats.get('prod_trends', {}).get(key, 0)
        input_df.loc[idx, 'CROP_MUNI_STD'] = feature_stats.get('crop_muni_std', {}).get(key, 0)
    
    input_df['EXPECTED_PRODUCTION'] = input_df['Area planted(ha)'] * input_df['PRODUCTIVITY_ESTIMATE']
    input_df['LOG_EXPECTED_PROD'] = np.log1p(input_df['EXPECTED_PRODUCTION'])
    input_df['YEAR_NORMALIZED'] = (input_df['YEAR'] - year_min) / (year_max - year_min)
    input_df['YEARS_FROM_START'] = input_df['YEAR'] - year_min
    input_df['PRODUCTIVITY_AREA'] = input_df['PRODUCTIVITY_ESTIMATE'] * input_df['Area planted(ha)']
    input_df['PROD_CV'] = input_df['CROP_MUNI_STD'] / (input_df['CROP_MUNI_AVG'] + 1)
    
    crop_area_avg = feature_stats.get('crop_area_avg', {})
    for idx in input_df.index:
        crop = input_df.loc[idx, 'CROP']
        avg_area = crop_area_avg.get(crop, input_df.loc[idx, 'Area planted(ha)'])
        input_df.loc[idx, 'AREA_VS_CROP_AVG'] = input_df.loc[idx, 'Area planted(ha)'] / (avg_area + 0.01)
    
    input_df['PROD_VS_MUNI_AVG'] = input_df['CROP_MUNI_AVG'] / (input_df['MUNI_AVG'] + 0.01)
    input_df['CROP_SEASON'] = input_df['CROP'] + '_' + input_df['SEASON']
    
    return input_df


def get_prediction_with_confidence(input_df, area):
    """Get prediction with confidence intervals using productivity-first approach"""
    try:
        # Get predictions from RF model trees
        preprocessor = rf_model.named_steps['preprocessor']
        rf = rf_model.named_steps['model']
        
        X_transformed = preprocessor.transform(input_df)
        all_tree_preds = np.array([tree.predict(X_transformed) for tree in rf.estimators_])
        
        # These are productivity predictions
        prod_mean = np.mean(all_tree_preds, axis=0)
        prod_std = np.std(all_tree_preds, axis=0)
        
        # Convert to production
        production_mean = prod_mean * area
        production_std = prod_std * area
        
        # 95% CI
        lower_95 = max(0, production_mean - 1.96 * production_std)
        upper_95 = production_mean + 1.96 * production_std
        
        # Confidence score based on CV
        cv = prod_std / (prod_mean + 0.01)
        confidence = max(0.5, min(0.99, 1 - cv))
        
        return {
            'production': float(production_mean[0]) if hasattr(production_mean, '__len__') else float(production_mean),
            'productivity': float(prod_mean[0]) if hasattr(prod_mean, '__len__') else float(prod_mean),
            'lower_95': float(lower_95[0]) if hasattr(lower_95, '__len__') else float(lower_95),
            'upper_95': float(upper_95[0]) if hasattr(upper_95, '__len__') else float(upper_95),
            'confidence': float(confidence[0]) if hasattr(confidence, '__len__') else float(confidence)
        }
    except Exception as e:
        print(f"Confidence calculation error: {e}")
        return None


def model_is_ready():
    return model is not None and rf_model is not None and PRODUCTIVITY_FIRST


@app.route('/')
def home():
    return jsonify({
        'message': 'Benguet Crop Production ML API V2',
        'version': '2.0 (Productivity-First)',
        'official_ml_task': OFFICIAL_ML_TASK,
        'training_pipeline': metadata.get('training_pipeline', OFFICIAL_TRAINING_PIPELINE),
        'model': metadata.get('model_type', 'Unknown'),
        'prediction_target': 'PRODUCTIVITY (MT/HA)',
        'area_handling': metadata.get(
            'area_handling',
            'Planted area is weak context only; production is calculated from predicted productivity multiplied by planted area.'
        ),
        'performance': {
            'r2_score': f"{metadata.get('test_r2_score', 0):.4f}",
            'mae_productivity': f"{metadata.get('test_mae', 0):.2f} MT/HA"
        },
        'prediction_systems': {
            'planning_estimate': 'Live productivity-first estimate for user-provided crop, municipality, month, year, farm type, and planted area.',
            'trend_projection': 'Separate chart-oriented monthly projection endpoint; not the same as the live planning estimate model.'
        },
        'endpoints': {
            '/predict': 'POST - Single prediction',
            '/batch-predict': 'POST - Batch predictions',
            '/model-info': 'GET - Model information',
            '/forecast/monthly': 'GET - Trend projection for charts',
            '/crops': 'GET - Available crops',
            '/municipalities': 'GET - Available municipalities'
        }
    })


@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a single prediction - returns both productivity and production"""
    try:
        if not model_is_ready():
            return jsonify({
                'success': False,
                'error': 'Model artifacts not loaded. Check deploy logs for startup warnings.',
                'startup_errors': startup_errors
            }), 503

        raw_data = request.get_json(silent=True) or {}
        try:
            data = normalize_prediction_payload(raw_data)
        except KeyError as e:
            return jsonify({'error': f'Missing: {e.args[0]}'}), 400
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        year = data['year']
        area = data['area_planted']
        
        if area <= 0:
            return jsonify({'error': 'area_planted must be positive'}), 400
        
        # Normalize municipality name
        normalized_muni = normalize_municipality(data['municipality'])
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'MUNICIPALITY': [normalized_muni],
            'FARM TYPE': [data['farm_type'].upper()],
            'YEAR': [year],
            'MONTH': [data['month']],
            'CROP': [data['crop'].upper()],
            'Area planted(ha)': [area]
        })
        
        input_data = calculate_features_v2(input_data)

        # Predict productivity for the official planning task.
        pred_productivity = model.predict(input_data)[0]
        pred_productivity = np.clip(pred_productivity, 0.5, 50)
        pred_production = pred_productivity * area
        
        # Get confidence intervals
        confidence_data = get_prediction_with_confidence(input_data, area)
        
        # Get historical productivity for comparison (use normalized name)
        crop = data['crop'].upper()
        key = f"{crop}_{normalized_muni}"
        
        historical_productivity = feature_stats.get('crop_muni_productivity', 
                                  feature_stats.get('crop_productivity', {})).get(key, 
                                  feature_stats.get('crop_productivity', {}).get(crop))
        
        response = {
            'success': True,
            'prediction': {
                'system_type': 'planning_estimate',
                'production_mt': round(pred_production, 2),
                'predicted_production_mt': round(pred_production, 2),
                'productivity_mt_ha': round(pred_productivity, 2),
                'predicted_productivity_mt_ha': round(pred_productivity, 2),
                'area_planted_ha': area,
                'data_source': 'database' if DB_AVAILABLE else 'csv',
                'confidence_intervals': {}
            },
            'historical_comparison': {
                'historical_productivity_mt_ha': round(historical_productivity, 2) if historical_productivity else None,
                'predicted_productivity_mt_ha': round(pred_productivity, 2),
                'productivity_difference_percent': round(
                    (pred_productivity - historical_productivity) / historical_productivity * 100, 1
                ) if historical_productivity else None
            },
            'model_info': {
                'model_type': metadata.get('model_type', 'Unknown'),
                'official_ml_task': OFFICIAL_ML_TASK,
                'prediction_method': 'PRODUCTIVITY_FIRST',
                'r2_score': round(metadata.get('test_r2_score', 0), 4),
                'training_pipeline': metadata.get('training_pipeline', OFFICIAL_TRAINING_PIPELINE),
                'area_handling': metadata.get(
                    'area_handling',
                    'Planted area is weak context only; production is calculated from predicted productivity multiplied by planted area.'
                )
            },
            'input': data
        }
        
        if confidence_data:
            response['prediction']['confidence_intervals']['95%'] = {
                'lower': round(confidence_data['lower_95'], 2),
                'upper': round(confidence_data['upper_95'], 2)
            }
            response['prediction']['confidence_score'] = round(confidence_data['confidence'] * 100, 2)
            response['prediction']['confidence_type'] = 'heuristic_tree_dispersion'
            response['prediction']['confidence_calibrated'] = False
            response['prediction']['confidence_note'] = HEURISTIC_CONFIDENCE_NOTE
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/batch-predict', methods=['POST'])
@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch predictions"""
    try:
        if not model_is_ready():
            return jsonify({
                'success': False,
                'error': 'Model artifacts not loaded. Check deploy logs for startup warnings.',
                'startup_errors': startup_errors
            }), 503

        data = request.get_json()
        
        if 'predictions' not in data:
            return jsonify({'error': 'Missing predictions array'}), 400
        
        results = []
        for item in data['predictions']:
            try:
                normalized_item = normalize_prediction_payload(item)
                area = normalized_item['area_planted']
                normalized_muni = normalize_municipality(normalized_item['municipality'])
                
                input_data = pd.DataFrame({
                    'MUNICIPALITY': [normalized_muni],
                    'FARM TYPE': [normalized_item['farm_type']],
                    'YEAR': [normalized_item['year']],
                    'MONTH': [normalized_item['month']],
                    'CROP': [normalized_item['crop']],
                    'Area planted(ha)': [area]
                })
                
                input_data = calculate_features_v2(input_data)
                pred_productivity = np.clip(model.predict(input_data)[0], 0.5, 50)
                pred_production = pred_productivity * area
                
                results.append({
                    'system_type': 'planning_estimate',
                    'production_mt': round(pred_production, 2),
                    'predicted_production_mt': round(pred_production, 2),
                    'productivity_mt_ha': round(pred_productivity, 2),
                    'predicted_productivity_mt_ha': round(pred_productivity, 2),
                    'area_planted_ha': area,
                    'data_source': 'database' if DB_AVAILABLE else 'csv',
                    'input': normalized_item
                })
            except Exception as e:
                results.append({'error': str(e), 'input': item})
        
        return jsonify({
            'success': True,
            'count': len(results),
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': metadata.get('model_type', 'Unknown'),
        'official_ml_task': OFFICIAL_ML_TASK,
        'prediction_target': 'PRODUCTIVITY',
        'prediction_method': 'PRODUCTIVITY_FIRST',
        'training_pipeline': metadata.get('training_pipeline', OFFICIAL_TRAINING_PIPELINE),
        'deployment_source_policy': metadata.get('deployment_source_policy', 'scripted_training_only'),
        'area_handling': metadata.get(
            'area_handling',
            'Planted area is weak context only; production is calculated from predicted productivity multiplied by planted area.'
        ),
        'training_date': metadata.get('training_date', 'Unknown'),
        'performance': {
            'r2_score': metadata.get('test_r2_score', 0),
            'mae': metadata.get('test_mae', 0),
            'mape': metadata.get('test_mape', 0)
        },
        'n_samples': metadata.get('n_samples_train', 0) + metadata.get('n_samples_test', 0),
        'data_source': 'database' if DB_AVAILABLE else 'csv'
    })


@app.route('/crops', methods=['GET'])
def get_crops():
    """Get available crops"""
    if DB_AVAILABLE and db_manager:
        crops = db_manager.get_available_crops()
        if crops:
            return jsonify({'crops': crops, 'source': 'database'})
    
    crops = list(feature_stats.get('crop_productivity', feature_stats.get('crop_avg', {})).keys())
    return jsonify({'crops': sorted(crops), 'source': 'model'})


@app.route('/municipalities', methods=['GET'])
def get_municipalities():
    """Get available municipalities"""
    if DB_AVAILABLE and db_manager:
        municipalities = db_manager.get_available_municipalities()
        if municipalities:
            return jsonify({'municipalities': municipalities, 'source': 'database'})
    
    municipalities = list(feature_stats.get('muni_productivity', feature_stats.get('muni_avg', {})).keys())
    return jsonify({'municipalities': sorted(municipalities), 'source': 'model'})


@app.route('/productivity/<crop>', methods=['GET'])
def get_crop_productivity(crop):
    """Get historical productivity for a crop"""
    crop = crop.upper()
    municipality = request.args.get('municipality', '').upper()
    
    if municipality:
        key = f"{crop}_{municipality}"
        productivity = feature_stats.get('crop_muni_productivity', {}).get(key)
        std = feature_stats.get('crop_muni_productivity_std', {}).get(key, 0)
    else:
        productivity = feature_stats.get('crop_productivity', {}).get(crop)
        std = 0
    
    if productivity is None:
        return jsonify({'error': f'No data for {crop}'}), 404
    
    return jsonify({
        'crop': crop,
        'municipality': municipality if municipality else 'ALL',
        'productivity_mt_ha': round(productivity, 2),
        'std': round(std, 2) if std else None
    })


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    status = 'healthy' if model_is_ready() else 'degraded'
    return jsonify({
        'status': status,
        'model_loaded': model is not None,
        'rf_model_loaded': rf_model is not None,
        'database_connected': DB_AVAILABLE,
        'official_ml_task': OFFICIAL_ML_TASK,
        'prediction_method': 'PRODUCTIVITY_FIRST',
        'startup_errors': startup_errors
    })


@app.route('/api/available-options', methods=['GET'])
def get_available_options():
    """Backward-compatible endpoint for Laravel dropdown data."""
    return jsonify(build_available_options_response())


@app.route('/api/forecast', methods=['POST'])
def legacy_forecast():
    """Backward-compatible trend projection endpoint for Laravel integrations."""
    try:
        from forecast_aggregated import generate_monthly_forecast_aggregated

        data = request.get_json(silent=True) or {}
        crop = get_first_present(data, ['crop', 'CROP'])
        municipality = get_first_present(data, ['municipality', 'MUNICIPALITY'])
        forecast_years = int(get_first_present(data, ['forecast_years', 'FORECAST_YEARS'], 2))
        start_year = get_first_present(data, ['start_year', 'START_YEAR'])

        if not crop:
            return jsonify({'error': 'crop is required', 'success': False}), 400

        result = generate_monthly_forecast_aggregated(
            crop=str(crop).upper(),
            municipality=str(municipality).upper() if municipality else None,
            forecast_years=forecast_years,
            start_year=int(start_year) if start_year else None,
        )

        if not result.get('success'):
            return jsonify(result), 404

        result['system_type'] = 'trend_projection'
        result['projection_method'] = 'AGGREGATED_TREND_PROJECTION'
        result['is_same_as_planning_estimate'] = False
        result['note'] = 'This endpoint is a chart-oriented trend projection and is separate from the live productivity-first planning estimate.'
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/production/history', methods=['GET'])
def legacy_production_history():
    """Backward-compatible historical production endpoint."""
    try:
        page = max(1, int(request.args.get('page', 1)))
        limit = max(1, min(500, int(request.args.get('limit', 50))))
        filters = {
            'municipality': request.args.get('municipality'),
            'crop': request.args.get('crop'),
            'year': request.args.get('year'),
        }
        return jsonify(build_legacy_history_response(filters, page, limit))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def legacy_statistics():
    """Backward-compatible statistics endpoint."""
    return jsonify(build_legacy_statistics_response())


@app.route('/forecast/monthly', methods=['GET'])
def get_monthly_forecast():
    """
    Get monthly forecast for a crop with both historical and predicted data
    
    Query params:
        crop: Crop name (required)
        municipality: Municipality name (optional - if not provided, aggregates all)
        forecast_years: Number of years to forecast (default: 2)
        start_year: Starting year for forecast (optional)
    
    Returns:
        JSON with historical data, predicted historical overlay, and future forecasts
    """
    try:
        from forecast_aggregated import generate_monthly_forecast_aggregated
        
        crop = request.args.get('crop', '').upper()
        if not crop:
            return jsonify({'error': 'crop parameter is required'}), 400
        
        municipality = request.args.get('municipality', None)
        if municipality:
            municipality = municipality.upper()
        
        forecast_years = int(request.args.get('forecast_years', 2))
        start_year = request.args.get('start_year')
        if start_year:
            start_year = int(start_year)
        
        result = generate_monthly_forecast_aggregated(
            crop=crop,
            municipality=municipality,
            forecast_years=forecast_years,
            start_year=start_year
        )
        
        if not result['success']:
            return jsonify(result), 404

        result['system_type'] = 'trend_projection'
        result['projection_method'] = 'AGGREGATED_TREND_PROJECTION'
        result['is_same_as_planning_estimate'] = False
        result['note'] = 'This endpoint is a chart-oriented trend projection and is separate from the live productivity-first planning estimate.'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/forecast/methodology', methods=['GET'])
def get_forecast_methodology():
    """
    Get detailed explanation of how predictions are calculated.
    
    This endpoint is useful for the frontend to show users exactly 
    how the ML model arrives at a prediction for a specific month.
    
    Query params:
        crop: Crop name (required)
        month: Month abbreviation (required, e.g., 'JAN', 'FEB')
        municipality: Municipality name (optional)
    
    Returns:
        JSON with step-by-step methodology explanation and example data
    """
    try:
        from forecast_aggregated import get_methodology_explanation
        
        crop = request.args.get('crop', '').upper()
        month = request.args.get('month', '').upper()
        municipality = request.args.get('municipality', None)
        
        if not crop:
            return jsonify({'error': 'crop parameter is required'}), 400
        if not month:
            return jsonify({'error': 'month parameter is required'}), 400
        
        if municipality:
            municipality = municipality.upper()
        
        result = get_methodology_explanation(
            crop=crop,
            month=month,
            municipality=municipality
        )
        
        if not result.get('success', False):
            return jsonify(result), 404

        result['system_type'] = 'trend_projection'
        result['projection_method'] = 'AGGREGATED_TREND_PROJECTION'
        result['is_same_as_planning_estimate'] = False
        result['note'] = 'This methodology describes the chart projection logic, not the live productivity-first planning estimate model.'
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/data/summary', methods=['GET'])
def get_data_summary():
    """Get summary statistics"""
    return jsonify(build_legacy_statistics_response())


if __name__ == '__main__':
    import argparse
    port_from_env = os.environ.get('PORT', '5000')
    try:
        default_port = int(port_from_env)
    except ValueError:
        default_port = 5000

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=default_port)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BENGUET CROP PRODUCTION ML API V2")
    print("="*70)
    print(f"Model: {metadata.get('model_type', 'Unknown')}")
    print(f"Prediction: PRODUCTIVITY-FIRST ({OFFICIAL_ML_TASK})")
    print(f"R²: {metadata.get('test_r2_score', 0):.4f}")
    print(f"Database: {'Connected' if DB_AVAILABLE else 'Not available'}")
    print(f"Port: {args.port}")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=args.port, debug=False)
