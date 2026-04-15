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

# Month name to abbreviation mapping
month_full_to_abbr = {
    'JANUARY': 'JAN', 'FEBRUARY': 'FEB', 'MARCH': 'MAR', 'APRIL': 'APR',
    'MAY': 'MAY', 'JUNE': 'JUN', 'JULY': 'JUL', 'AUGUST': 'AUG',
    'SEPTEMBER': 'SEP', 'OCTOBER': 'OCT', 'NOVEMBER': 'NOV', 'DECEMBER': 'DEC',
    'JAN': 'JAN', 'FEB': 'FEB', 'MAR': 'MAR', 'APR': 'APR',
    'JUN': 'JUN', 'JUL': 'JUL', 'AUG': 'AUG', 'SEP': 'SEP', 
    'OCT': 'OCT', 'NOV': 'NOV', 'DEC': 'DEC'
}

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
        
        # Area features
        area_median = feature_stats.get('crop_area_median', feature_stats.get('crop_area_avg', {})).get(crop, 5)
        input_df.loc[idx, 'LOG_AREA'] = np.log1p(area)
        input_df.loc[idx, 'AREA_RATIO'] = area / (area_median + 0.01)
        
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
    return model is not None and rf_model is not None


@app.route('/')
def home():
    return jsonify({
        'message': 'Benguet Crop Production ML API V2',
        'version': '2.0 (Productivity-First)',
        'model': metadata.get('model_type', 'Unknown'),
        'prediction_target': 'PRODUCTIVITY (MT/HA)',
        'performance': {
            'r2_score': f"{metadata.get('test_r2_score', 0):.4f}",
            'mae_productivity': f"{metadata.get('test_mae', 0):.2f} MT/HA"
        },
        'endpoints': {
            '/predict': 'POST - Single prediction',
            '/batch-predict': 'POST - Batch predictions',
            '/model-info': 'GET - Model information',
            '/crops': 'GET - Available crops',
            '/municipalities': 'GET - Available municipalities'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Make a single prediction - returns both productivity and production"""
    try:
        if not model_is_ready():
            return jsonify({
                'success': False,
                'error': 'Model artifacts not loaded. Check deploy logs for startup warnings.',
                'startup_errors': startup_errors
            }), 503

        data = request.get_json()
        
        required = ['municipality', 'farm_type', 'year', 'month', 'crop', 'area_planted']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing: {field}'}), 400
        
        year = int(data['year'])
        area = float(data['area_planted'])
        
        if area <= 0:
            return jsonify({'error': 'area_planted must be positive'}), 400
        
        # Normalize municipality name
        normalized_muni = normalize_municipality(data['municipality'])
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'MUNICIPALITY': [normalized_muni],
            'FARM TYPE': [data['farm_type'].upper()],
            'YEAR': [year],
            'MONTH': [data['month'].upper()],
            'CROP': [data['crop'].upper()],
            'Area planted(ha)': [area]
        })
        
        # Calculate features based on model type
        if PRODUCTIVITY_FIRST:
            input_data = calculate_features_v2(input_data)
            
            # Predict productivity
            pred_productivity = model.predict(input_data)[0]
            pred_productivity = np.clip(pred_productivity, 0.5, 50)  # Reasonable bounds
            
            # Calculate production
            pred_production = pred_productivity * area
        else:
            input_data = calculate_features_legacy(input_data)
            pred_production = model.predict(input_data)[0]
            pred_production = max(0, pred_production)
            pred_productivity = pred_production / area if area > 0 else 0
        
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
                'production_mt': round(pred_production, 2),
                'productivity_mt_ha': round(pred_productivity, 2),
                'area_planted_ha': area,
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
                'prediction_method': 'PRODUCTIVITY_FIRST' if PRODUCTIVITY_FIRST else 'PRODUCTION',
                'r2_score': round(metadata.get('test_r2_score', 0), 4)
            },
            'input': data
        }
        
        if confidence_data:
            response['prediction']['confidence_intervals']['95%'] = {
                'lower': round(confidence_data['lower_95'], 2),
                'upper': round(confidence_data['upper_95'], 2)
            }
            response['prediction']['confidence_score'] = round(confidence_data['confidence'] * 100, 2)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/batch-predict', methods=['POST'])
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
                area = float(item.get('area_planted', 1))
                normalized_muni = normalize_municipality(item['municipality'])
                
                input_data = pd.DataFrame({
                    'MUNICIPALITY': [normalized_muni],
                    'FARM TYPE': [item['farm_type'].upper()],
                    'YEAR': [int(item['year'])],
                    'MONTH': [item['month'].upper()],
                    'CROP': [item['crop'].upper()],
                    'Area planted(ha)': [area]
                })
                
                if PRODUCTIVITY_FIRST:
                    input_data = calculate_features_v2(input_data)
                    pred_productivity = np.clip(model.predict(input_data)[0], 0.5, 50)
                    pred_production = pred_productivity * area
                else:
                    input_data = calculate_features_legacy(input_data)
                    pred_production = max(0, model.predict(input_data)[0])
                    pred_productivity = pred_production / area if area > 0 else 0
                
                results.append({
                    'production_mt': round(pred_production, 2),
                    'productivity_mt_ha': round(pred_productivity, 2),
                    'area_planted_ha': area,
                    'input': item
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
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': metadata.get('model_type', 'Unknown'),
        'prediction_target': 'PRODUCTIVITY' if PRODUCTIVITY_FIRST else 'PRODUCTION',
        'prediction_method': metadata.get('prediction_method', 'STANDARD'),
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
def health():
    status = 'healthy' if model_is_ready() else 'degraded'
    return jsonify({
        'status': status,
        'model_loaded': model is not None,
        'rf_model_loaded': rf_model is not None,
        'database_connected': DB_AVAILABLE,
        'prediction_method': 'PRODUCTIVITY_FIRST' if PRODUCTIVITY_FIRST else 'PRODUCTION',
        'startup_errors': startup_errors
    })


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
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/data/summary', methods=['GET'])
def get_data_summary():
    """Get summary statistics"""
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
    
    return jsonify({
        'year_range': {'min': min_year, 'max': max_year},
        'forecast_range': {'min': max_year + 1, 'max': max_year + 5},
        'crops_count': len(crops),
        'municipalities_count': len(municipalities),
        'crops': sorted(crops),
        'municipalities': sorted(municipalities),
        'data_source': source
    })


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
    print(f"Prediction: {'PRODUCTIVITY-FIRST' if PRODUCTIVITY_FIRST else 'PRODUCTION'}")
    print(f"R²: {metadata.get('test_r2_score', 0):.4f}")
    print(f"Database: {'Connected' if DB_AVAILABLE else 'Not available'}")
    print(f"Port: {args.port}")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=args.port, debug=False)
