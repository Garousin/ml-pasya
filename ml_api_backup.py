"""
Benguet Crop Production ML API
Uses Extra Trees model with proper confidence intervals
Supports both database and CSV data sources
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os
import time
from datetime import datetime

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

print("Loading ML API...")

# Load the best model (Extra Trees)
model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)
with open(os.path.join(MODEL_DIR, 'feature_statistics.json'), 'r') as f:
    feature_stats = json.load(f)

# Also load RF model for confidence intervals
rf_model = joblib.load(os.path.join(MODEL_DIR, 'best_rf_model.pkl'))

print(f"[OK] Model loaded: {metadata['model_type']}")
print(f"[OK] R2 Score: {metadata['test_r2_score']:.4f}")
print(f"[OK] MAPE: {metadata['test_mape']:.2f}%")
print(f"[OK] Database: {'Connected' if DB_AVAILABLE else 'Not available (using CSV)'}")

# Feature engineering maps
season_map = metadata['feature_engineering']['season_map']
month_to_num = metadata['feature_engineering']['month_to_num']
year_min = metadata['feature_engineering']['year_min']
year_max = metadata['feature_engineering']['year_max']

# Month name to abbreviation mapping
month_full_to_abbr = {
    'JANUARY': 'JAN', 'FEBRUARY': 'FEB', 'MARCH': 'MAR', 'APRIL': 'APR',
    'MAY': 'MAY', 'JUNE': 'JUN', 'JULY': 'JUL', 'AUGUST': 'AUG',
    'SEPTEMBER': 'SEP', 'OCTOBER': 'OCT', 'NOVEMBER': 'NOV', 'DECEMBER': 'DEC',
    'JAN': 'JAN', 'FEB': 'FEB', 'MAR': 'MAR', 'APR': 'APR',
    'JUN': 'JUN', 'JUL': 'JUL', 'AUG': 'AUG', 'SEP': 'SEP', 'OCT': 'OCT', 'NOV': 'NOV', 'DEC': 'DEC'
}

def get_historical_data_range():
    """Get min and max years from historical data (database or CSV)"""
    if DB_AVAILABLE and db_manager:
        min_year, max_year = db_manager.get_historical_data_range()
        if min_year and max_year:
            return min_year, max_year
    
    # Fallback to CSV
    df = pd.read_csv('fulldataset.csv')
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df = df.dropna(subset=['YEAR'])
    return int(df['YEAR'].min()), int(df['YEAR'].max())

def calculate_features(input_df):
    """Calculate all features needed for prediction"""
    # Convert full month names to abbreviations
    input_df['MONTH'] = input_df['MONTH'].map(lambda x: month_full_to_abbr.get(x.upper(), x.upper()[:3]))
    
    # Season
    input_df['SEASON'] = input_df['MONTH'].map(season_map)
    
    # Month features
    input_df['MONTH_NUM'] = input_df['MONTH'].map(month_to_num)
    input_df['MONTH_SIN'] = np.sin(2 * np.pi * input_df['MONTH_NUM'] / 12)
    input_df['MONTH_COS'] = np.cos(2 * np.pi * input_df['MONTH_NUM'] / 12)
    
    # Area features
    input_df['LOG_AREA'] = np.log1p(input_df['Area planted(ha)'])
    input_df['AREA_SQUARED'] = input_df['Area planted(ha)'] ** 2
    input_df['SQRT_AREA'] = np.sqrt(input_df['Area planted(ha)'])
    
    # Historical averages from feature_stats
    overall_mean = feature_stats['overall_mean']
    
    for idx in input_df.index:
        crop = input_df.loc[idx, 'CROP']
        muni = input_df.loc[idx, 'MUNICIPALITY']
        month = input_df.loc[idx, 'MONTH']
        
        # Crop average
        input_df.loc[idx, 'CROP_AVG'] = feature_stats['crop_avg'].get(crop, overall_mean)
        
        # Municipality average
        input_df.loc[idx, 'MUNI_AVG'] = feature_stats['muni_avg'].get(muni, overall_mean)
        
        # Crop-Municipality average
        key = f"{crop}_{muni}"
        input_df.loc[idx, 'CROP_MUNI_AVG'] = feature_stats['crop_muni_avg'].get(key, input_df.loc[idx, 'CROP_AVG'])
        
        # Crop-Municipality-Month average
        key_month = f"{crop}_{muni}_{month}"
        input_df.loc[idx, 'CROP_MUNI_MONTH_AVG'] = feature_stats['crop_muni_month_avg'].get(key_month, input_df.loc[idx, 'CROP_MUNI_AVG'])
        
        # Productivity
        prod_key = f"{crop}_{muni}"
        productivity = feature_stats.get('prod_trends', {}).get(prod_key, 0)
        input_df.loc[idx, 'PRODUCTIVITY_ESTIMATE'] = feature_stats['crop_productivity'].get(crop, 10)
        
        # Trend
        input_df.loc[idx, 'PROD_TREND'] = feature_stats.get('prod_trends', {}).get(prod_key, 0)
        
        # Std
        input_df.loc[idx, 'CROP_MUNI_STD'] = feature_stats.get('crop_muni_std', {}).get(prod_key, 0)
    
    # Expected production
    input_df['EXPECTED_PRODUCTION'] = input_df['Area planted(ha)'] * input_df['PRODUCTIVITY_ESTIMATE']
    input_df['LOG_EXPECTED_PROD'] = np.log1p(input_df['EXPECTED_PRODUCTION'])
    
    # Year features
    input_df['YEAR_NORMALIZED'] = (input_df['YEAR'] - year_min) / (year_max - year_min)
    input_df['YEARS_FROM_START'] = input_df['YEAR'] - year_min
    
    # Productivity area
    input_df['PRODUCTIVITY_AREA'] = input_df['PRODUCTIVITY_ESTIMATE'] * input_df['Area planted(ha)']
    
    # Coefficient of variation
    input_df['PROD_CV'] = input_df['CROP_MUNI_STD'] / (input_df['CROP_MUNI_AVG'] + 1)
    
    # Area vs average
    crop_area_avg = feature_stats.get('crop_area_avg', {})
    for idx in input_df.index:
        crop = input_df.loc[idx, 'CROP']
        avg_area = crop_area_avg.get(crop, input_df.loc[idx, 'Area planted(ha)'])
        input_df.loc[idx, 'AREA_VS_CROP_AVG'] = input_df.loc[idx, 'Area planted(ha)'] / (avg_area + 0.01)
    
    # Production vs muni average
    input_df['PROD_VS_MUNI_AVG'] = input_df['CROP_MUNI_AVG'] / (input_df['MUNI_AVG'] + 0.01)
    
    # Interaction
    input_df['CROP_SEASON'] = input_df['CROP'] + '_' + input_df['SEASON']
    
    return input_df

def get_prediction_intervals(prediction, X, confidence=0.95):
    """
    Calculate prediction intervals using RF tree variance.
    Works with sklearn 1.3.2 trained models.
    """
    try:
        # Try RF tree-based confidence intervals first
        pipeline = rf_model.regressor_
        rf = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Transform input data
        X_transformed = preprocessor.transform(X)
        
        # Get predictions from all trees
        all_tree_preds = np.array([tree.predict(X_transformed) for tree in rf.estimators_])
        # Convert from log space
        all_tree_preds_original = np.expm1(all_tree_preds)
        
        # Calculate percentiles
        alpha = (1 - confidence) / 2
        lower = np.percentile(all_tree_preds_original, alpha * 100, axis=0)
        upper = np.percentile(all_tree_preds_original, (1 - alpha) * 100, axis=0)
        
        return lower, upper
        
    except Exception as e:
        print(f"RF CI error: {e}, falling back to MAPE-based CI")
        
        # Fallback to MAPE-based intervals
        try:
            crop = X['CROP'].iloc[0] if 'CROP' in X.columns else None
            muni = X['MUNICIPALITY'].iloc[0] if 'MUNICIPALITY' in X.columns else None
            
            if crop and muni:
                key = f"{crop}_{muni}"
                cv = feature_stats.get('crop_muni_std', {}).get(key, 0)
                avg = feature_stats.get('crop_muni_avg', {}).get(key, prediction)
                
                if cv > 0 and avg > 0:
                    rel_std = cv / avg
                else:
                    rel_std = metadata['test_mape'] / 100
            else:
                rel_std = metadata['test_mape'] / 100
            
            if confidence == 0.95:
                z_score = 1.96
            elif confidence == 0.68:
                z_score = 1.0
            else:
                z_score = 1.96
            
            margin = prediction * rel_std * z_score
            lower = max(0, prediction - margin)
            upper = prediction + margin
            
            return np.array([lower]), np.array([upper])
        except Exception as e2:
            print(f"Fallback CI error: {e2}")
            return None, None

@app.route('/')
def home():
    return jsonify({
        'message': 'Benguet Crop Production ML API',
        'version': '2.0',
        'model': metadata['model_type'],
        'performance': {
            'r2_score': f"{metadata['test_r2_score']:.4f}",
            'mape': f"{metadata['test_mape']:.2f}%",
            'mae': f"{metadata['test_mae']:.2f} MT"
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
    """Make a single prediction with confidence intervals"""
    try:
        data = request.get_json()
        
        # Required fields
        required = ['municipality', 'farm_type', 'year', 'month', 'crop', 'area_planted']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing: {field}'}), 400
        
        # Year validation
        min_year, max_year = get_historical_data_range()
        min_forecast = max_year + 1
        max_forecast = max_year + 5
        
        year = int(data['year'])
        if year < min_forecast or year > max_forecast:
            return jsonify({
                'error': f'Year must be {min_forecast}-{max_forecast}',
                'historical_range': f'{min_year}-{max_year}'
            }), 400
        
        # Create input
        input_data = pd.DataFrame({
            'MUNICIPALITY': [data['municipality'].upper()],
            'FARM TYPE': [data['farm_type'].upper()],
            'YEAR': [year],
            'MONTH': [data['month'].upper()],
            'CROP': [data['crop'].upper()],
            'Area planted(ha)': [float(data['area_planted'])]
        })
        
        # Calculate features
        input_data = calculate_features(input_data)
        
        # Predict
        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Confidence intervals (pass prediction value)
        lower_95, upper_95 = get_prediction_intervals(prediction, input_data, 0.95)
        lower_68, upper_68 = get_prediction_intervals(prediction, input_data, 0.68)
        
        # Get historical productivity for comparison
        area = float(data['area_planted'])
        predicted_productivity = prediction / area if area > 0 else 0
        
        # Get historical stats for this crop/municipality
        historical_productivity = None
        if DB_AVAILABLE and db_manager:
            try:
                hist_df = db_manager.get_crop_production_data({
                    'crop': data['crop'].upper(),
                    'municipality': data['municipality'].upper()
                })
                if not hist_df.empty:
                    historical_productivity = {
                        'mean': round(float(hist_df['PRODUCTIVITY'].mean()), 2),
                        'median': round(float(hist_df['PRODUCTIVITY'].median()), 2),
                        'min': round(float(hist_df['PRODUCTIVITY'].min()), 2),
                        'max': round(float(hist_df['PRODUCTIVITY'].max()), 2)
                    }
            except:
                pass
        
        response = {
            'success': True,
            'prediction': {
                'production_mt': round(prediction, 2),
                'productivity_mt_ha': round(predicted_productivity, 2),
                'confidence_intervals': {}
            },
            'historical_comparison': {
                'historical_productivity': historical_productivity,
                'predicted_productivity': round(predicted_productivity, 2),
                'difference_percent': round((predicted_productivity - historical_productivity['mean']) / historical_productivity['mean'] * 100, 1) if historical_productivity else None
            } if historical_productivity else None,
            'model_quality': {
                'r2_score': round(metadata['test_r2_score'], 4),
                'mape': f"{metadata['test_mape']:.2f}%",
                'description': f"Model explains {metadata['test_r2_score']*100:.1f}% of variance"
            },
            'input': data
        }
        
        if lower_95 is not None:
            response['prediction']['confidence_intervals']['95%'] = {
                'lower': round(max(0, lower_95[0]), 2),
                'upper': round(upper_95[0], 2),
                'margin': round((upper_95[0] - lower_95[0]) / 2, 2)
            }
        
        if lower_68 is not None:
            response['prediction']['confidence_intervals']['68%'] = {
                'lower': round(max(0, lower_68[0]), 2),
                'upper': round(upper_68[0], 2)
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch predictions"""
    try:
        data = request.get_json()
        
        if 'predictions' not in data:
            return jsonify({'error': 'Missing predictions array'}), 400
        
        min_year, max_year = get_historical_data_range()
        min_forecast = max_year + 1
        max_forecast = max_year + 5
        
        input_list = []
        for item in data['predictions']:
            year = int(item['year'])
            if year < min_forecast or year > max_forecast:
                return jsonify({'error': f'Year {year} out of range'}), 400
            
            input_list.append({
                'MUNICIPALITY': item['municipality'].upper(),
                'FARM TYPE': item['farm_type'].upper(),
                'YEAR': year,
                'MONTH': item['month'].upper(),
                'CROP': item['crop'].upper(),
                'Area planted(ha)': float(item['area_planted'])
            })
        
        input_df = pd.DataFrame(input_list)
        input_df = calculate_features(input_df)
        
        predictions = model.predict(input_df)
        predictions = np.maximum(0, predictions)
        
        lower_95, upper_95 = get_prediction_intervals(input_df, 0.95)
        
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'production_mt': round(pred, 2),
                'input': data['predictions'][i]
            }
            if lower_95 is not None:
                result['confidence_interval_95'] = {
                    'lower': round(max(0, lower_95[i]), 2),
                    'upper': round(upper_95[i], 2)
                }
            results.append(result)
        
        return jsonify({
            'success': True,
            'count': len(results),
            'predictions': results,
            'model_quality': {
                'r2_score': metadata['test_r2_score'],
                'mape': f"{metadata['test_mape']:.2f}%"
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': metadata['model_type'],
        'training_date': metadata['training_date'],
        'performance': {
            'r2_score': metadata['test_r2_score'],
            'mae': metadata['test_mae'],
            'mape': metadata['test_mape']
        },
        'n_samples': metadata['n_samples_train'] + metadata['n_samples_test'],
        'features': len(metadata['categorical_features']) + len(metadata['numerical_features']),
        'all_models': metadata.get('all_models_performance', {}),
        'data_source': 'database' if DB_AVAILABLE else 'csv'
    })

@app.route('/crops', methods=['GET'])
def get_crops():
    """Get available crops from database or feature stats"""
    if DB_AVAILABLE and db_manager:
        crops = db_manager.get_available_crops()
        if crops:
            return jsonify({'crops': crops, 'source': 'database'})
    return jsonify({'crops': list(feature_stats['crop_avg'].keys()), 'source': 'model'})

@app.route('/municipalities', methods=['GET'])
def get_municipalities():
    """Get available municipalities from database or feature stats"""
    if DB_AVAILABLE and db_manager:
        municipalities = db_manager.get_available_municipalities()
        if municipalities:
            return jsonify({'municipalities': municipalities, 'source': 'database'})
    return jsonify({'municipalities': list(feature_stats['muni_avg'].keys()), 'source': 'model'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'database_connected': DB_AVAILABLE
    })

@app.route('/data', methods=['GET'])
def get_data():
    """Get historical crop production data from database"""
    if not DB_AVAILABLE or not db_manager:
        return jsonify({'error': 'Database not available', 'source': 'csv_only'}), 503
    
    # Parse query parameters
    filters = {}
    if request.args.get('municipality'):
        filters['municipality'] = request.args.get('municipality').upper()
    if request.args.get('crop'):
        filters['crop'] = request.args.get('crop').upper()
    if request.args.get('year_from'):
        filters['year_from'] = int(request.args.get('year_from'))
    if request.args.get('year_to'):
        filters['year_to'] = int(request.args.get('year_to'))
    if request.args.get('month'):
        filters['month'] = request.args.get('month').upper()
    
    try:
        df = db_manager.get_crop_production_data(filters if filters else None)
        if df.empty:
            return jsonify({'data': [], 'count': 0})
        
        # Convert to list of dicts
        data = df.to_dict('records')
        return jsonify({
            'data': data,
            'count': len(data),
            'filters_applied': filters
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/data/historical-stats', methods=['GET'])
def get_historical_stats():
    """Get historical statistics for a specific crop/municipality combination"""
    if not DB_AVAILABLE or not db_manager:
        return jsonify({'error': 'Database not available'}), 503
    
    # Parse query parameters
    crop = request.args.get('crop', '').upper()
    municipality = request.args.get('municipality', '').upper()
    farm_type = request.args.get('farm_type', '').upper() if request.args.get('farm_type') else None
    
    if not crop:
        return jsonify({'error': 'crop parameter required'}), 400
    
    try:
        filters = {'crop': crop}
        if municipality:
            filters['municipality'] = municipality
        
        df = db_manager.get_crop_production_data(filters)
        
        if farm_type:
            df = df[df['FARM_TYPE'] == farm_type]
        
        if df.empty:
            return jsonify({'error': 'No data found', 'filters': filters}), 404
        
        # Calculate statistics
        stats = {
            'crop': crop,
            'municipality': municipality if municipality else 'ALL',
            'farm_type': farm_type if farm_type else 'ALL',
            'record_count': len(df),
            'year_range': {
                'min': int(df['YEAR'].min()),
                'max': int(df['YEAR'].max())
            },
            'productivity': {
                'mean': round(float(df['PRODUCTIVITY'].mean()), 2),
                'median': round(float(df['PRODUCTIVITY'].median()), 2),
                'min': round(float(df['PRODUCTIVITY'].min()), 2),
                'max': round(float(df['PRODUCTIVITY'].max()), 2),
                'std': round(float(df['PRODUCTIVITY'].std()), 2)
            },
            'production': {
                'mean': round(float(df['PRODUCTION'].mean()), 2),
                'median': round(float(df['PRODUCTION'].median()), 2),
                'total': round(float(df['PRODUCTION'].sum()), 2)
            },
            'area_planted': {
                'mean': round(float(df['AREA_PLANTED'].mean()), 2),
                'median': round(float(df['AREA_PLANTED'].median()), 2),
                'total': round(float(df['AREA_PLANTED'].sum()), 2)
            },
            'by_year': {}
        }
        
        # Yearly breakdown
        for year in sorted(df['YEAR'].unique()):
            year_df = df[df['YEAR'] == year]
            stats['by_year'][int(year)] = {
                'productivity_mean': round(float(year_df['PRODUCTIVITY'].mean()), 2),
                'production_total': round(float(year_df['PRODUCTION'].sum()), 2),
                'area_planted_total': round(float(year_df['AREA_PLANTED'].sum()), 2)
            }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/data/summary', methods=['GET'])
def get_data_summary():
    """Get summary statistics of the data"""
    if DB_AVAILABLE and db_manager:
        min_year, max_year = db_manager.get_historical_data_range()
        crops = db_manager.get_available_crops()
        municipalities = db_manager.get_available_municipalities()
        source = 'database'
    else:
        min_year, max_year = get_historical_data_range()
        crops = list(feature_stats['crop_avg'].keys())
        municipalities = list(feature_stats['muni_avg'].keys())
        source = 'csv'
    
    return jsonify({
        'year_range': {'min': min_year, 'max': max_year},
        'forecast_range': {'min': max_year + 1, 'max': max_year + 5},
        'crops_count': len(crops),
        'municipalities_count': len(municipalities),
        'crops': crops,
        'municipalities': municipalities,
        'data_source': source
    })

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BENGUET CROP PRODUCTION ML API")
    print("="*80)
    print(f"Model: {metadata['model_type']}")
    print(f"R2: {metadata['test_r2_score']:.4f} | MAPE: {metadata['test_mape']:.2f}%")
    print(f"Database: {'Connected' if DB_AVAILABLE else 'Not available (using CSV)'}")
    print(f"Running on port: {args.port}")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=args.port, debug=False)
