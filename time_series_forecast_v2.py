"""
Time-series forecasting module V2 - Monthly ML-based predictions
Generates monthly forecasts using the trained ML model for both 
Production and Productivity
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = 'model_artifacts'

# Load models and metadata
model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
rf_model = joblib.load(os.path.join(MODEL_DIR, 'best_rf_model.pkl'))

with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)
with open(os.path.join(MODEL_DIR, 'feature_statistics.json'), 'r') as f:
    feature_stats = json.load(f)

# Feature engineering maps
season_map = metadata['feature_engineering']['season_map']
month_to_num = metadata['feature_engineering']['month_to_num']
year_min = metadata['feature_engineering']['year_min']
year_max = metadata['feature_engineering']['year_max']

MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

PRODUCTIVITY_FIRST = metadata.get('prediction_target') == 'PRODUCTIVITY'


def calculate_features_v2(input_df):
    """Calculate features for productivity-first model"""
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
    
    # Productivity features
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
        input_df.loc[idx, 'CROP_MUNI_PRODUCTIVITY_MEAN'] = crop_muni_prod
        
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
    
    return input_df


def get_historical_monthly_data(crop, municipality, df):
    """Get historical monthly data for a crop-municipality combination"""
    filtered = df[(df['CROP'] == crop.upper()) & 
                  (df['MUNICIPALITY'] == municipality.upper())].copy()
    
    if len(filtered) == 0:
        return None
    
    # Clean numeric columns
    filtered['Production(mt)'] = pd.to_numeric(filtered['Production(mt)'], errors='coerce')
    filtered['Productivity(mt/ha)'] = pd.to_numeric(filtered['Productivity(mt/ha)'], errors='coerce')
    filtered['Area planted(ha)'] = pd.to_numeric(filtered['Area planted(ha)'], errors='coerce')
    filtered['YEAR'] = pd.to_numeric(filtered['YEAR'], errors='coerce')
    
    filtered = filtered.dropna(subset=['Production(mt)', 'YEAR'])
    
    if len(filtered) == 0:
        return None
    
    # Aggregate by year-month (sum production, average productivity)
    monthly = filtered.groupby(['YEAR', 'MONTH']).agg({
        'Production(mt)': 'sum',
        'Productivity(mt/ha)': 'mean',
        'Area planted(ha)': 'sum'
    }).reset_index()
    
    return monthly


def get_average_area_by_month(crop, municipality, df):
    """Get average planted area by month for predictions"""
    filtered = df[(df['CROP'] == crop.upper()) & 
                  (df['MUNICIPALITY'] == municipality.upper())].copy()
    
    if len(filtered) == 0:
        return {m: 5.0 for m in MONTHS}  # Default if no data
    
    filtered['Area planted(ha)'] = pd.to_numeric(filtered['Area planted(ha)'], errors='coerce')
    filtered = filtered.dropna(subset=['Area planted(ha)'])
    
    # Get average area by month
    avg_area = filtered.groupby('MONTH')['Area planted(ha)'].mean().to_dict()
    
    # Fill missing months with overall average
    overall_avg = filtered['Area planted(ha)'].mean()
    for m in MONTHS:
        if m not in avg_area:
            avg_area[m] = overall_avg
    
    return avg_area


def generate_monthly_forecast(crop, municipality, start_year=2025, end_year=2027):
    """
    Generate monthly forecasts using the ML model
    
    Args:
        crop: Crop name
        municipality: Municipality name
        start_year: First year to forecast
        end_year: Last year to forecast
    
    Returns:
        dict with historical and forecast data at monthly granularity
    """
    
    # Load data
    df = pd.read_csv('fulldataset.csv')
    
    crop = crop.upper()
    municipality = municipality.upper()
    
    # Get historical monthly data
    historical = get_historical_monthly_data(crop, municipality, df)
    
    if historical is None or len(historical) == 0:
        return {
            'success': False,
            'error': f'No data found for {crop} in {municipality}'
        }
    
    # Get average area by month for predictions
    avg_area_by_month = get_average_area_by_month(crop, municipality, df)
    
    # Get last year of historical data
    last_hist_year = int(historical['YEAR'].max())
    
    # Generate predictions for each month in forecast period
    forecasts = []
    
    for year in range(start_year, end_year + 1):
        for month in MONTHS:
            # Get area for this month (use historical average)
            area = avg_area_by_month.get(month, 5.0)
            
            # Create input dataframe for ML model
            input_data = pd.DataFrame({
                'MUNICIPALITY': [municipality],
                'FARM TYPE': ['IRRIGATED'],  # Default, as it's most common
                'YEAR': [year],
                'MONTH': [month],
                'CROP': [crop],
                'Area planted(ha)': [area]
            })
            
            # Calculate features
            input_data = calculate_features_v2(input_data)
            
            # Make prediction
            if PRODUCTIVITY_FIRST:
                # Model predicts productivity, calculate production
                pred_productivity = model.predict(input_data)[0]
                pred_productivity = np.clip(pred_productivity, 0.5, 50)  # Reasonable bounds
                pred_production = pred_productivity * area
            else:
                # Model predicts production directly
                pred_production = model.predict(input_data)[0]
                pred_production = max(0, pred_production)
                pred_productivity = pred_production / area if area > 0 else 0
            
            forecasts.append({
                'year': year,
                'month': month,
                'month_num': MONTHS.index(month) + 1,
                'production': round(float(pred_production), 2),
                'productivity': round(float(pred_productivity), 2),
                'area_planted': round(float(area), 2),
                'is_forecast': True
            })
    
    # Format historical data
    historical_list = []
    for _, row in historical.iterrows():
        month = row['MONTH']
        month_num = MONTHS.index(month) + 1 if month in MONTHS else month_to_num.get(month, 1)
        historical_list.append({
            'year': int(row['YEAR']),
            'month': row['MONTH'],
            'month_num': month_num,
            'production': round(float(row['Production(mt)']), 2),
            'productivity': round(float(row['Productivity(mt/ha)']), 2) if pd.notna(row['Productivity(mt/ha)']) else None,
            'area_planted': round(float(row['Area planted(ha)']), 2) if pd.notna(row['Area planted(ha)']) else None,
            'is_forecast': False
        })
    
    # Sort historical by year and month
    historical_list.sort(key=lambda x: (x['year'], x['month_num']))
    
    # Calculate statistics
    hist_production = [h['production'] for h in historical_list]
    hist_productivity = [h['productivity'] for h in historical_list if h['productivity'] is not None]
    
    return {
        'success': True,
        'crop': crop,
        'municipality': municipality,
        'historical': historical_list,
        'forecast': forecasts,
        'statistics': {
            'historical_years': f"{int(historical['YEAR'].min())}-{last_hist_year}",
            'forecast_years': f"{start_year}-{end_year}",
            'avg_production': round(np.mean(hist_production), 2),
            'avg_productivity': round(np.mean(hist_productivity), 2) if hist_productivity else None,
            'min_production': round(min(hist_production), 2),
            'max_production': round(max(hist_production), 2)
        },
        'model_info': {
            'model_type': metadata.get('model_type', 'Unknown'),
            'prediction_method': 'PRODUCTIVITY_FIRST' if PRODUCTIVITY_FIRST else 'PRODUCTION',
            'r2_score': round(metadata.get('test_r2_score', 0), 4)
        },
        'generated_at': str(datetime.now())
    }


def generate_yearly_forecast(crop, municipality, forecast_years=3):
    """
    Generate yearly aggregated forecasts (for backward compatibility)
    Uses monthly ML predictions aggregated by year
    """
    
    # Get current year
    current_year = datetime.now().year
    
    # Generate monthly forecasts
    result = generate_monthly_forecast(
        crop, 
        municipality, 
        start_year=current_year,
        end_year=current_year + forecast_years - 1
    )
    
    if not result['success']:
        return result
    
    # Aggregate monthly forecasts by year
    forecasts_by_year = {}
    for f in result['forecast']:
        year = f['year']
        if year not in forecasts_by_year:
            forecasts_by_year[year] = {
                'production_sum': 0,
                'productivity_values': [],
                'count': 0
            }
        forecasts_by_year[year]['production_sum'] += f['production']
        forecasts_by_year[year]['productivity_values'].append(f['productivity'])
        forecasts_by_year[year]['count'] += 1
    
    yearly_forecasts = []
    for year in sorted(forecasts_by_year.keys()):
        data = forecasts_by_year[year]
        yearly_forecasts.append({
            'year': year,
            'production': round(data['production_sum'], 2),
            'productivity': round(np.mean(data['productivity_values']), 2),
            'months_forecasted': data['count']
        })
    
    # Similarly aggregate historical
    historical_by_year = {}
    for h in result['historical']:
        year = h['year']
        if year not in historical_by_year:
            historical_by_year[year] = {
                'production_sum': 0,
                'productivity_values': [],
                'count': 0
            }
        historical_by_year[year]['production_sum'] += h['production']
        if h['productivity'] is not None:
            historical_by_year[year]['productivity_values'].append(h['productivity'])
        historical_by_year[year]['count'] += 1
    
    yearly_historical = []
    for year in sorted(historical_by_year.keys()):
        data = historical_by_year[year]
        yearly_historical.append({
            'year': year,
            'production': round(data['production_sum'], 2),
            'productivity': round(np.mean(data['productivity_values']), 2) if data['productivity_values'] else None
        })
    
    return {
        'success': True,
        'crop': crop,
        'municipality': municipality,
        'historical': yearly_historical,
        'forecast': yearly_forecasts,
        'statistics': result['statistics'],
        'model_info': result['model_info'],
        'generated_at': str(datetime.now())
    }


# Test the function
if __name__ == '__main__':
    print("="*80)
    print("TIME-SERIES FORECAST V2 - MONTHLY ML PREDICTIONS")
    print("="*80)
    
    # Test: CABBAGE in ATOK (matching the graph example)
    crop = 'CABBAGE'
    municipality = 'ATOK'
    
    result = generate_monthly_forecast(crop, municipality, start_year=2025, end_year=2027)
    
    if result['success']:
        print(f"\nCrop: {result['crop']}")
        print(f"Municipality: {result['municipality']}")
        print(f"Model: {result['model_info']['model_type']}")
        print(f"Prediction Method: {result['model_info']['prediction_method']}")
        print(f"R² Score: {result['model_info']['r2_score']}")
        
        print(f"\n--- Historical Data ({result['statistics']['historical_years']}) ---")
        print(f"Total records: {len(result['historical'])}")
        print(f"Avg Production: {result['statistics']['avg_production']:,.2f} MT")
        print(f"Avg Productivity: {result['statistics']['avg_productivity']} MT/ha")
        
        print(f"\n--- Last 6 Historical Months ---")
        for h in result['historical'][-6:]:
            print(f"  {h['month']} {h['year']}: {h['production']:,.2f} MT | {h['productivity']} MT/ha")
        
        print(f"\n--- Forecasts ({result['statistics']['forecast_years']}) ---")
        print(f"Total predictions: {len(result['forecast'])}")
        
        print(f"\n--- First 12 Forecast Months ---")
        for f in result['forecast'][:12]:
            print(f"  {f['month']} {f['year']}: {f['production']:,.2f} MT | {f['productivity']:.2f} MT/ha")
        
        print("\n" + "="*80)
        
        # Also test yearly aggregation
        print("\n--- YEARLY AGGREGATED FORECAST ---")
        yearly_result = generate_yearly_forecast(crop, municipality, forecast_years=3)
        
        if yearly_result['success']:
            print(f"\nHistorical (yearly):")
            for h in yearly_result['historical'][-5:]:
                print(f"  {h['year']}: {h['production']:,.2f} MT | {h['productivity']} MT/ha")
            
            print(f"\nForecasts (yearly):")
            for f in yearly_result['forecast']:
                print(f"  {f['year']}: {f['production']:,.2f} MT | {f['productivity']:.2f} MT/ha")
    else:
        print(f"\nError: {result['error']}")
    
    print("\n" + "="*80)
