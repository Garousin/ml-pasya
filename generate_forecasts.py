"""
Generate and save all time-series forecasts to model_artifacts
This creates pre-computed forecasts for all crop/municipality combinations
Now includes MONTHLY forecasts using the aggregated prediction method
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from time_series_forecast import generate_forecast
from forecast_aggregated import generate_monthly_forecast_aggregated
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING TIME-SERIES FORECASTS FOR ALL COMBINATIONS")
print("="*80)

# Load data to get all combinations
df = pd.read_csv('fulldataset.csv')

# Get unique crops and municipalities
crops = sorted(df['CROP'].dropna().unique().tolist())
municipalities = sorted(df['MUNICIPALITY'].dropna().unique().tolist())

print(f"\nFound:")
print(f"  {len(crops)} crops")
print(f"  {len(municipalities)} municipalities")
print(f"  Total combinations: {len(crops) * len(municipalities)}")

# Generate forecasts for each combination
all_forecasts = {}
trends = {}
historical_aggregates = {}

forecast_years = 6  # 2025-2030 (6 years)
successful = 0
failed = 0

print(f"\nGenerating forecasts (this may take a few minutes)...")
print("-"*80)

for i, crop in enumerate(crops, 1):
    print(f"[{i}/{len(crops)}] Processing {crop}...")
    
    for municipality in municipalities:
        key = f"{crop}_{municipality}"
        
        try:
            result = generate_forecast(crop, municipality, forecast_years)
            
            if result['success']:
                # Save forecast data
                all_forecasts[key] = {
                    'crop': crop,
                    'municipality': municipality,
                    'forecast': result['forecast'],
                    'last_update': str(datetime.now())
                }
                
                # Save trend info
                trends[key] = {
                    'crop': crop,
                    'municipality': municipality,
                    'direction': result['trend']['direction'],
                    'growth_rate_percent': result['trend']['growth_rate_percent'],
                    'slope': result['trend']['slope']
                }
                
                # Save historical aggregates
                historical_aggregates[key] = {
                    'crop': crop,
                    'municipality': municipality,
                    'average': result['historical']['average'],
                    'min': result['historical']['min'],
                    'max': result['historical']['max'],
                    'last_year': result['historical']['last_year'],
                    'last_production': result['historical']['last_production'],
                    'years_available': len(result['historical']['years'])
                }
                
                successful += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            # print(f"  Warning: {crop} in {municipality} - {str(e)}")

print("-"*80)
print(f"\nResults:")
print(f"  Successful: {successful}")
print(f"  Failed (no data): {failed}")
print(f"  Total: {successful + failed}")

# Save to JSON files in model_artifacts
model_dir = 'model_artifacts'

# 1. Save all forecasts
print(f"\nSaving files to {model_dir}/...")
with open(f'{model_dir}/forecasts_all.json', 'w') as f:
    json.dump(all_forecasts, f, indent=2)
print(f"  ✓ forecasts_all.json ({len(all_forecasts)} forecasts)")

# 2. Save trends
with open(f'{model_dir}/trends.json', 'w') as f:
    json.dump(trends, f, indent=2)
print(f"  ✓ trends.json ({len(trends)} trends)")

# 3. Save historical aggregates
with open(f'{model_dir}/historical_aggregates.json', 'w') as f:
    json.dump(historical_aggregates, f, indent=2)
print(f"  ✓ historical_aggregates.json ({len(historical_aggregates)} aggregates)")

# 4. Create metadata file
metadata = {
    'generated_date': str(datetime.now()),
    'forecast_years': forecast_years,
    'total_forecasts': len(all_forecasts),
    'crops_included': crops,
    'municipalities_included': municipalities,
    'data_source': 'fulldataset.csv',
    'note': 'Pre-computed forecasts for all crop/municipality combinations'
}

with open(f'{model_dir}/forecast_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ forecast_metadata.json")

# 5. Generate MONTHLY forecasts for all crops (aggregated)
print("\n" + "="*80)
print("GENERATING MONTHLY AGGREGATED FORECASTS")
print("="*80)

monthly_forecasts = {}
for i, crop in enumerate(crops, 1):
    print(f"[{i}/{len(crops)}] Generating monthly forecast for {crop}...")
    try:
        result = generate_monthly_forecast_aggregated(crop, municipality=None, forecast_years=3)
        if result['success']:
            monthly_forecasts[crop] = result
    except Exception as e:
        print(f"  Warning: {crop} - {str(e)}")

# Save monthly forecasts
with open(f'{model_dir}/monthly_forecasts.json', 'w') as f:
    json.dump(monthly_forecasts, f, indent=2)
print(f"  ✓ monthly_forecasts.json ({len(monthly_forecasts)} crops)")

# Show some examples
print("\n" + "="*80)
print("EXAMPLE FORECASTS")
print("="*80)

# Show 3 random examples
import random
sample_keys = random.sample(list(all_forecasts.keys()), min(3, len(all_forecasts)))

for key in sample_keys:
    forecast_data = all_forecasts[key]
    trend_data = trends[key]
    hist_data = historical_aggregates[key]
    
    print(f"\n{forecast_data['crop']} in {forecast_data['municipality']}:")
    print(f"  Historical avg: {hist_data['average']:,.2f} MT")
    print(f"  Trend: {trend_data['direction']} ({trend_data['growth_rate_percent']:+.2f}% per year)")
    print(f"  Forecasts:")
    for f in forecast_data['forecast']:
        print(f"    {f['year']}: {f['production']:,.2f} MT")

# Show monthly forecast example
if monthly_forecasts:
    sample_crop = list(monthly_forecasts.keys())[0]
    mf = monthly_forecasts[sample_crop]
    print(f"\n--- Monthly Forecast Example: {sample_crop} ---")
    print(f"  Accuracy: {mf['accuracy']['production_accuracy_pct']:.1f}%")
    print(f"  Forecast (first 3 months of {mf['forecast_start_year']}):")
    for f in mf['forecast'][:3]:
        print(f"    {f['month']} {f['year']}: {f['production']:,.2f} MT, {f['productivity']:.2f} MT/ha")

print("\n" + "="*80)
print("FORECAST GENERATION COMPLETE!")
print("="*80)
print("\nFiles created in model_artifacts/:")
print("  1. forecasts_all.json - All forecast predictions (yearly)")
print("  2. trends.json - Trend analysis for all combinations")
print("  3. historical_aggregates.json - Historical statistics")
print("  4. forecast_metadata.json - Generation metadata")
print("  5. monthly_forecasts.json - Monthly forecasts with historical overlay")
print("\nYour web app can now read these files for instant forecasts!")
print("="*80)
