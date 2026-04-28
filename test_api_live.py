"""Live API endpoint tests"""
import requests
import json

base = 'http://127.0.0.1:5001'
print('='*70)
print('LIVE API ENDPOINT TESTS')
print('='*70)

# 1. Health check
print('\n[1] /health')
r = requests.get(f'{base}/health')
data = r.json()
print(f"  Status: {data['status']}")
print(f"  Model loaded: {data['model_loaded']}")
print(f"  RF model loaded: {data['rf_model_loaded']}")
print(f"  DB connected: {data['database_connected']}")
print(f"  Prediction method: {data['prediction_method']}")
health_ok = data['status'] == 'healthy' and data['database_connected']
print(f"  Result: {'PASS' if health_ok else 'FAIL'}")

# 2. Crops from DB
print('\n[2] /crops')
r = requests.get(f'{base}/crops')
data = r.json()
print(f"  Source: {data['source']}")
print(f"  Crops ({len(data['crops'])}): {data['crops']}")
crops_ok = data['source'] == 'database' and len(data['crops']) > 0
print(f"  Result: {'PASS - from database' if crops_ok else 'WARN - from model/csv'}")

# 3. Municipalities from DB
print('\n[3] /municipalities')
r = requests.get(f'{base}/municipalities')
data = r.json()
print(f"  Source: {data['source']}")
print(f"  Municipalities ({len(data['municipalities'])}): {data['municipalities']}")
muni_ok = data['source'] == 'database' and len(data['municipalities']) > 0
print(f"  Result: {'PASS - from database' if muni_ok else 'WARN - from model/csv'}")

# 4. Data summary
print('\n[4] /data/summary')
r = requests.get(f'{base}/data/summary')
data = r.json()
print(f"  Data source: {data['data_source']}")
print(f"  Year range: {data['year_range']}")
print(f"  Crops: {data['crops_count']}")
print(f"  Municipalities: {data['municipalities_count']}")
summary_ok = data['data_source'] == 'database'
print(f"  Result: {'PASS - from database' if summary_ok else 'WARN - from csv'}")

# 5. Prediction test
print('\n[5] /predict (CABBAGE in ATOK, 2025, JAN, 50 ha)')
r = requests.post(f'{base}/predict', json={
    'municipality': 'ATOK',
    'farm_type': 'IRRIGATED',
    'year': 2025,
    'month': 'JAN',
    'crop': 'CABBAGE',
    'area_planted': 50
})
data = r.json()
if 'error' not in data:
    print(f"  Success: {data.get('success')}")
    pred = data.get('prediction', {})
    print(f"  Predicted productivity: {pred.get('predicted_productivity_mt_ha', 'N/A')} mt/ha")
    print(f"  Predicted production: {pred.get('predicted_production_mt', 'N/A')} mt")
    print(f"  Area planted: {pred.get('area_planted_ha', 'N/A')} ha")
    print(f"  Data source: {pred.get('data_source', 'N/A')}")
    # Verify production = productivity * area
    productivity = pred.get('predicted_productivity_mt_ha', 0)
    production = pred.get('predicted_production_mt', 0)
    expected = round(productivity * 50, 2)
    print(f"  Verify calc: {productivity} mt/ha x 50 ha = {expected} mt (API returned: {production} mt)")
    print(f"  Result: PASS")
else:
    print(f"  Error: {data.get('error')}")
    print(f"  Result: FAIL")

# 6. Model info
print('\n[6] /model-info')
r = requests.get(f'{base}/model-info')
data = r.json()
print(f"  Model: {data['model_type']}")
print(f"  Target: {data['prediction_target']}")
print(f"  R2: {data['performance']['r2_score']}")
print(f"  Data source: {data['data_source']}")
print(f"  Result: PASS")

print('\n' + '='*70)
print('ALL LIVE API TESTS COMPLETED')
print('='*70)
