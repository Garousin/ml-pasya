"""Scaling and multi-crop validation test"""
import requests
import json

base = 'http://127.0.0.1:5001'
print('SCALING & MULTI-CROP VALIDATION')
print('=' * 65)

# Scaling test
print('\nLinear Scaling (CABBAGE, ATOK, JAN 2025):')
results = {}
for area in [1, 5, 10, 50, 100]:
    r = requests.post(f'{base}/predict', json={
        'municipality': 'ATOK', 'farm_type': 'IRRIGATED',
        'year': 2025, 'month': 'JAN', 'crop': 'CABBAGE', 'area_planted': area
    })
    d = r.json()['prediction']
    results[area] = d
    print(f"  {area:>4} ha -> {d['productivity_mt_ha']:>6.2f} MT/HA x {area} ha = {d['production_mt']:>8.2f} MT")

base_prod = results[1]['production_mt']
print(f"\n  Scaling check vs 1 ha ({base_prod:.2f} MT):")
for area in [5, 10, 50, 100]:
    ratio = results[area]['production_mt'] / base_prod
    print(f"    {area:>4} ha: {ratio:>6.2f}x (ideal: {area}x)")

# Multi-crop test
print('\nMulti-crop predictions (10 ha, ATOK, JAN 2025):')
for crop in ['CABBAGE', 'CARROTS', 'LETTUCE', 'WHITE POTATO', 'BROCCOLI', 'CHINESE CABBAGE']:
    r = requests.post(f'{base}/predict', json={
        'municipality': 'ATOK', 'farm_type': 'IRRIGATED',
        'year': 2025, 'month': 'JAN', 'crop': crop, 'area_planted': 10
    })
    d = r.json()
    pred = d['prediction']
    hist = d['historical_comparison']
    print(f"  {crop:<18}: {pred['productivity_mt_ha']:>6.2f} MT/HA | Hist: {hist['historical_productivity_mt_ha']:>6.2f} | Prod: {pred['production_mt']:>7.2f} MT")

# Multi-municipality test
print('\nMulti-municipality (CABBAGE, 10 ha, JAN 2025):')
for muni in ['ATOK', 'BUGUIAS', 'LATRINIDAD', 'BOKOD', 'KABAYAN']:
    r = requests.post(f'{base}/predict', json={
        'municipality': muni, 'farm_type': 'IRRIGATED',
        'year': 2025, 'month': 'JAN', 'crop': 'CABBAGE', 'area_planted': 10
    })
    d = r.json()
    pred = d['prediction']
    hist = d['historical_comparison']
    print(f"  {muni:<15}: {pred['productivity_mt_ha']:>6.2f} MT/HA | Hist: {hist['historical_productivity_mt_ha']:>6.2f} | Prod: {pred['production_mt']:>7.2f} MT")

print('\n' + '=' * 65)
print('ALL VALIDATION TESTS DONE')
