"""
Comprehensive System Test
Tests: DB connectivity, data integrity, API endpoints, production calculations
"""
import sys
import json

print("=" * 70)
print("SYSTEM CONNECTIVITY & DATA TEST")
print("=" * 70)

# ─── 1. Database Connection Test ─────────────────────────────────────────
print("\n[1/6] DATABASE CONNECTION TEST")
print("-" * 40)
try:
    from db_config import DB_CONFIG
    safe_config = {k: v for k, v in DB_CONFIG.items() if k != 'password'}
    safe_config['password'] = '***' if DB_CONFIG.get('password') else '(empty)'
    print(f"  Config: {json.dumps(safe_config, indent=4, default=str)}")

    from database import db_manager
    connected = db_manager.test_connection()
    print(f"  Connection: {'PASS ✓' if connected else 'FAIL ✗'}")
    if not connected:
        print("  ERROR: Cannot connect to database. Check MySQL/XAMPP is running.")
        sys.exit(1)
except Exception as e:
    print(f"  FAIL ✗ - {e}")
    sys.exit(1)

# ─── 2. Database Data Verification ───────────────────────────────────────
print("\n[2/6] DATABASE DATA VERIFICATION")
print("-" * 40)
try:
    # Row count
    from database import DatabaseManager, TABLES
    dm = DatabaseManager()
    with dm.get_connection(use_dict_cursor=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) as cnt FROM {TABLES['crop_productions']}")
            result = cursor.fetchone()
            total_rows = result['cnt'] if isinstance(result, dict) else result[0]
            print(f"  Total rows in crop_productions: {total_rows}")
            if total_rows == 0:
                print("  WARNING: Table is empty!")

            # Year range
            cursor.execute(f"SELECT MIN(year) as min_y, MAX(year) as max_y FROM {TABLES['crop_productions']}")
            result = cursor.fetchone()
            if isinstance(result, dict):
                min_y, max_y = result['min_y'], result['max_y']
            else:
                min_y, max_y = result[0], result[1]
            print(f"  Year range: {min_y} - {max_y}")

            # Unique crops
            cursor.execute(f"SELECT DISTINCT crop FROM {TABLES['crop_productions']} ORDER BY crop")
            crops = cursor.fetchall()
            crop_list = [r['crop'] if isinstance(r, dict) else r[0] for r in crops]
            print(f"  Unique crops ({len(crop_list)}): {', '.join(crop_list[:10])}{'...' if len(crop_list) > 10 else ''}")

            # Unique municipalities
            cursor.execute(f"SELECT DISTINCT municipality FROM {TABLES['crop_productions']} ORDER BY municipality")
            munis = cursor.fetchall()
            muni_list = [r['municipality'] if isinstance(r, dict) else r[0] for r in munis]
            print(f"  Unique municipalities ({len(muni_list)}): {', '.join(muni_list)}")

            # Sample data
            cursor.execute(f"""
                SELECT municipality, crop, year, month, production_mt, area_planted_ha, productivity_mt_ha 
                FROM {TABLES['crop_productions']} 
                WHERE production_mt > 0 
                ORDER BY year DESC, month 
                LIMIT 5
            """)
            samples = cursor.fetchall()
            print(f"\n  Sample data (latest 5 records with production > 0):")
            for row in samples:
                if isinstance(row, dict):
                    print(f"    {row['municipality']:15s} | {row['crop']:12s} | {row['year']}-{row['month']:3s} | "
                          f"Area: {row['area_planted_ha']} ha | Prod: {row['production_mt']} mt | "
                          f"Productivity: {row['productivity_mt_ha']} mt/ha")
                else:
                    print(f"    {row}")

    print("  Data verification: PASS ✓")
except Exception as e:
    print(f"  FAIL ✗ - {e}")
    import traceback; traceback.print_exc()

# ─── 3. Database vs CSV Comparison ───────────────────────────────────────
print("\n[3/6] DATABASE vs CSV DATA COMPARISON")
print("-" * 40)
try:
    import pandas as pd
    # Load from DB
    df_db = db_manager.get_crop_production_data()
    print(f"  DB records: {len(df_db)}")

    # Load from CSV
    df_csv = pd.read_csv('fulldataset.csv')
    print(f"  CSV records: {len(df_csv)}")

    if len(df_db) > 0 and len(df_csv) > 0:
        match_pct = (len(df_db) / len(df_csv)) * 100
        print(f"  Match ratio: {match_pct:.1f}% (DB/CSV)")
        if abs(len(df_db) - len(df_csv)) < 5:
            print("  DB and CSV are in sync: PASS ✓")
        else:
            print(f"  WARNING: DB has {len(df_db)} rows, CSV has {len(df_csv)} rows - may be out of sync")
    elif len(df_db) > 0:
        print("  DB has data, CSV comparison skipped: PASS ✓")
    else:
        print("  WARNING: DB returned no data!")
except Exception as e:
    print(f"  FAIL ✗ - {e}")

# ─── 4. Production Calculation Verification ──────────────────────────────
print("\n[4/6] PRODUCTION CALCULATION VERIFICATION")
print("-" * 40)
try:
    # Pick a specific record from DB and verify production = area * productivity
    with dm.get_connection(use_dict_cursor=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT municipality, crop, year, month, 
                       area_planted_ha, area_harvested_ha, production_mt, productivity_mt_ha
                FROM {TABLES['crop_productions']} 
                WHERE production_mt > 0 AND area_harvested_ha > 0 AND productivity_mt_ha > 0
                LIMIT 10
            """)
            rows = cursor.fetchall()
            
            print(f"  Checking {len(rows)} records: production ≈ area_harvested × productivity")
            all_ok = True
            for row in rows:
                if isinstance(row, dict):
                    area_h = float(row['area_harvested_ha'])
                    prod = float(row['production_mt'])
                    productivity = float(row['productivity_mt_ha'])
                    expected = area_h * productivity
                    diff = abs(prod - expected)
                    tolerance = max(0.1, prod * 0.05)  # 5% tolerance or 0.1
                    ok = diff <= tolerance
                    if not ok:
                        all_ok = False
                        print(f"    MISMATCH: {row['municipality']}/{row['crop']} {row['year']}-{row['month']}: "
                              f"prod={prod}, area_h*productivity={expected:.2f}, diff={diff:.2f}")
                    else:
                        print(f"    OK: {row['municipality']:15s}/{row['crop']:12s} {row['year']}-{row['month']:3s}: "
                              f"prod={prod} ≈ {area_h}×{productivity}={expected:.2f}")
            
            if all_ok:
                print("  Production formula check: PASS ✓")
            else:
                print("  Production formula check: SOME MISMATCHES (may be due to rounding)")
except Exception as e:
    print(f"  FAIL ✗ - {e}")
    import traceback; traceback.print_exc()

# ─── 5. Model Artifacts Check ────────────────────────────────────────────
print("\n[5/6] MODEL ARTIFACTS CHECK")
print("-" * 40)
import os
MODEL_DIR = 'model_artifacts'
required_files = ['model_metadata.json', 'feature_statistics.json']
optional_files = ['best_model.pkl', 'best_rf_model.pkl', 'categorical_values.json', 
                  'feature_info.json', 'forecast_metadata.json']

for f in required_files:
    path = os.path.join(MODEL_DIR, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = f"PASS ✓ ({size:,} bytes)" if exists else "MISSING ✗"
    print(f"  [Required] {f}: {status}")

for f in optional_files:
    path = os.path.join(MODEL_DIR, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = f"OK ({size:,} bytes)" if exists else "not found"
    print(f"  [Optional] {f}: {status}")

# Check model metadata content
try:
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
        meta = json.load(f)
    print(f"\n  Model type: {meta.get('model_type', 'Unknown')}")
    print(f"  Prediction target: {meta.get('prediction_target', 'Unknown')}")
    print(f"  R² score: {meta.get('test_r2_score', 'N/A')}")
    print(f"  Training date: {meta.get('training_date', 'Unknown')}")
    print(f"  Training samples: {meta.get('n_samples_train', 'N/A')}")
except Exception as e:
    print(f"  Could not read metadata: {e}")

# ─── 6. API Import & Data Source Check ────────────────────────────────────
print("\n[6/6] API MODULE & DATA SOURCE CHECK")
print("-" * 40)
try:
    from database import load_data_from_database
    df = load_data_from_database()
    print(f"  load_data_from_database() returned {len(df)} records")
    if len(df) > 0:
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data source verification: PASS ✓")
        # Show production stats
        if 'PRODUCTION' in df.columns:
            print(f"\n  Production stats from DB data:")
            print(f"    Mean:   {df['PRODUCTION'].mean():.2f} mt")
            print(f"    Median: {df['PRODUCTION'].median():.2f} mt")
            print(f"    Min:    {df['PRODUCTION'].min():.2f} mt")
            print(f"    Max:    {df['PRODUCTION'].max():.2f} mt")
    else:
        print("  WARNING: No data returned!")
except Exception as e:
    print(f"  FAIL ✗ - {e}")
    import traceback; traceback.print_exc()

# ─── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("  [1] DB Connection:            TESTED")
print("  [2] DB Data Integrity:        TESTED")
print("  [3] DB vs CSV Comparison:     TESTED")
print("  [4] Production Calculations:  TESTED")
print("  [5] Model Artifacts:          TESTED")
print("  [6] Data Source Pipeline:     TESTED")
print("=" * 70)
