# Dynamic Year Range Handling

## Overview
The ML API now **automatically adjusts** its prediction ranges based on the actual historical data. When you update `fulldataset.csv` with new years (e.g., 2025, 2026), the system automatically recalculates valid prediction windows **without any code changes**.

## How It Works

### 1. **Automatic Data Range Detection**
```python
def get_historical_data_range():
    """
    Dynamically determine the historical data range from the dataset.
    This automatically updates when historical data is refreshed.
    """
    try:
        df = pd.read_csv('fulldataset.csv')
        max_year = int(df['YEAR'].max())
        min_year = int(df['YEAR'].min())
        return min_year, max_year
    except Exception as e:
        # Fallback to defaults if CSV is unavailable
        return 2015, 2024
```

### 2. **Dynamic Range Calculation**
At startup, the API automatically:
- Scans `fulldataset.csv` to find the latest year
- Calculates prediction range as `MAX_HISTORICAL_YEAR + 1` to `MAX_HISTORICAL_YEAR + 5`
- Logs the ranges for transparency

**Current Example:**
```
Historical data range: 2015-2024
Valid prediction range: 2025-2029 (5 years)
```

**After updating to 2025 data:**
```
Historical data range: 2015-2025
Valid prediction range: 2026-2030 (5 years)
```

**After updating to 2026 data:**
```
Historical data range: 2015-2026
Valid prediction range: 2027-2031 (5 years)
```

### 3. **Configuration**
The system has one configurable parameter:
```python
MAX_FORECAST_YEARS = 5  # Maximum years to predict into the future
```

This ensures predictions stay within a reliable accuracy window. You can adjust this value if needed.

## Benefits

### ✅ **Zero Manual Updates**
When you add 2025 data to `fulldataset.csv`:
1. Simply restart the API
2. System automatically detects: 2015-2025 historical range
3. Prediction range updates to: 2026-2030
4. All endpoints work with new ranges immediately

### ✅ **Consistent Accuracy**
- Always maintains 5-year prediction window from latest data
- Prevents unreliable long-term forecasts
- ML models stay within their optimal accuracy range

### ✅ **Transparent Operation**
API startup logs show:
```
Historical data range: 2015-2024
Valid prediction range: 2025-2029 (5 years)
```

API responses include:
```json
{
  "error": "Year 2035 is too far...",
  "historical_data_ends": 2024,
  "note": "Prediction range automatically updates when historical data is refreshed"
}
```

### ✅ **Database Support**
For scalable API (`ml_api_scalable.py`), the system can read from:
- **Database**: `SELECT MIN(year), MAX(year) FROM crop_productions`
- **CSV Fallback**: Reads from `fulldataset.csv` if database unavailable

## Update Workflow

### Scenario: Adding 2025 Historical Data

**Step 1: Update Data**
```csv
# Add 2025 records to fulldataset.csv
MUNICIPALITY,FARM TYPE,YEAR,MONTH,CROP,Area planted(ha),Production(mt)
ATOK,IRRIGATED,2025,JAN,CABBAGE,15.5,245.2
...
```

**Step 2: Restart API**
```powershell
# Stop current API
Ctrl+C

# Restart
python ml_api.py
```

**Step 3: Automatic Detection**
```
Historical data range: 2015-2025  ← Auto-detected
Valid prediction range: 2026-2030 (5 years)  ← Auto-calculated
```

**Step 4: Verify**
```bash
curl http://127.0.0.1:5000/api/health
```
Response:
```json
{
  "status": "healthy",
  "data_range": {
    "historical": "2015-2025",
    "prediction": "2026-2030",
    "max_forecast_years": 5
  }
}
```

## API Endpoints Affected

### `/api/health`
Shows current data ranges:
```json
{
  "data_range": {
    "historical": "2015-2024",
    "prediction": "2025-2029",
    "max_forecast_years": 5
  }
}
```

### `/api/predict`
Validates year against dynamic ranges:
```json
// Request for 2026 when data ends at 2024
{
  "error": "Year 2026 is valid",
  "prediction": {...}
}

// After adding 2025 data, request for 2031 when data ends at 2025
{
  "error": "Year 2031 is too far in the future...",
  "valid_range": {"min": 2026, "max": 2030},
  "historical_data_ends": 2025
}
```

### `/api/forecast`
Filters forecasts to valid range:
```json
{
  "forecast": [/* only shows 2025-2029 or 2026-2030 depending on historical data */],
  "metadata": {
    "forecast_range": "2025-2029",  // Auto-updated
    "historical_data_ends": 2024,    // Auto-detected
    "auto_update_note": "Prediction range automatically adjusts..."
  }
}
```

### `/api/top-crops`
Uses dynamic year validation:
```json
// Request for specific year
{
  "MUNICIPALITY": "ATOK",
  "YEAR": 2028  // Validated against current prediction range
}
```

## Fallback Mechanism

If the CSV file is missing or corrupted:
```python
# Logs warning
"Warning: Could not read historical data range, using defaults: [error]"

# Uses safe defaults
MIN_HISTORICAL_YEAR = 2015
MAX_HISTORICAL_YEAR = 2024
```

This prevents API crashes when data sources are unavailable.

## Advanced: Database Integration

For `ml_api_scalable.py` with database enabled:

```python
def get_historical_data_range():
    if model_config.USE_DATABASE:
        # Query database for year range
        min_year, max_year = data_layer.get_year_range()
        return min_year, max_year
    else:
        # Fallback to CSV
        df = pd.read_csv('fulldataset.csv')
        return int(df['YEAR'].min()), int(df['YEAR'].max())
```

**Database Method:**
```python
def get_year_range(self) -> Tuple[int, int]:
    from sqlalchemy import func
    result = session.query(
        func.min(CropProduction.year),
        func.max(CropProduction.year)
    ).first()
    return int(result[0]), int(result[1])
```

## Testing Dynamic Updates

**Test 1: Check Current Range**
```bash
curl http://127.0.0.1:5000/api/health | python -m json.tool
```

**Test 2: Try Future Year (Should Accept)**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"MUNICIPALITY":"ATOK","FARM_TYPE":"IRRIGATED","YEAR":2026,"MONTH":"JAN","CROP":"CABBAGE","Area_planted_ha":10}'
```

**Test 3: Try Too-Far Year (Should Reject)**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"MUNICIPALITY":"ATOK","FARM_TYPE":"IRRIGATED","YEAR":2035,"MONTH":"JAN","CROP":"CABBAGE","Area_planted_ha":10}'
```

**Test 4: Add 2025 Data & Restart**
```powershell
# 1. Add 2025 records to fulldataset.csv
# 2. Restart API
python ml_api.py

# 3. Verify new range in startup logs
# Historical data range: 2015-2025
# Valid prediction range: 2026-2030 (5 years)

# 4. Verify with health endpoint
curl http://127.0.0.1:5000/api/health
```

## Summary

| Feature | Before (Static) | After (Dynamic) |
|---------|----------------|-----------------|
| **Year limits** | Hardcoded 2024/2029 | Auto-detected from data |
| **Data updates** | Required code changes | Just restart API |
| **Prediction window** | Fixed 2025-2029 | Always 5 years from latest data |
| **Maintenance** | Manual updates needed | Zero maintenance |
| **Transparency** | Hidden in code | Logged and exposed via API |
| **Accuracy** | Could drift out of sync | Always aligned with data |

## Key Takeaway

**No code changes needed when you add new historical data** - the system automatically adjusts all year validations, ranges, and forecasts based on the actual data in your CSV or database.
