# Prediction Accuracy Improvements

## Overview
Enhanced the ML API with year validation to ensure prediction accuracy by limiting forecasts to 5 years from the most recent historical data.

## Key Changes

### 1. **Year Validation Constants**
```python
MAX_HISTORICAL_YEAR = 2024  # Latest year in historical data
MAX_FORECAST_YEARS = 5      # Limit predictions to 5 years from historical data
MIN_PREDICTION_YEAR = 2025  # First valid prediction year
MAX_PREDICTION_YEAR = 2029  # Last valid prediction year (2024 + 5)
```

### 2. **Prediction Accuracy Rationale**
Machine learning models become less reliable when predicting too far into the future:
- **Historical data**: 2015-2024 (10 years)
- **Valid prediction range**: 2025-2029 (5 years maximum)
- **Reason**: Agricultural patterns, climate changes, and market conditions change over time. Limiting predictions to 5 years maintains accuracy while providing useful forecasts.

### 3. **Endpoints Updated**

#### `/api/predict` (Single Prediction)
- **Added validation**: Rejects years > 2029 or < 2015
- **Error message**: Provides clear guidance on valid year range
- **Example error response**:
```json
{
  "error": "Year 2035 is too far in the future. For accurate predictions, year must be between 2025 and 2029 (maximum 5 years from historical data ending at 2024).",
  "valid_range": {"min": 2025, "max": 2029},
  "historical_data_ends": 2024
}
```

#### `/api/forecast` (Pre-generated Forecasts)
- **Filters forecasts**: Only returns years 2025-2029
- **Added metadata**: Includes `forecast_range`, `historical_data_ends`, and `accuracy_note`
- **Response changes**:
```json
{
  "forecast": [/* only years 2025-2029 */],
  "metadata": {
    "forecast_years": 5,
    "forecast_range": "2025-2029",
    "historical_data_ends": 2024,
    "accuracy_note": "Predictions limited to 5 years from historical data for maximum accuracy"
  }
}
```

#### `/api/top-crops` (Top 5 Crops)
- **Validates target year**: Rejects years > 2029
- **Filters predictions**: Shows only valid forecast years
- **Added accuracy notes**: Informs users about prediction window

### 4. **Benefits**

✅ **Improved Accuracy**: Predictions stay within reliable timeframe
✅ **User Guidance**: Clear error messages explain valid ranges  
✅ **Data Integrity**: Prevents misleading long-term forecasts
✅ **Transparent**: Metadata shows prediction limitations
✅ **Flexible**: Easy to adjust MAX_FORECAST_YEARS if needed

### 5. **Testing Examples**

**Valid Request (2026)**:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"MUNICIPALITY":"ATOK","FARM_TYPE":"IRRIGATED","YEAR":2026,"MONTH":"JAN","CROP":"CABBAGE","Area_planted_ha":10}'
```

**Invalid Request (2035 - too far)**:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"MUNICIPALITY":"ATOK","FARM_TYPE":"IRRIGATED","YEAR":2035,"MONTH":"JAN","CROP":"CABBAGE","Area_planted_ha":10}'
```
Response: Error message with valid range guidance

### 6. **Configuration**
To adjust the prediction window, modify these constants in `ml_api.py`:
```python
MAX_FORECAST_YEARS = 5  # Change this to allow more/fewer years
```

## Impact
- **Predictions**: Limited to 2025-2029 (5 years from 2024)
- **Historical queries**: Still work for 2015-2024
- **Forecasts**: Automatically filtered to show only valid years
- **User experience**: Clear validation errors guide correct usage

## Version Compatibility
- **Python**: 3.14.0
- **scikit-learn**: 1.7.2 (upgraded from 1.3.2)
- **Historical data**: 2015-2024
- **Prediction window**: 2025-2029
