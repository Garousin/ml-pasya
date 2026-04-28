# ML Model → Laravel Integration

This directory contains everything you need to integrate your ML model with Laravel.

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Start the API
```powershell
.\start_api.ps1
```

Or manually:
```powershell
python ml_api.py
```

### 3. Test the API
In a new terminal:
```powershell
.\test_api.ps1
```

### 4. Integrate with Laravel
Follow the detailed guide in `LARAVEL_INTEGRATION_GUIDE.md`

---

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `ml_api.py` | Flask API server for ML model |
| `requirements.txt` | Python dependencies |
| `start_api.ps1` | PowerShell script to start the API |
| `test_api.ps1` | PowerShell script to test the API |
| `LARAVEL_INTEGRATION_GUIDE.md` | Complete Laravel integration guide |
| `model_artifacts/` | Trained model files (pkl, json) |

---

## 🔗 API Endpoints

Once running on `http://127.0.0.1:5000`:

### GET `/api/health`
Check if API is running
```json
{
  "status": "healthy",
  "model_type": "Random Forest Regressor",
  "training_date": "2025-11-02 00:11:54"
}
```

### GET `/api/available-options`
Get dropdown options for the form
```json
{
  "municipalities": ["ATOK", "BAKUN", ...],
  "farm_types": ["IRRIGATED", "RAINFED"],
  "crops": ["CABBAGE", "CARROTS", ...],
  "months": [{"value": 1, "label": "January"}, ...]
}
```

### POST `/api/predict`
Make a single prediction
```json
{
  "MUNICIPALITY": "ATOK",
  "FARM_TYPE": "IRRIGATED",
  "YEAR": 2024,
  "MONTH": 1,
  "CROP": "CABBAGE",
  "Area_planted_ha": 10.5,
  "Area_harvested_ha": 10.0,
  "Productivity_mt_ha": 15.5
}
```

Response:
```json
{
  "success": true,
  "prediction": {
    "production_mt": 155.32,
    "expected_from_productivity": 155.00,
    "difference": 0.32,
    "confidence_score": 0.9888
  }
}
```

---

## 🔧 Laravel Integration Steps

### 1. Create Service Class
```bash
php artisan make:service CropPredictionService
```
Copy code from `LARAVEL_INTEGRATION_GUIDE.md` → Section 2.1

### 2. Add Configuration
Edit `config/services.php` and `.env` (see Section 2.2)

### 3. Create Controller
```bash
php artisan make:controller CropPredictionController
```
Copy code from Section 2.3

### 4. Add Routes
Edit `routes/web.php` (see Section 2.4)

### 5. Create View
Create `resources/views/predictions/index.blade.php` (see Section 2.5)

---

## 💡 Usage in Laravel

```php
use App\Services\CropPredictionService;

$service = new CropPredictionService();

// Check if API is healthy
$health = $service->healthCheck();

// Get available options
$options = $service->getAvailableOptions();

// Make a prediction
$result = $service->predict([
    'municipality' => 'ATOK',
    'farm_type' => 'IRRIGATED',
    'year' => 2024,
    'month' => 1,
    'crop' => 'CABBAGE',
    'area_planted' => 10.5,
    'area_harvested' => 10.0,
    'productivity' => 15.5
]);

echo "Predicted: " . $result['prediction']['production_mt'] . " mt";
```

---

## 🐛 Troubleshooting

### API won't start
- ✅ Check Python is installed: `python --version`
- ✅ Install dependencies: `pip install -r requirements.txt`
- ✅ Verify model files exist in `model_artifacts/`

### Laravel can't connect
- ✅ Make sure Flask API is running
- ✅ Check `ML_API_URL` in Laravel `.env`
- ✅ Test API manually: visit `http://127.0.0.1:5000/api/health`

### Predictions are errors
- ✅ Verify input data format matches API requirements
- ✅ Check municipality/crop names are uppercase
- ✅ Ensure values are in valid ranges

---

## 📚 Documentation

- **Complete Guide**: `LARAVEL_INTEGRATION_GUIDE.md`
- **Official Training Pipeline**: `retrain_model_optimized.py`
- **Research Notebook**: `UPDATED ML MODEL.ipynb` (analysis only, not a deployment source)
- **Feature Analysis**: Other markdown files in this directory

---

## 🔄 Workflow

```
1. Train Model (`retrain_model_optimized.py`)
   ↓
2. Export Model Files (model_artifacts/)
   ↓
3. Start Flask API (ml_api.py)
   ↓
4. Integrate with Laravel (CropPredictionService)
   ↓
5. Users make predictions via Laravel UI
```

---

## 🚀 Production Deployment

For production, use Gunicorn instead of Flask development server:

```bash
pip install gunicorn
gunicorn -w 4 -b 127.0.0.1:5000 ml_api:app
```

Or run as a Windows service using NSSM (see guide for details).

---

## 📈 Model Performance

- **Model**: Random Forest (tuned)
- **Official Task**: Productivity-first planning estimate
- **Training Pipeline**: `retrain_model_optimized.py`
- **R² Score**: 0.8145
- **MAE**: 0.86 MT/HA
- **Production MAPE**: 148.1%
- **Target**: Productivity (MT/HA), then production = productivity × planted area

---

## 🎯 Next Steps

1. ✅ Start the Flask API
2. ✅ Test API endpoints
3. ✅ Create Laravel service
4. ✅ Build prediction form
5. ✅ Store predictions in database
6. ✅ Add user authentication
7. ✅ Deploy to production

---

## 💬 Support

For detailed instructions, see `LARAVEL_INTEGRATION_GUIDE.md`

For deployment model details, inspect `model_artifacts/model_metadata.json` and `model_artifacts/evaluation_report.json`
