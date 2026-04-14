# 🚀 Quick Start Guide - Scalable ML API

## Choose Your Setup Path

### Path 1: File-Only Mode (Fastest - No Database)
✅ Best for: Quick testing, development, single server  
⏱️ Setup time: 2 minutes

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment file
cp .env.example .env

# 3. Edit .env - set these values:
#    USE_DATABASE=false
notepad .env

# 4. Run the API
python ml_api_scalable.py
```

**Test it:**
```powershell
curl http://127.0.0.1:5000/api/health
```

---

### Path 2: Database Mode (Recommended for Production)
✅ Best for: Production, Laravel integration, multi-server  
⏱️ Setup time: 10 minutes

#### Step 1: Setup Database (XAMPP MySQL)

```powershell
# Start XAMPP, then create database
mysql -u root -p
```

```sql
CREATE DATABASE benguet_crops CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE benguet_crops;
SOURCE database/schema.sql;
EXIT;
```

#### Step 2: Install Python Dependencies

```powershell
pip install -r requirements.txt
```

#### Step 3: Configure Environment

```powershell
cp .env.example .env
notepad .env
```

**Set these values in `.env`:**
```env
USE_DATABASE=true
DB_TYPE=mysql
DB_HOST=localhost
DB_PORT=3306
DB_NAME=benguet_crops
DB_USER=root
DB_PASSWORD=
```

#### Step 4: Migrate Data (Optional)

```powershell
python database/migrate_csv_to_db.py --csv fulldataset.csv
```

This imports your CSV data into the database.

#### Step 5: Run the API

```powershell
python ml_api_scalable.py
```

**Test it:**
```powershell
curl http://127.0.0.1:5000/api/health
```

---

### Path 3: Automated Setup (Interactive Wizard)

```powershell
.\setup.ps1
```

This interactive script will guide you through all setup steps.

---

## 🧪 Test the API

### 1. Check Health
```powershell
curl http://127.0.0.1:5000/api/health
```

### 2. Get Available Options
```powershell
curl http://127.0.0.1:5000/api/available-options
```

### 3. Make a Prediction
```powershell
curl -X POST http://127.0.0.1:5000/api/predict `
  -H "Content-Type: application/json" `
  -d '{\"MUNICIPALITY\":\"ATOK\",\"FARM_TYPE\":\"IRRIGATED\",\"YEAR\":2024,\"MONTH\":\"JAN\",\"CROP\":\"CABBAGE\",\"Area_planted_ha\":10.5}'
```

### 4. Get a Forecast
```powershell
curl -X POST http://127.0.0.1:5000/api/forecast `
  -H "Content-Type: application/json" `
  -d '{\"CROP\":\"BROCCOLI\",\"MUNICIPALITY\":\"ATOK\"}'
```

---

## 🔗 Integrate with Laravel

### 1. Add to Laravel's `.env`
```env
ML_API_URL=http://127.0.0.1:5000
ML_API_TIMEOUT=30
ML_API_CACHE_ENABLED=true
```

### 2. Update `config/services.php`
```php
'ml_api' => [
    'url' => env('ML_API_URL', 'http://127.0.0.1:5000'),
    'timeout' => env('ML_API_TIMEOUT', 30),
    'cache_enabled' => env('ML_API_CACHE_ENABLED', true),
],
```

### 3. Copy ML API Service
Copy `laravel_integration/MLApiService.php` to your Laravel app:
```powershell
cp laravel_integration/MLApiService.php path/to/laravel/app/Services/
```

### 4. Use in Controller
```php
use App\Services\MLApiService;

class CropController extends Controller
{
    protected $mlApi;
    
    public function __construct(MLApiService $mlApi)
    {
        $this->mlApi = $mlApi;
    }
    
    public function predict(Request $request)
    {
        $result = $this->mlApi->predict([
            'municipality' => $request->municipality,
            'farm_type' => $request->farm_type,
            'year' => $request->year,
            'month' => $request->month,
            'crop' => $request->crop,
            'area_planted_ha' => $request->area_planted
        ]);
        
        return response()->json($result);
    }
}
```

---

## 🎯 Common Use Cases

### Use Case 1: Get Dropdown Options for Forms
```php
$mlApi = new MLApiService();
$options = $mlApi->getAvailableOptions();

// Returns:
// {
//   "municipalities": ["ATOK", "BAKUN", ...],
//   "crops": ["BROCCOLI", "CABBAGE", ...],
//   "farm_types": ["IRRIGATED", "RAINFED"],
//   "months": [...]
// }
```

### Use Case 2: Predict Crop Production
```php
$prediction = $mlApi->predict([
    'municipality' => 'ATOK',
    'farm_type' => 'IRRIGATED',
    'year' => 2024,
    'month' => 'JAN',
    'crop' => 'CABBAGE',
    'area_planted_ha' => 10.5
]);

// Returns:
// {
//   "success": true,
//   "prediction": {
//     "production_mt": 123.45,
//     "confidence_score": 0.8765
//   },
//   ...
// }
```

### Use Case 3: Get Multi-Year Forecast
```php
$forecast = $mlApi->getForecast('BROCCOLI', 'ATOK');

// Returns:
// {
//   "success": true,
//   "crop": "BROCCOLI",
//   "municipality": "ATOK",
//   "forecasts": [
//     {"year": 2025, "production": 150.2},
//     {"year": 2026, "production": 155.8},
//     ...
//   ],
//   "trend": {
//     "direction": "increasing",
//     "growth_rate_percent": 3.5
//   }
// }
```

### Use Case 4: Batch Predictions (Multiple at Once)
```php
$results = $mlApi->batchPredict([
    [
        'municipality' => 'ATOK',
        'farm_type' => 'IRRIGATED',
        'year' => 2024,
        'month' => 'JAN',
        'crop' => 'CABBAGE',
        'area_planted_ha' => 10.5
    ],
    [
        'municipality' => 'BAKUN',
        'farm_type' => 'RAINFED',
        'year' => 2024,
        'month' => 'FEB',
        'crop' => 'BROCCOLI',
        'area_planted_ha' => 8.2
    ]
]);

// Returns predictions for all inputs
```

### Use Case 5: Get Historical Data with Filters
```php
$history = $mlApi->getProductionHistory([
    'municipality' => 'ATOK',
    'crop' => 'CABBAGE',
    'year' => 2023,
    'page' => 1,
    'limit' => 50
]);

// Returns paginated historical records
```

---

## 🔧 Configuration Options

### Performance Tuning

**Enable Caching (Recommended)**
```env
CACHE_ENABLED=true
CACHE_TYPE=memory  # or redis
```

**Connection Pool Settings**
```env
DB_POOL_SIZE=10  # More connections for high traffic
DB_POOL_MAX_OVERFLOW=20
```

**Pagination Limits**
```env
DEFAULT_PAGE_SIZE=50
MAX_PAGE_SIZE=1000
```

### Feature Flags

```env
ENABLE_BATCH_PREDICTIONS=true
ENABLE_FORECASTING=true
ENABLE_REQUEST_LOGGING=true
```

---

## 📊 Monitoring

### View Prediction Logs (if using database)
```sql
SELECT 
    crop,
    COUNT(*) as total_predictions,
    AVG(predicted_production_mt) as avg_prediction
FROM prediction_logs
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY crop
ORDER BY total_predictions DESC;
```

### Check API Performance
```sql
SELECT 
    DATE(created_at) as date,
    COUNT(*) as requests,
    AVG(processing_time_ms) as avg_time_ms,
    MAX(processing_time_ms) as max_time_ms
FROM prediction_logs
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY DATE(created_at);
```

---

## ❓ Troubleshooting

### Issue: "Cannot connect to database"
**Solution:**
1. Check XAMPP MySQL is running
2. Verify credentials in `.env`
3. Or set `USE_DATABASE=false` to use file mode

### Issue: "Module not found"
**Solution:**
```powershell
pip install -r requirements.txt --upgrade
```

### Issue: Slow API responses
**Solution:**
1. Enable caching: `CACHE_ENABLED=true`
2. Increase connection pool: `DB_POOL_SIZE=10`
3. Use database instead of CSV files

### Issue: sklearn version warnings
**Solution:**
The API includes compatibility shims. For best results:
```powershell
pip install scikit-learn==1.6.1
```

---

## 📚 Next Steps

1. **Read Full Documentation:** `README_SCALABLE.md`
2. **Explore API Endpoints:** Test all endpoints with Postman or curl
3. **Integrate with Laravel:** Use `MLApiService.php`
4. **Enable Monitoring:** Set up prediction logs and analytics
5. **Deploy to Production:** See deployment checklist in README

---

## 🆘 Get Help

- Check logs: Set `LOG_LEVEL=DEBUG` in `.env`
- Test health: `curl http://127.0.0.1:5000/api/health`
- Review documentation: `README_SCALABLE.md`

**Happy forecasting! 🌾**
