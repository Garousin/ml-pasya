# System Architecture Diagram

## How Everything Connects

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Your Laravel Application UI                   │  │
│  │  (Blade Views, Forms, Dashboard, etc.)               │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          │ HTTP Request                     │
│                          │ (Form Submission)                │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   LARAVEL APPLICATION                       │
│                   (Your PHP Backend)                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Routes (routes/api.php)                  │  │
│  │  - POST /api/ml/predict                              │  │
│  │  - GET  /api/ml/options                              │  │
│  │  - GET  /api/ml/health                               │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Controllers                                   │  │
│  │  (CropPredictionController.php)                      │  │
│  │  - Validation                                         │  │
│  │  - Request handling                                   │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Service Layer                                 │  │
│  │  (CropPredictionService.php)                         │  │
│  │  - HTTP client                                        │  │
│  │  - Business logic                                     │  │
│  │  - Error handling                                     │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          │ HTTP POST                        │
│                          │ JSON Data                        │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FLASK API (Python Server)                      │
│              Running on http://127.0.0.1:5000               │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              ml_api.py (Flask App)                    │  │
│  │                                                       │  │
│  │  Endpoints:                                           │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │ GET  /api/health         → Check status        │ │  │
│  │  │ GET  /api/available-options → Get dropdowns    │ │  │
│  │  │ POST /api/predict        → Single prediction   │ │  │
│  │  │ POST /api/batch-predict  → Batch predictions   │ │  │
│  │  │ GET  /api/model-info     → Model metadata      │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Data Processing                               │  │
│  │  - Validate input                                     │  │
│  │  - Transform to DataFrame                             │  │
│  │  - Preprocess features                                │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      ML Model (Random Forest)                         │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  best_rf_model.pkl                              │ │  │
│  │  │  - Trained model                                │ │  │
│  │  │  - 200 estimators                               │ │  │
│  │  │  - 98.88% accuracy                              │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  preprocessor.pkl                               │ │  │
│  │  │  - Feature transformation                       │ │  │
│  │  │  - OneHotEncoder + StandardScaler              │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  model_metadata.json                            │ │  │
│  │  │  categorical_values.json                        │ │  │
│  │  │  feature_info.json                              │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Prediction Result                             │  │
│  │  {                                                    │  │
│  │    "production_mt": 155.32,                          │  │
│  │    "confidence_score": 0.9888                        │  │
│  │  }                                                    │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          │ JSON Response                    │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   LARAVEL APPLICATION                       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      Response Processing                              │  │
│  │  - Parse JSON                                         │  │
│  │  - Store in database (optional)                       │  │
│  │  - Format for display                                 │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│                          │ Return to User                   │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Display Results                               │  │
│  │  - Predicted Production: 155.32 mt                   │  │
│  │  - Confidence: 98.88%                                │  │
│  │  - Charts and visualizations                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow Example

### 1. User Submits Form
```javascript
{
  municipality: "ATOK",
  farm_type: "IRRIGATED",
  crop: "CABBAGE",
  year: 2024,
  month: 1,
  area_planted: 10.5,
  area_harvested: 10.0,
  productivity: 15.5
}
```

### 2. Laravel Transforms and Sends
```php
// CropPredictionService.php
Http::post('http://127.0.0.1:5000/api/predict', [
    'MUNICIPALITY' => 'ATOK',
    'FARM_TYPE' => 'IRRIGATED',
    'CROP' => 'CABBAGE',
    'YEAR' => 2024,
    'MONTH' => 1,
    'Area_planted_ha' => 10.5,
    'Area_harvested_ha' => 10.0,
    'Productivity_mt_ha' => 15.5
]);
```

### 3. Flask Processes
```python
# ml_api.py
input_data = pd.DataFrame([{
    'MUNICIPALITY': 'ATOK',
    'FARM TYPE': 'IRRIGATED',
    'CROP': 'CABBAGE',
    # ... other features
}])

# Transform and predict
prediction = model.predict(input_data)[0]
```

### 4. Returns Prediction
```json
{
  "success": true,
  "prediction": {
    "production_mt": 155.32,
    "expected_from_productivity": 155.00,
    "difference": 0.32,
    "confidence_score": 0.9888
  },
  "input": { ... },
  "timestamp": "2025-11-03T10:30:00"
}
```

### 5. Laravel Displays
```blade
<div class="alert alert-success">
  <h4>Predicted Production: 155.32 metric tons</h4>
  <p>Model Confidence: 98.88%</p>
</div>
```

---

## File Structure

```
Machine Learning/
│
├── ml_api.py                      # Flask API server
├── requirements.txt               # Python dependencies
├── start_api.ps1                  # Startup script
├── test_api.ps1                   # Test script
├── test_page.html                 # Browser test interface
│
├── model_artifacts/               # Trained model files
│   ├── best_rf_model.pkl         # Trained model
│   ├── preprocessor.pkl          # Feature transformer
│   ├── model_metadata.json       # Model info
│   ├── categorical_values.json   # Valid categories
│   └── feature_info.json         # Feature details
│
├── Documentation/
│   ├── QUICKSTART.md             # Quick start guide (this file)
│   ├── LARAVEL_INTEGRATION_GUIDE.md  # Detailed Laravel guide
│   ├── INTEGRATION_README.md     # Full documentation
│   └── laravel_api_routes_example.php  # Example routes
│
└── UPDATED ML MODEL.ipynb        # Model training notebook


Your Laravel Project/
│
├── app/
│   ├── Services/
│   │   └── CropPredictionService.php  # ML API client
│   │
│   └── Http/Controllers/
│       └── CropPredictionController.php  # Request handler
│
├── routes/
│   ├── web.php                   # Web routes
│   └── api.php                   # API routes (add ML endpoints here)
│
├── resources/views/
│   └── predictions/
│       └── index.blade.php       # Prediction form
│
├── config/
│   └── services.php              # Add ML API config
│
└── .env                          # Add ML_API_URL
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | HTML/CSS/JS or Blade | User interface |
| Backend | Laravel (PHP) | Web application |
| API | Flask (Python) | ML model serving |
| ML Model | Scikit-learn | Predictions |
| Data | Pandas/NumPy | Data processing |
| Server | Development Server | Local testing |
| Production | Gunicorn/NSSM | Production serving |

---

## Security Considerations

1. **API Authentication**: Add API keys for production
2. **Input Validation**: Both Laravel and Flask validate inputs
3. **Rate Limiting**: Prevent API abuse
4. **CORS**: Configured for your domain
5. **Error Handling**: Don't expose internal errors
6. **HTTPS**: Use HTTPS in production

---

## Performance Tips

1. **Caching**: Cache available options (municipalities, crops)
2. **Queue Jobs**: Use Laravel queues for batch predictions
3. **Connection Pooling**: Keep connections alive
4. **Response Compression**: Enable gzip
5. **Load Balancing**: Scale Flask API horizontally
6. **Monitoring**: Track API response times

---

## Common Integration Patterns

### Pattern 1: Direct API Call (Simplest)
```
User → Laravel Controller → Flask API → Response
```

### Pattern 2: Service Layer (Recommended)
```
User → Laravel Controller → Service → Flask API → Response
```

### Pattern 3: Queue + Service (For Heavy Loads)
```
User → Laravel Controller → Queue Job → Service → Flask API → Response
```

### Pattern 4: Cached Options (Optimized)
```
User → Laravel Controller → Cache → Service → Flask API → Response
                          ↑
                    First load only
```

---

## Development vs Production

### Development (Current Setup)
- Flask development server
- Running manually via `python ml_api.py`
- Logs to console
- No authentication

### Production (Recommended)
- Gunicorn WSGI server
- Runs as Windows service
- Logs to files
- API key authentication
- HTTPS enabled
- Monitored with tools

---

## Next Steps After Integration

1. **Add Authentication**: Protect your prediction endpoints
2. **Create Dashboard**: Visualize historical predictions
3. **Add Analytics**: Track model usage and accuracy
4. **Store Predictions**: Save to database for analysis
5. **Email Reports**: Send prediction summaries
6. **Add Exports**: Download predictions as CSV/PDF
7. **Mobile App**: Build mobile interface using same API

---

This architecture allows you to:
- ✅ Keep Python ML code separate from PHP
- ✅ Scale each component independently
- ✅ Update models without touching Laravel
- ✅ Reuse the API for multiple applications
- ✅ Monitor and debug each layer separately
