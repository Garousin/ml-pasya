# Laravel Integration Guide for Benguet Crop ML Model

This guide explains how to integrate the Machine Learning model with your Laravel project.

## Architecture Overview

```
Laravel Application <--HTTP--> Flask API <---> ML Model (Python)
```

The ML model runs as a separate Flask API service that your Laravel application communicates with via HTTP requests.

---

## Step 1: Set Up the Flask API

### 1.1 Install Python Dependencies

Open a terminal in the `Machine Learning` directory and run:

```bash
pip install -r requirements.txt
```

### 1.2 Start the Flask API Server

```bash
python ml_api.py
```

The API will run on `http://127.0.0.1:5000`

**Important:** Keep this terminal running while using the API.

### 1.3 Test the API

Open your browser or use Postman to test:
- Health Check: `http://127.0.0.1:5000/api/health`
- Available Options: `http://127.0.0.1:5000/api/available-options`

---

## Step 2: Laravel Integration

### 2.1 Create a Service Class

Create a new service in Laravel: `app/Services/CropPredictionService.php`

```php
<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Exception;

class CropPredictionService
{
    protected $apiUrl;
    
    public function __construct()
    {
        // API URL from config or env
        $this->apiUrl = config('services.ml_api.url', 'http://127.0.0.1:5000');
    }
    
    /**
     * Check if the ML API is healthy
     */
    public function healthCheck()
    {
        try {
            $response = Http::timeout(5)->get("{$this->apiUrl}/api/health");
            return $response->successful() ? $response->json() : null;
        } catch (Exception $e) {
            Log::error('ML API Health Check Failed: ' . $e->getMessage());
            return null;
        }
    }
    
    /**
     * Get available options for the form
     */
    public function getAvailableOptions()
    {
        try {
            $response = Http::get("{$this->apiUrl}/api/available-options");
            
            if ($response->successful()) {
                return $response->json();
            }
            
            throw new Exception('Failed to fetch available options');
        } catch (Exception $e) {
            Log::error('Failed to get available options: ' . $e->getMessage());
            throw $e;
        }
    }
    
    /**
     * Make a single prediction
     * 
     * @param array $data
     * @return array
     */
    public function predict(array $data)
    {
        try {
            $response = Http::post("{$this->apiUrl}/api/predict", [
                'MUNICIPALITY' => $data['municipality'],
                'FARM_TYPE' => $data['farm_type'],
                'YEAR' => $data['year'],
                'MONTH' => $data['month'],
                'CROP' => $data['crop'],
                'Area_planted_ha' => $data['area_planted'],
                'Area_harvested_ha' => $data['area_harvested'],
                'Productivity_mt_ha' => $data['productivity']
            ]);
            
            if ($response->successful()) {
                return $response->json();
            }
            
            $error = $response->json()['error'] ?? 'Unknown error occurred';
            throw new Exception($error);
            
        } catch (Exception $e) {
            Log::error('Prediction failed: ' . $e->getMessage());
            throw $e;
        }
    }
    
    /**
     * Make batch predictions
     * 
     * @param array $predictions
     * @return array
     */
    public function batchPredict(array $predictions)
    {
        try {
            $formattedData = array_map(function($item) {
                return [
                    'MUNICIPALITY' => $item['municipality'],
                    'FARM_TYPE' => $item['farm_type'],
                    'YEAR' => $item['year'],
                    'MONTH' => $item['month'],
                    'CROP' => $item['crop'],
                    'Area_planted_ha' => $item['area_planted'],
                    'Area_harvested_ha' => $item['area_harvested'],
                    'Productivity_mt_ha' => $item['productivity']
                ];
            }, $predictions);
            
            $response = Http::post("{$this->apiUrl}/api/batch-predict", [
                'predictions' => $formattedData
            ]);
            
            if ($response->successful()) {
                return $response->json();
            }
            
            throw new Exception('Batch prediction failed');
            
        } catch (Exception $e) {
            Log::error('Batch prediction failed: ' . $e->getMessage());
            throw $e;
        }
    }
    
    /**
     * Get model information and metadata
     */
    public function getModelInfo()
    {
        try {
            $response = Http::get("{$this->apiUrl}/api/model-info");
            
            if ($response->successful()) {
                return $response->json();
            }
            
            throw new Exception('Failed to fetch model info');
        } catch (Exception $e) {
            Log::error('Failed to get model info: ' . $e->getMessage());
            throw $e;
        }
    }
}
```

### 2.2 Add Configuration

Add to `config/services.php`:

```php
'ml_api' => [
    'url' => env('ML_API_URL', 'http://127.0.0.1:5000'),
],
```

Add to `.env`:

```
ML_API_URL=http://127.0.0.1:5000
```

### 2.3 Create a Controller

Create `app/Http/Controllers/CropPredictionController.php`:

```php
<?php

namespace App\Http\Controllers;

use App\Services\CropPredictionService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;

class CropPredictionController extends Controller
{
    protected $predictionService;
    
    public function __construct(CropPredictionService $predictionService)
    {
        $this->predictionService = $predictionService;
    }
    
    /**
     * Show the prediction form
     */
    public function index()
    {
        try {
            $options = $this->predictionService->getAvailableOptions();
            return view('predictions.index', compact('options'));
        } catch (\Exception $e) {
            return back()->with('error', 'Unable to load prediction form: ' . $e->getMessage());
        }
    }
    
    /**
     * Make a prediction
     */
    public function predict(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'municipality' => 'required|string',
            'farm_type' => 'required|string',
            'year' => 'required|integer|min:2020|max:2030',
            'month' => 'required|integer|min:1|max:12',
            'crop' => 'required|string',
            'area_planted' => 'required|numeric|min:0',
            'area_harvested' => 'required|numeric|min:0',
            'productivity' => 'required|numeric|min:0'
        ]);
        
        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'errors' => $validator->errors()
            ], 422);
        }
        
        try {
            $result = $this->predictionService->predict($request->all());
            return response()->json($result);
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }
    
    /**
     * Get available options for AJAX
     */
    public function getOptions()
    {
        try {
            $options = $this->predictionService->getAvailableOptions();
            return response()->json($options);
        } catch (\Exception $e) {
            return response()->json([
                'error' => $e->getMessage()
            ], 500);
        }
    }
}
```

### 2.4 Add Routes

Add to `routes/web.php`:

```php
use App\Http\Controllers\CropPredictionController;

Route::prefix('predictions')->group(function () {
    Route::get('/', [CropPredictionController::class, 'index'])->name('predictions.index');
    Route::post('/predict', [CropPredictionController::class, 'predict'])->name('predictions.predict');
    Route::get('/options', [CropPredictionController::class, 'getOptions'])->name('predictions.options');
});
```

### 2.5 Create a Blade View

Create `resources/views/predictions/index.blade.php`:

```blade
@extends('layouts.app')

@section('content')
<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header">
                    <h3>Crop Production Prediction</h3>
                </div>
                <div class="card-body">
                    <form id="predictionForm">
                        @csrf
                        
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label for="municipality" class="form-label">Municipality</label>
                                <select class="form-select" id="municipality" name="municipality" required>
                                    <option value="">Select Municipality</option>
                                    @foreach($options['municipalities'] as $municipality)
                                        <option value="{{ $municipality }}">{{ $municipality }}</option>
                                    @endforeach
                                </select>
                            </div>
                            
                            <div class="col-md-6 mb-3">
                                <label for="farm_type" class="form-label">Farm Type</label>
                                <select class="form-select" id="farm_type" name="farm_type" required>
                                    <option value="">Select Farm Type</option>
                                    @foreach($options['farm_types'] as $farmType)
                                        <option value="{{ $farmType }}">{{ $farmType }}</option>
                                    @endforeach
                                </select>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label for="crop" class="form-label">Crop</label>
                                <select class="form-select" id="crop" name="crop" required>
                                    <option value="">Select Crop</option>
                                    @foreach($options['crops'] as $crop)
                                        <option value="{{ $crop }}">{{ $crop }}</option>
                                    @endforeach
                                </select>
                            </div>
                            
                            <div class="col-md-3 mb-3">
                                <label for="year" class="form-label">Year</label>
                                <input type="number" class="form-control" id="year" name="year" 
                                       value="{{ date('Y') }}" min="2020" max="2030" required>
                            </div>
                            
                            <div class="col-md-3 mb-3">
                                <label for="month" class="form-label">Month</label>
                                <select class="form-select" id="month" name="month" required>
                                    <option value="">Select Month</option>
                                    @foreach($options['months'] as $month)
                                        <option value="{{ $month['value'] }}">{{ $month['label'] }}</option>
                                    @endforeach
                                </select>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-4 mb-3">
                                <label for="area_planted" class="form-label">Area Planted (ha)</label>
                                <input type="number" step="0.01" class="form-control" id="area_planted" 
                                       name="area_planted" required>
                            </div>
                            
                            <div class="col-md-4 mb-3">
                                <label for="area_harvested" class="form-label">Area Harvested (ha)</label>
                                <input type="number" step="0.01" class="form-control" id="area_harvested" 
                                       name="area_harvested" required>
                            </div>
                            
                            <div class="col-md-4 mb-3">
                                <label for="productivity" class="form-label">Productivity (mt/ha)</label>
                                <input type="number" step="0.01" class="form-control" id="productivity" 
                                       name="productivity" required>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-primary" id="predictBtn">
                            <span class="spinner-border spinner-border-sm d-none" id="spinner"></span>
                            Predict Production
                        </button>
                    </form>
                    
                    <div id="result" class="mt-4 d-none">
                        <h4>Prediction Result</h4>
                        <div class="alert alert-success">
                            <p class="mb-2"><strong>Predicted Production:</strong> <span id="prediction"></span> metric tons</p>
                            <p class="mb-2"><strong>Expected from Productivity:</strong> <span id="expected"></span> metric tons</p>
                            <p class="mb-0"><strong>Difference:</strong> <span id="difference"></span> metric tons</p>
                        </div>
                    </div>
                    
                    <div id="error" class="alert alert-danger mt-4 d-none"></div>
                </div>
            </div>
        </div>
    </div>
</div>

@push('scripts')
<script>
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const btn = document.getElementById('predictBtn');
    const spinner = document.getElementById('spinner');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    
    // Show loading state
    btn.disabled = true;
    spinner.classList.remove('d-none');
    resultDiv.classList.add('d-none');
    errorDiv.classList.add('d-none');
    
    try {
        const formData = new FormData(this);
        const data = Object.fromEntries(formData);
        
        const response = await fetch('{{ route("predictions.predict") }}', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-TOKEN': document.querySelector('input[name="_token"]').value
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('prediction').textContent = result.prediction.production_mt;
            document.getElementById('expected').textContent = result.prediction.expected_from_productivity;
            document.getElementById('difference').textContent = result.prediction.difference;
            resultDiv.classList.remove('d-none');
        } else {
            errorDiv.textContent = result.error || 'An error occurred';
            errorDiv.classList.remove('d-none');
        }
    } catch (error) {
        errorDiv.textContent = 'Network error: ' + error.message;
        errorDiv.classList.remove('d-none');
    } finally {
        btn.disabled = false;
        spinner.classList.add('d-none');
    }
});
</script>
@endpush
@endsection
```

---

## Step 3: Usage Examples

### Example 1: Simple Prediction

```php
use App\Services\CropPredictionService;

$service = new CropPredictionService();

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

// Result contains:
// - prediction.production_mt
// - prediction.expected_from_productivity
// - prediction.difference
// - prediction.confidence_score
```

### Example 2: Check API Health

```php
$health = $service->healthCheck();

if ($health) {
    echo "ML API is running";
} else {
    echo "ML API is down";
}
```

---

## Step 4: Running for Production

### Option 1: Use Gunicorn (Recommended)

Install Gunicorn:
```bash
pip install gunicorn
```

Run:
```bash
gunicorn -w 4 -b 127.0.0.1:5000 ml_api:app
```

### Option 2: Run as Windows Service

Use `nssm` (Non-Sucking Service Manager):
1. Download NSSM from https://nssm.cc/
2. Install the service:
```cmd
nssm install MLPredictionAPI "C:\path\to\python.exe" "C:\xampp\htdocs\Machine Learning\ml_api.py"
nssm start MLPredictionAPI
```

---

## Troubleshooting

### Issue: Connection Refused
- Make sure Flask API is running
- Check firewall settings
- Verify the API URL in `.env`

### Issue: CORS Error
- The Flask API includes CORS headers
- Check browser console for specific errors

### Issue: Predictions are inaccurate
- Verify input data matches training data format
- Check for data validation errors
- Review model confidence score

---

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check API health |
| `/api/model-info` | GET | Get model metadata |
| `/api/available-options` | GET | Get dropdown options |
| `/api/predict` | POST | Single prediction |
| `/api/batch-predict` | POST | Multiple predictions |

---

## Next Steps

1. **Database Integration**: Store predictions in your Laravel database
2. **Authentication**: Add authentication to your prediction routes
3. **Caching**: Cache available options to reduce API calls
4. **Queue Jobs**: Use Laravel queues for batch predictions
5. **Monitoring**: Add logging and monitoring for the ML API

---

## Additional Resources

- Flask Documentation: https://flask.palletsprojects.com/
- Laravel HTTP Client: https://laravel.com/docs/http-client
- Model Training Notebook: `UPDATED ML MODEL.ipynb`
