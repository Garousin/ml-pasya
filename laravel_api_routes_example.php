<?php

/**
 * Example Laravel API Routes for ML Predictions
 * Add these to your routes/api.php file
 */

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Validator;

/*
|--------------------------------------------------------------------------
| ML Prediction API Routes
|--------------------------------------------------------------------------
|
| These routes provide API access to the ML prediction service
| You can use these for AJAX calls or as a REST API
|
*/

// Get available options for dropdowns
Route::get('/ml/options', function () {
    try {
        $mlApiUrl = config('services.ml_api.url', 'http://127.0.0.1:5000');
        $response = Http::get("{$mlApiUrl}/api/available-options");
        
        if ($response->successful()) {
            return response()->json($response->json());
        }
        
        return response()->json([
            'error' => 'Unable to fetch options from ML service'
        ], 503);
        
    } catch (\Exception $e) {
        return response()->json([
            'error' => 'ML service is unavailable',
            'message' => $e->getMessage()
        ], 503);
    }
});

// Check ML API health
Route::get('/ml/health', function () {
    try {
        $mlApiUrl = config('services.ml_api.url', 'http://127.0.0.1:5000');
        $response = Http::timeout(5)->get("{$mlApiUrl}/api/health");
        
        if ($response->successful()) {
            return response()->json([
                'status' => 'healthy',
                'data' => $response->json()
            ]);
        }
        
        return response()->json([
            'status' => 'unhealthy',
            'error' => 'ML service returned error'
        ], 503);
        
    } catch (\Exception $e) {
        return response()->json([
            'status' => 'down',
            'error' => 'ML service is not responding',
            'message' => $e->getMessage()
        ], 503);
    }
});

// Make a single prediction
Route::post('/ml/predict', function (Request $request) {
    $validator = Validator::make($request->all(), [
        'municipality' => 'required|string',
        'farm_type' => 'required|string|in:IRRIGATED,RAINFED',
        'year' => 'required|integer|min:2020|max:2030',
        'month' => 'required|integer|min:1|max:12',
        'crop' => 'required|string',
        'area_planted' => 'required|numeric|min:0',
        'area_harvested' => 'required|numeric|min:0|lte:area_planted',
        'productivity' => 'required|numeric|min:0',
    ], [
        'area_harvested.lte' => 'Area harvested cannot be greater than area planted',
    ]);
    
    if ($validator->fails()) {
        return response()->json([
            'success' => false,
            'errors' => $validator->errors()
        ], 422);
    }
    
    try {
        $mlApiUrl = config('services.ml_api.url', 'http://127.0.0.1:5000');
        
        $response = Http::timeout(30)->post("{$mlApiUrl}/api/predict", [
            'MUNICIPALITY' => strtoupper($request->municipality),
            'FARM_TYPE' => strtoupper($request->farm_type),
            'YEAR' => (int) $request->year,
            'MONTH' => (int) $request->month,
            'CROP' => strtoupper($request->crop),
            'Area_planted_ha' => (float) $request->area_planted,
            'Area_harvested_ha' => (float) $request->area_harvested,
            'Productivity_mt_ha' => (float) $request->productivity,
        ]);
        
        if ($response->successful()) {
            $data = $response->json();
            
            // Optionally save to database here
            // DB::table('predictions')->insert([
            //     'municipality' => $data['input']['municipality'],
            //     'production_mt' => $data['prediction']['production_mt'],
            //     'created_at' => now(),
            //     ...
            // ]);
            
            return response()->json($data);
        }
        
        $error = $response->json()['error'] ?? 'Unknown error occurred';
        return response()->json([
            'success' => false,
            'error' => $error
        ], $response->status());
        
    } catch (\Exception $e) {
        return response()->json([
            'success' => false,
            'error' => 'Failed to make prediction',
            'message' => $e->getMessage()
        ], 500);
    }
});

// Batch predictions
Route::post('/ml/batch-predict', function (Request $request) {
    $validator = Validator::make($request->all(), [
        'predictions' => 'required|array|min:1|max:100',
        'predictions.*.municipality' => 'required|string',
        'predictions.*.farm_type' => 'required|string',
        'predictions.*.year' => 'required|integer|min:2020|max:2030',
        'predictions.*.month' => 'required|integer|min:1|max:12',
        'predictions.*.crop' => 'required|string',
        'predictions.*.area_planted' => 'required|numeric|min:0',
        'predictions.*.area_harvested' => 'required|numeric|min:0',
        'predictions.*.productivity' => 'required|numeric|min:0',
    ]);
    
    if ($validator->fails()) {
        return response()->json([
            'success' => false,
            'errors' => $validator->errors()
        ], 422);
    }
    
    try {
        $mlApiUrl = config('services.ml_api.url', 'http://127.0.0.1:5000');
        
        $formattedData = array_map(function($item) {
            return [
                'MUNICIPALITY' => strtoupper($item['municipality']),
                'FARM_TYPE' => strtoupper($item['farm_type']),
                'YEAR' => (int) $item['year'],
                'MONTH' => (int) $item['month'],
                'CROP' => strtoupper($item['crop']),
                'Area_planted_ha' => (float) $item['area_planted'],
                'Area_harvested_ha' => (float) $item['area_harvested'],
                'Productivity_mt_ha' => (float) $item['productivity'],
            ];
        }, $request->predictions);
        
        $response = Http::timeout(60)->post("{$mlApiUrl}/api/batch-predict", [
            'predictions' => $formattedData
        ]);
        
        if ($response->successful()) {
            return response()->json($response->json());
        }
        
        return response()->json([
            'success' => false,
            'error' => 'Batch prediction failed'
        ], $response->status());
        
    } catch (\Exception $e) {
        return response()->json([
            'success' => false,
            'error' => 'Failed to make batch prediction',
            'message' => $e->getMessage()
        ], 500);
    }
});

// Get model information
Route::get('/ml/model-info', function () {
    try {
        $mlApiUrl = config('services.ml_api.url', 'http://127.0.0.1:5000');
        $response = Http::get("{$mlApiUrl}/api/model-info");
        
        if ($response->successful()) {
            return response()->json($response->json());
        }
        
        return response()->json([
            'error' => 'Unable to fetch model info'
        ], 503);
        
    } catch (\Exception $e) {
        return response()->json([
            'error' => 'ML service is unavailable',
            'message' => $e->getMessage()
        ], 503);
    }
});
