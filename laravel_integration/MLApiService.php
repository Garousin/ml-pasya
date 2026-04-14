<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Log;

/**
 * ML API Service for Laravel
 * 
 * Provides a convenient interface to interact with the ML API
 * 
 * Usage:
 * $mlService = new MLApiService();
 * $prediction = $mlService->predict([...]);
 */
class MLApiService
{
    protected $baseUrl;
    protected $timeout;
    protected $cacheEnabled;

    public function __construct()
    {
        $this->baseUrl = config('services.ml_api.url', 'http://127.0.0.1:5000');
        $this->timeout = config('services.ml_api.timeout', 30);
        $this->cacheEnabled = config('services.ml_api.cache_enabled', true);
    }

    /**
     * Check if ML API is healthy
     */
    public function health(): array
    {
        try {
            $response = Http::timeout(5)->get("{$this->baseUrl}/api/health");
            
            if ($response->successful()) {
                return [
                    'status' => 'healthy',
                    'data' => $response->json()
                ];
            }
            
            return [
                'status' => 'unhealthy',
                'error' => 'ML service returned error'
            ];
        } catch (\Exception $e) {
            Log::error('ML API health check failed: ' . $e->getMessage());
            return [
                'status' => 'down',
                'error' => $e->getMessage()
            ];
        }
    }

    /**
     * Get available options for dropdowns
     * Results are cached for 1 hour
     */
    public function getAvailableOptions(): array
    {
        $cacheKey = 'ml_api_available_options';
        
        if ($this->cacheEnabled && Cache::has($cacheKey)) {
            return Cache::get($cacheKey);
        }
        
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/api/available-options");
            
            if ($response->successful()) {
                $data = $response->json();
                
                if ($this->cacheEnabled) {
                    Cache::put($cacheKey, $data, now()->addHour());
                }
                
                return $data;
            }
            
            throw new \Exception('Failed to get available options');
        } catch (\Exception $e) {
            Log::error('ML API getAvailableOptions failed: ' . $e->getMessage());
            throw $e;
        }
    }

    /**
     * Make a production prediction
     * 
     * @param array $data ['municipality' => 'ATOK', 'farm_type' => 'IRRIGATED', ...]
     * @return array
     */
    public function predict(array $data): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/api/predict", [
                    'MUNICIPALITY' => strtoupper($data['municipality'] ?? ''),
                    'FARM_TYPE' => strtoupper($data['farm_type'] ?? ''),
                    'YEAR' => (int) ($data['year'] ?? date('Y')),
                    'MONTH' => $data['month'] ?? 1,
                    'CROP' => strtoupper($data['crop'] ?? ''),
                    'Area_planted_ha' => (float) ($data['area_planted_ha'] ?? 0)
                ]);
            
            if ($response->successful()) {
                return $response->json();
            }
            
            throw new \Exception('Prediction failed: ' . $response->body());
        } catch (\Exception $e) {
            Log::error('ML API prediction failed: ' . $e->getMessage(), $data);
            throw $e;
        }
    }

    /**
     * Make batch predictions
     * 
     * @param array $predictions Array of prediction inputs
     * @return array
     */
    public function batchPredict(array $predictions): array
    {
        try {
            $formattedPredictions = array_map(function($item) {
                return [
                    'MUNICIPALITY' => strtoupper($item['municipality'] ?? ''),
                    'FARM_TYPE' => strtoupper($item['farm_type'] ?? ''),
                    'YEAR' => (int) ($item['year'] ?? date('Y')),
                    'MONTH' => $item['month'] ?? 1,
                    'CROP' => strtoupper($item['crop'] ?? ''),
                    'Area_planted_ha' => (float) ($item['area_planted_ha'] ?? 0)
                ];
            }, $predictions);

            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/api/batch-predict", [
                    'predictions' => $formattedPredictions
                ]);
            
            if ($response->successful()) {
                return $response->json();
            }
            
            throw new \Exception('Batch prediction failed: ' . $response->body());
        } catch (\Exception $e) {
            Log::error('ML API batch prediction failed: ' . $e->getMessage());
            throw $e;
        }
    }

    /**
     * Get forecast for a crop and municipality
     * Results are cached for 1 hour
     * 
     * @param string $crop
     * @param string $municipality
     * @return array
     */
    public function getForecast(string $crop, string $municipality): array
    {
        $cacheKey = "ml_api_forecast_{$crop}_{$municipality}";
        
        if ($this->cacheEnabled && Cache::has($cacheKey)) {
            return Cache::get($cacheKey);
        }
        
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/api/forecast", [
                    'CROP' => strtoupper($crop),
                    'MUNICIPALITY' => strtoupper($municipality)
                ]);
            
            if ($response->successful()) {
                $data = $response->json();
                
                if ($this->cacheEnabled && ($data['success'] ?? false)) {
                    Cache::put($cacheKey, $data, now()->addHour());
                }
                
                return $data;
            }
            
            throw new \Exception('Forecast request failed: ' . $response->body());
        } catch (\Exception $e) {
            Log::error('ML API forecast failed: ' . $e->getMessage(), [
                'crop' => $crop,
                'municipality' => $municipality
            ]);
            throw $e;
        }
    }

    /**
     * Get production history with filters
     * 
     * @param array $filters ['municipality' => '...', 'crop' => '...', 'year' => 2023, etc.]
     * @return array
     */
    public function getProductionHistory(array $filters = []): array
    {
        try {
            $queryParams = [];
            
            if (isset($filters['municipality'])) {
                $queryParams['municipality'] = $filters['municipality'];
            }
            if (isset($filters['crop'])) {
                $queryParams['crop'] = $filters['crop'];
            }
            if (isset($filters['year'])) {
                $queryParams['year'] = $filters['year'];
            }
            if (isset($filters['page'])) {
                $queryParams['page'] = $filters['page'];
            }
            if (isset($filters['limit'])) {
                $queryParams['limit'] = $filters['limit'];
            }

            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/api/production/history", $queryParams);
            
            if ($response->successful()) {
                return $response->json();
            }
            
            throw new \Exception('History request failed: ' . $response->body());
        } catch (\Exception $e) {
            Log::error('ML API history request failed: ' . $e->getMessage(), $filters);
            throw $e;
        }
    }

    /**
     * Get database statistics
     */
    public function getStatistics(): array
    {
        $cacheKey = 'ml_api_statistics';
        
        if ($this->cacheEnabled && Cache::has($cacheKey)) {
            return Cache::get($cacheKey);
        }
        
        try {
            $response = Http::timeout($this->timeout)
                ->get("{$this->baseUrl}/api/statistics");
            
            if ($response->successful()) {
                $data = $response->json();
                
                if ($this->cacheEnabled) {
                    Cache::put($cacheKey, $data, now()->addMinutes(30));
                }
                
                return $data;
            }
            
            throw new \Exception('Statistics request failed: ' . $response->body());
        } catch (\Exception $e) {
            Log::error('ML API statistics request failed: ' . $e->getMessage());
            throw $e;
        }
    }

    /**
     * Clear ML API cache
     */
    public function clearCache(): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/api/cache/clear");
            
            // Also clear Laravel's local cache
            if ($this->cacheEnabled) {
                Cache::forget('ml_api_available_options');
                Cache::forget('ml_api_statistics');
                // Clear all forecast caches
                Cache::flush(); // Or use tags if available
            }
            
            if ($response->successful()) {
                return $response->json();
            }
            
            throw new \Exception('Cache clear failed: ' . $response->body());
        } catch (\Exception $e) {
            Log::error('ML API cache clear failed: ' . $e->getMessage());
            throw $e;
        }
    }
}
