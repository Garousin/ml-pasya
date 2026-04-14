# 🚀 BenguetCropMap - Quick Start Implementation Guide

## 📋 TABLE OF CONTENTS
1. [Setup Environment](#setup-environment)
2. [Database Setup](#database-setup)
3. [Laravel API](#laravel-api)
4. [Frontend Map](#frontend-map)
5. [Python ML API](#python-ml-api)
6. [Testing](#testing)

---

## 1️⃣ SETUP ENVIRONMENT

### Prerequisites Check
```powershell
# Check if you have everything
php --version        # Should be 7.4+
composer --version   # Composer installed
node --version       # Node.js 14+
python --version     # Python 3.8+
mysql --version      # MySQL running
```

### Install Laravel Packages
```powershell
cd "c:\xampp\htdocs"

# Create new Laravel project (or use existing)
composer create-project laravel/laravel benguet-crop-map

cd benguet-crop-map

# Install required packages
composer require maatwebsite/excel
composer require barryvdh/dompdf
composer require guzzlehttp/guzzle
```

### Install Frontend Libraries
```powershell
npm install leaflet chart.js axios
```

---

## 2️⃣ DATABASE SETUP

### Step 1: Create Migration
```powershell
php artisan make:migration create_crop_production_table
```

**File: `database/migrations/xxxx_create_crop_production_table.php`**
```php
<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateCropProductionTable extends Migration
{
    public function up()
    {
        Schema::create('crop_production', function (Blueprint $table) {
            $table->id();
            $table->string('municipality', 50);
            $table->enum('farm_type', ['IRRIGATED', 'RAINFED']);
            $table->integer('year');
            $table->string('month', 3);
            $table->string('crop', 50);
            $table->decimal('area_planted', 10, 2);
            $table->decimal('area_harvested', 10, 2);
            $table->decimal('production', 10, 2);
            $table->decimal('productivity', 10, 2)->nullable();
            $table->timestamps();
            
            // Indexes for fast queries
            $table->index('municipality');
            $table->index('year');
            $table->index('crop');
            $table->index(['municipality', 'year', 'crop']);
        });
    }

    public function down()
    {
        Schema::dropIfExists('crop_production');
    }
}
```

```powershell
# Run migration
php artisan migrate
```

### Step 2: Create Model
```powershell
php artisan make:model CropProduction
```

**File: `app/Models/CropProduction.php`**
```php
<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class CropProduction extends Model
{
    protected $table = 'crop_production';
    
    protected $fillable = [
        'municipality', 'farm_type', 'year', 'month', 'crop',
        'area_planted', 'area_harvested', 'production', 'productivity'
    ];
    
    protected $casts = [
        'year' => 'integer',
        'area_planted' => 'float',
        'area_harvested' => 'float',
        'production' => 'float',
        'productivity' => 'float',
    ];
    
    // Scopes for easy filtering
    public function scopeMunicipality($query, $municipality)
    {
        return $query->where('municipality', $municipality);
    }
    
    public function scopeCrop($query, $crop)
    {
        return $query->where('crop', $crop);
    }
    
    public function scopeYear($query, $year)
    {
        return $query->where('year', $year);
    }
    
    public function scopeFarmType($query, $farmType)
    {
        return $query->where('farm_type', $farmType);
    }
}
```

### Step 3: Import CSV Data
```powershell
php artisan make:import CropProductionImport --model=CropProduction
```

**File: `app/Imports/CropProductionImport.php`**
```php
<?php

namespace App\Imports;

use App\Models\CropProduction;
use Maatwebsite\Excel\Concerns\ToModel;
use Maatwebsite\Excel\Concerns\WithHeadingRow;

class CropProductionImport implements ToModel, WithHeadingRow
{
    public function model(array $row)
    {
        return new CropProduction([
            'municipality' => $row['municipality'],
            'farm_type' => $row['farm_type'],
            'year' => (int) $row['year'],
            'month' => strtoupper(substr($row['month'], 0, 3)),
            'crop' => $row['crop'],
            'area_planted' => (float) $row['area_plantedha'] ?? 0,
            'area_harvested' => (float) $row['area_harvestedha'] ?? 0,
            'production' => (float) $row['productionmt'] ?? 0,
            'productivity' => (float) $row['productivitymtha'] ?? null,
        ]);
    }
}
```

**Create Command to Import**
```powershell
php artisan make:command ImportCropData
```

**File: `app/Console/Commands/ImportCropData.php`**
```php
<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Imports\CropProductionImport;
use Maatwebsite\Excel\Facades\Excel;

class ImportCropData extends Command
{
    protected $signature = 'import:crop-data {file}';
    protected $description = 'Import crop production data from CSV';

    public function handle()
    {
        $file = $this->argument('file');
        
        if (!file_exists($file)) {
            $this->error("File not found: {$file}");
            return 1;
        }
        
        $this->info('Importing data...');
        
        Excel::import(new CropProductionImport, $file);
        
        $this->info('Import completed successfully!');
        return 0;
    }
}
```

```powershell
# Import your data
php artisan import:crop-data "c:\xampp\htdocs\ML UPDATED\fulldataset.csv"
```

---

## 3️⃣ LARAVEL API

### Step 1: Create API Controller
```powershell
php artisan make:controller Api/MapDataController
```

**File: `app/Http/Controllers/Api/MapDataController.php`**
```php
<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\CropProduction;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class MapDataController extends Controller
{
    /**
     * Get aggregated production data for map
     * GET /api/map/data?crop=CABBAGE&year=2024&view=production
     */
    public function getMapData(Request $request)
    {
        $crop = $request->input('crop', 'CABBAGE');
        $year = $request->input('year', 2024);
        $view = $request->input('view', 'production'); // production, productivity, area
        $farmType = $request->input('farm_type'); // optional
        
        $query = CropProduction::query()
            ->where('crop', $crop)
            ->where('year', $year);
        
        if ($farmType) {
            $query->where('farm_type', $farmType);
        }
        
        // Determine aggregation based on view type
        $selectField = match($view) {
            'productivity' => 'AVG(productivity) as value',
            'area' => 'SUM(area_planted) as value',
            default => 'SUM(production) as value'
        };
        
        $data = $query
            ->select('municipality', DB::raw($selectField))
            ->groupBy('municipality')
            ->get()
            ->map(function ($item) {
                return [
                    'municipality' => $item->municipality,
                    'value' => round($item->value, 2)
                ];
            });
        
        return response()->json([
            'success' => true,
            'data' => $data,
            'filters' => [
                'crop' => $crop,
                'year' => $year,
                'view' => $view,
                'farm_type' => $farmType
            ]
        ]);
    }
    
    /**
     * Get municipality details
     * GET /api/map/municipality/{name}
     */
    public function getMunicipalityDetails(Request $request, $municipality)
    {
        $crop = $request->input('crop', 'CABBAGE');
        $year = $request->input('year', 2024);
        
        // Monthly production
        $monthlyData = CropProduction::where('municipality', $municipality)
            ->where('crop', $crop)
            ->where('year', $year)
            ->select('month', DB::raw('SUM(production) as total_production'))
            ->groupBy('month')
            ->orderByRaw("FIELD(month, 'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC')")
            ->get();
        
        // Crop distribution
        $cropDistribution = CropProduction::where('municipality', $municipality)
            ->where('year', $year)
            ->select('crop', DB::raw('SUM(production) as total'))
            ->groupBy('crop')
            ->orderByDesc('total')
            ->get();
        
        // Farm type breakdown
        $farmTypeBreakdown = CropProduction::where('municipality', $municipality)
            ->where('crop', $crop)
            ->where('year', $year)
            ->select('farm_type', DB::raw('SUM(production) as total'), DB::raw('AVG(productivity) as avg_productivity'))
            ->groupBy('farm_type')
            ->get();
        
        // Summary stats
        $summary = CropProduction::where('municipality', $municipality)
            ->where('crop', $crop)
            ->where('year', $year)
            ->select(
                DB::raw('SUM(production) as total_production'),
                DB::raw('AVG(productivity) as avg_productivity'),
                DB::raw('SUM(area_planted) as total_area_planted'),
                DB::raw('SUM(area_harvested) as total_area_harvested')
            )
            ->first();
        
        return response()->json([
            'success' => true,
            'municipality' => $municipality,
            'summary' => $summary,
            'monthly_data' => $monthlyData,
            'crop_distribution' => $cropDistribution,
            'farm_type_breakdown' => $farmTypeBreakdown
        ]);
    }
    
    /**
     * Get filter options
     * GET /api/map/filters
     */
    public function getFilterOptions()
    {
        $municipalities = CropProduction::distinct()->pluck('municipality')->sort()->values();
        $crops = CropProduction::distinct()->pluck('crop')->sort()->values();
        $years = CropProduction::distinct()->pluck('year')->sort()->values();
        $farmTypes = ['IRRIGATED', 'RAINFED'];
        
        return response()->json([
            'success' => true,
            'municipalities' => $municipalities,
            'crops' => $crops,
            'years' => $years,
            'farm_types' => $farmTypes
        ]);
    }
    
    /**
     * Get timeline data for animation
     * GET /api/map/timeline?crop=CABBAGE&year=2024
     */
    public function getTimelineData(Request $request)
    {
        $crop = $request->input('crop', 'CABBAGE');
        $year = $request->input('year', 2024);
        
        $months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
        $timelineData = [];
        
        foreach ($months as $month) {
            $data = CropProduction::where('crop', $crop)
                ->where('year', $year)
                ->where('month', $month)
                ->select('municipality', DB::raw('SUM(production) as value'))
                ->groupBy('municipality')
                ->get()
                ->pluck('value', 'municipality');
            
            $timelineData[$month] = $data;
        }
        
        return response()->json([
            'success' => true,
            'timeline' => $timelineData,
            'crop' => $crop,
            'year' => $year
        ]);
    }
    
    /**
     * Export data
     * POST /api/map/export
     */
    public function exportData(Request $request)
    {
        $format = $request->input('format', 'csv'); // csv or pdf
        $filters = $request->only(['crop', 'year', 'municipality', 'farm_type']);
        
        $query = CropProduction::query();
        
        foreach ($filters as $key => $value) {
            if ($value) {
                $query->where($key, $value);
            }
        }
        
        $data = $query->get();
        
        if ($format === 'csv') {
            $filename = 'crop_production_export_' . now()->format('Y-m-d_His') . '.csv';
            $headers = [
                'Content-Type' => 'text/csv',
                'Content-Disposition' => "attachment; filename=\"{$filename}\"",
            ];
            
            $callback = function() use ($data) {
                $file = fopen('php://output', 'w');
                fputcsv($file, ['Municipality', 'Farm Type', 'Year', 'Month', 'Crop', 'Area Planted', 'Area Harvested', 'Production', 'Productivity']);
                
                foreach ($data as $row) {
                    fputcsv($file, [
                        $row->municipality,
                        $row->farm_type,
                        $row->year,
                        $row->month,
                        $row->crop,
                        $row->area_planted,
                        $row->area_harvested,
                        $row->production,
                        $row->productivity,
                    ]);
                }
                fclose($file);
            };
            
            return response()->stream($callback, 200, $headers);
        }
        
        // PDF export can be implemented with DomPDF
    }
}
```

### Step 2: Add API Routes

**File: `routes/api.php`**
```php
<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\MapDataController;

Route::prefix('map')->group(function () {
    Route::get('/data', [MapDataController::class, 'getMapData']);
    Route::get('/municipality/{municipality}', [MapDataController::class, 'getMunicipalityDetails']);
    Route::get('/filters', [MapDataController::class, 'getFilterOptions']);
    Route::get('/timeline', [MapDataController::class, 'getTimelineData']);
    Route::post('/export', [MapDataController::class, 'exportData']);
});
```

### Step 3: Enable CORS

**File: `config/cors.php`**
```php
return [
    'paths' => ['api/*'],
    'allowed_methods' => ['*'],
    'allowed_origins' => ['*'],
    'allowed_origins_patterns' => [],
    'allowed_headers' => ['*'],
    'exposed_headers' => [],
    'max_age' => 0,
    'supports_credentials' => false,
];
```

---

## 4️⃣ FRONTEND MAP

### Step 1: Create Map HTML
**File: `resources/views/map.blade.php`**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benguet Crop Map</title>
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        
        .container {
            display: grid;
            grid-template-columns: 300px 1fr;
            height: 100vh;
        }
        
        .sidebar {
            background: #2c3e50;
            color: white;
            padding: 20px;
            overflow-y: auto;
        }
        
        .map-container {
            position: relative;
        }
        
        #map {
            width: 100%;
            height: calc(100vh - 100px);
        }
        
        .timeline-control {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: white;
            padding: 15px 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            display: none; /* Show when needed */
        }
        
        .filter-group {
            margin-bottom: 20px;
        }
        
        .filter-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
        }
        
        .filter-group select,
        .filter-group input {
            width: 100%;
            padding: 8px;
            border-radius: 5px;
            border: none;
        }
        
        .btn {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 600;
            margin-top: 10px;
        }
        
        .btn-primary { background: #3498db; color: white; }
        .btn-secondary { background: #95a5a6; color: white; }
        
        .legend {
            position: absolute;
            bottom: 120px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
        }
        
        .legend-title {
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .legend-scale {
            display: flex;
            height: 20px;
            border-radius: 3px;
            overflow: hidden;
        }
        
        .legend-labels {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            margin-top: 5px;
        }
        
        .detail-panel {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 400px;
            max-height: calc(100vh - 200px);
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.3);
            padding: 20px;
            z-index: 1001;
            display: none;
            overflow-y: auto;
        }
        
        .detail-panel.active {
            display: block;
        }
        
        .close-btn {
            float: right;
            cursor: pointer;
            font-size: 24px;
            line-height: 1;
        }
        
        h1 { font-size: 24px; margin-bottom: 10px; }
        h2 { font-size: 18px; margin: 15px 0 10px; color: #2c3e50; }
        h3 { font-size: 16px; margin: 10px 0 5px; }
        
        .stat-card {
            background: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .stat-label { font-size: 12px; color: #7f8c8d; }
        .stat-value { font-size: 20px; font-weight: 600; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Sidebar with filters -->
        <div class="sidebar">
            <h1>🗺️ Benguet Crop Map</h1>
            <p style="font-size: 14px; margin-bottom: 20px; color: #bdc3c7;">
                Interactive crop production mapping system
            </p>
            
            <div class="filter-group">
                <label>Crop</label>
                <select id="cropFilter">
                    <option value="CABBAGE">Cabbage</option>
                    <option value="BROCCOLI">Broccoli</option>
                    <option value="LETTUCE">Lettuce</option>
                    <option value="CAULIFLOWER">Cauliflower</option>
                    <option value="CHINESE CABBAGE">Chinese Cabbage</option>
                    <option value="CARROTS">Carrots</option>
                    <option value="GARDEN PEAS">Garden Peas</option>
                    <option value="WHITE POTATO">White Potato</option>
                    <option value="SNAP BEANS">Snap Beans</option>
                    <option value="SWEET PEPPER">Sweet Pepper</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Year</label>
                <select id="yearFilter">
                    <option value="2024">2024</option>
                    <option value="2023">2023</option>
                    <option value="2022">2022</option>
                    <option value="2021">2021</option>
                    <option value="2020">2020</option>
                    <option value="2019">2019</option>
                    <option value="2018">2018</option>
                    <option value="2017">2017</option>
                    <option value="2016">2016</option>
                    <option value="2015">2015</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>View Type</label>
                <select id="viewFilter">
                    <option value="production">Production (mt)</option>
                    <option value="productivity">Productivity (mt/ha)</option>
                    <option value="area">Area Planted (ha)</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Farm Type</label>
                <select id="farmTypeFilter">
                    <option value="">All</option>
                    <option value="IRRIGATED">Irrigated</option>
                    <option value="RAINFED">Rainfed</option>
                </select>
            </div>
            
            <button class="btn btn-primary" onclick="updateMap()">Apply Filters</button>
            <button class="btn btn-secondary" onclick="resetFilters()">Reset</button>
            
            <hr style="margin: 20px 0; border-color: #34495e;">
            
            <button class="btn btn-primary" onclick="toggleTimeline()">
                🎬 Toggle Timeline
            </button>
            
            <button class="btn btn-secondary" onclick="exportData()">
                📥 Export Data
            </button>
        </div>
        
        <!-- Map container -->
        <div class="map-container">
            <div id="map"></div>
            
            <!-- Legend -->
            <div class="legend">
                <div class="legend-title">Production (mt)</div>
                <div class="legend-scale" id="legendScale"></div>
                <div class="legend-labels">
                    <span id="legendMin">0</span>
                    <span id="legendMax">1000</span>
                </div>
            </div>
            
            <!-- Timeline control -->
            <div class="timeline-control" id="timelineControl">
                <button onclick="playTimeline()">▶️ Play</button>
                <button onclick="pauseTimeline()">⏸️ Pause</button>
                <input type="range" id="timelineSlider" min="0" max="11" value="0" style="width: 300px;">
                <span id="currentPeriod">JAN 2024</span>
            </div>
            
            <!-- Detail panel -->
            <div class="detail-panel" id="detailPanel">
                <span class="close-btn" onclick="closeDetailPanel()">&times;</span>
                <h2 id="panelTitle">Municipality Details</h2>
                
                <div class="stat-card">
                    <div class="stat-label">Total Production</div>
                    <div class="stat-value" id="totalProduction">-</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-label">Avg Productivity</div>
                    <div class="stat-value" id="avgProductivity">-</div>
                </div>
                
                <h3>Monthly Production</h3>
                <canvas id="monthlyChart"></canvas>
                
                <h3>Crop Distribution</h3>
                <canvas id="cropChart"></canvas>
            </div>
        </div>
    </div>
    
    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <!-- Axios for API calls -->
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    
    <script>
        // Initialize map centered on Benguet
        const map = L.map('map').setView([16.4023, 120.5960], 10);
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        let currentLayer = null;
        let timelineInterval = null;
        
        // API base URL
        const API_BASE = '/api/map';
        
        // Function to get color based on value
        function getColor(value, min, max) {
            const ratio = (value - min) / (max - min);
            const hue = ratio * 120; // 0 (red) to 120 (green)
            return `hsl(${hue}, 70%, 50%)`;
        }
        
        // Update map with data
        async function updateMap() {
            const crop = document.getElementById('cropFilter').value;
            const year = document.getElementById('yearFilter').value;
            const view = document.getElementById('viewFilter').value;
            const farmType = document.getElementById('farmTypeFilter').value;
            
            try {
                const response = await axios.get(`${API_BASE}/data`, {
                    params: { crop, year, view, farm_type: farmType }
                });
                
                const data = response.data.data;
                
                // Calculate min/max for color scaling
                const values = data.map(d => d.value);
                const min = Math.min(...values);
                const max = Math.max(...values);
                
                // Update legend
                document.getElementById('legendMin').textContent = min.toFixed(2);
                document.getElementById('legendMax').textContent = max.toFixed(2);
                
                // Remove existing layer
                if (currentLayer) {
                    map.removeLayer(currentLayer);
                }
                
                // Add markers (replace with GeoJSON polygons when you have the data)
                currentLayer = L.layerGroup();
                
                data.forEach(item => {
                    // Temporary: use markers until GeoJSON is available
                    const coords = getMunicipalityCoords(item.municipality);
                    if (coords) {
                        const marker = L.circleMarker(coords, {
                            radius: 15,
                            fillColor: getColor(item.value, min, max),
                            color: '#fff',
                            weight: 2,
                            opacity: 1,
                            fillOpacity: 0.8
                        });
                        
                        marker.bindPopup(`
                            <b>${item.municipality}</b><br>
                            ${view}: ${item.value.toFixed(2)}
                        `);
                        
                        marker.on('click', () => {
                            loadMunicipalityDetails(item.municipality);
                        });
                        
                        currentLayer.addLayer(marker);
                    }
                });
                
                currentLayer.addTo(map);
                
            } catch (error) {
                console.error('Error loading map data:', error);
                alert('Error loading data. Please try again.');
            }
        }
        
        // Temporary function to get municipality coordinates
        // Replace with actual GeoJSON centroids
        function getMunicipalityCoords(municipality) {
            const coords = {
                'ATOK': [16.5869, 120.6993],
                'BAKUN': [16.7833, 120.6667],
                'BOKOD': [16.4667, 120.8333],
                'BUGUIAS': [16.7333, 120.8167],
                'ITOGON': [16.3667, 120.6833],
                'KABAYAN': [16.6167, 120.8500],
                'KAPANGAN': [16.5833, 120.5833],
                'KIBUNGAN': [16.7000, 120.6333],
                'LATRINIDAD': [16.4564, 120.5928],
                'MANKAYAN': [16.8667, 120.8000],
                'SABLAN': [16.4833, 120.5333],
                'TUBA': [16.3167, 120.5667],
                'TUBLAY': [16.5167, 120.6167]
            };
            return coords[municipality];
        }
        
        // Load municipality details
        async function loadMunicipalityDetails(municipality) {
            const crop = document.getElementById('cropFilter').value;
            const year = document.getElementById('yearFilter').value;
            
            try {
                const response = await axios.get(`${API_BASE}/municipality/${municipality}`, {
                    params: { crop, year }
                });
                
                const data = response.data;
                
                // Update panel
                document.getElementById('panelTitle').textContent = `${municipality} - ${crop}`;
                document.getElementById('totalProduction').textContent = 
                    `${data.summary.total_production.toFixed(2)} mt`;
                document.getElementById('avgProductivity').textContent = 
                    `${data.summary.avg_productivity.toFixed(2)} mt/ha`;
                
                // Show panel
                document.getElementById('detailPanel').classList.add('active');
                
                // Create monthly chart
                createMonthlyChart(data.monthly_data);
                createCropChart(data.crop_distribution);
                
            } catch (error) {
                console.error('Error loading municipality details:', error);
            }
        }
        
        // Create monthly production chart
        function createMonthlyChart(data) {
            const ctx = document.getElementById('monthlyChart');
            if (window.monthlyChart) {
                window.monthlyChart.destroy();
            }
            
            window.monthlyChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.map(d => d.month),
                    datasets: [{
                        label: 'Production (mt)',
                        data: data.map(d => d.total_production),
                        backgroundColor: 'rgba(52, 152, 219, 0.6)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
        }
        
        // Create crop distribution chart
        function createCropChart(data) {
            const ctx = document.getElementById('cropChart');
            if (window.cropChart) {
                window.cropChart.destroy();
            }
            
            window.cropChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: data.map(d => d.crop),
                    datasets: [{
                        data: data.map(d => d.total),
                        backgroundColor: [
                            '#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
                            '#9b59b6', '#1abc9c', '#34495e', '#e67e22'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true
                }
            });
        }
        
        // Close detail panel
        function closeDetailPanel() {
            document.getElementById('detailPanel').classList.remove('active');
        }
        
        // Reset filters
        function resetFilters() {
            document.getElementById('cropFilter').value = 'CABBAGE';
            document.getElementById('yearFilter').value = '2024';
            document.getElementById('viewFilter').value = 'production';
            document.getElementById('farmTypeFilter').value = '';
            updateMap();
        }
        
        // Toggle timeline
        function toggleTimeline() {
            const timeline = document.getElementById('timelineControl');
            timeline.style.display = timeline.style.display === 'none' ? 'block' : 'none';
        }
        
        // Export data
        async function exportData() {
            const crop = document.getElementById('cropFilter').value;
            const year = document.getElementById('yearFilter').value;
            
            window.location.href = `${API_BASE}/export?crop=${crop}&year=${year}&format=csv`;
        }
        
        // Initialize map on load
        updateMap();
    </script>
</body>
</html>
```

### Step 2: Add Route

**File: `routes/web.php`**
```php
Route::get('/map', function () {
    return view('map');
});
```

---

## 5️⃣ PYTHON ML API

### Create Python API

**File: `predict_api.py` (in your ML folder)**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from typing import Optional

app = FastAPI(title="Benguet Crop Prediction API")

# Load model artifacts
MODEL_PATH = "model_artifacts/best_rf_model.pkl"
PREPROCESSOR_PATH = "model_artifacts/preprocessor.pkl"
CATEGORICAL_VALUES_PATH = "model_artifacts/categorical_values.json"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    with open(CATEGORICAL_VALUES_PATH, 'r') as f:
        categorical_values = json.load(f)
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class PredictionInput(BaseModel):
    MUNICIPALITY: str
    FARM_TYPE: str
    YEAR: int
    MONTH: str
    CROP: str
    area_planted: float
    area_harvested: float
    productivity: Optional[float] = None

class PredictionOutput(BaseModel):
    predicted_production: float
    confidence: float
    input_data: dict

@app.get("/")
def root():
    return {
        "message": "Benguet Crop Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Validate categorical values
        if input_data.MUNICIPALITY not in categorical_values["MUNICIPALITY"]:
            raise HTTPException(400, f"Invalid municipality: {input_data.MUNICIPALITY}")
        if input_data.CROP not in categorical_values["CROP"]:
            raise HTTPException(400, f"Invalid crop: {input_data.CROP}")
        if input_data.FARM_TYPE not in categorical_values["FARM TYPE"]:
            raise HTTPException(400, f"Invalid farm type: {input_data.FARM_TYPE}")
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data.dict()])
        
        # Rename columns to match training data
        df = df.rename(columns={
            'area_planted': 'Area planted(ha)',
            'area_harvested': 'Area harvested(ha)',
            'productivity': 'Productivity(mt/ha)'
        })
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Calculate confidence (simplified - use model's feature importance or prediction variance)
        confidence = 0.95  # Placeholder
        
        return PredictionOutput(
            predicted_production=float(prediction),
            confidence=confidence,
            input_data=input_data.dict()
        )
        
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.get("/valid-values")
def get_valid_values():
    """Get valid values for categorical features"""
    return categorical_values

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

### Run Python API

```powershell
cd "c:\xampp\htdocs\ML UPDATED"
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn joblib pandas scikit-learn
python predict_api.py
```

API will run on: `http://localhost:8000`

---

## 6️⃣ TESTING

### Test Laravel API
```powershell
# Start Laravel
php artisan serve

# Test endpoints
curl http://localhost:8000/api/map/data?crop=CABBAGE&year=2024
curl http://localhost:8000/api/map/filters
curl http://localhost:8000/api/map/municipality/LATRINIDAD?crop=CABBAGE&year=2024
```

### Test Python API
```powershell
# Test prediction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{\"MUNICIPALITY\":\"LATRINIDAD\",\"FARM_TYPE\":\"IRRIGATED\",\"YEAR\":2025,\"MONTH\":\"JAN\",\"CROP\":\"CABBAGE\",\"area_planted\":100,\"area_harvested\":95,\"productivity\":20}'
```

### Access Map
Open browser: `http://localhost:8000/map`

---

## 🎉 YOU'RE DONE!

Your basic map is now running! Next steps:
1. ✅ Download GeoJSON boundaries for Benguet
2. ✅ Replace markers with actual polygons
3. ✅ Add more features from the feature list
4. ✅ Style and polish the UI

**Happy Mapping!** 🗺️
