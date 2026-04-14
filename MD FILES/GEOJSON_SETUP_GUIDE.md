# 📍 Benguet Municipality GeoJSON Setup Guide

## WHERE TO GET GEOJSON DATA

### Option 1: GADM (Recommended) ⭐
**Best for: Accurate administrative boundaries**

1. Visit: https://gadm.org/download_country.html
2. Select: Philippines
3. Download: Shapefile or GeoJSON
4. Level: ADM2 (Municipality level)
5. Filter for Benguet province

**Direct Link:**
```
https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_PHL_2.json
```

### Option 2: PhilGIS
**Best for: Philippine-specific data**

1. Visit: https://philgis.org/
2. Navigate to: Open Data Portal
3. Search: "Benguet boundaries"
4. Download GeoJSON format

### Option 3: OpenStreetMap
**Best for: Customizable data**

1. Visit: https://www.openstreetmap.org/
2. Navigate to Benguet, Philippines
3. Use Export feature
4. Select "GeoJSON" format
5. Use Overpass API for specific queries

**Overpass API Query:**
```
[out:json];
area["name"="Benguet"]["admin_level"="6"];
rel(area)["admin_level"="7"];
out geom;
```

Use at: https://overpass-turbo.eu/

---

## SAMPLE GEOJSON STRUCTURE

Once you download, your GeoJSON should look like this:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "NAME_2": "ATOK",
        "NAME_1": "Benguet",
        "TYPE_2": "Municipality"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [120.6993, 16.5869],
            [120.7100, 16.5900],
            [120.7000, 16.6000],
            [120.6993, 16.5869]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "NAME_2": "LA TRINIDAD",
        "NAME_1": "Benguet",
        "TYPE_2": "Municipality"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [120.5928, 16.4564],
            [120.6000, 16.4600],
            [120.5900, 16.4700],
            [120.5928, 16.4564]
          ]
        ]
      }
    }
    // ... more municipalities
  ]
}
```

---

## HOW TO USE GEOJSON IN YOUR MAP

### Method 1: Load from File

**Step 1:** Save GeoJSON as `public/data/benguet.geojson`

**Step 2:** Update map JavaScript:

```javascript
// Load GeoJSON
fetch('/data/benguet.geojson')
    .then(response => response.json())
    .then(geoData => {
        // Add to map
        L.geoJSON(geoData, {
            style: function(feature) {
                return {
                    fillColor: '#3498db',
                    weight: 2,
                    opacity: 1,
                    color: 'white',
                    fillOpacity: 0.7
                };
            },
            onEachFeature: function(feature, layer) {
                const municipality = feature.properties.NAME_2;
                
                layer.bindPopup(`
                    <b>${municipality}</b><br>
                    Click for details
                `);
                
                layer.on('click', function() {
                    loadMunicipalityDetails(municipality);
                });
            }
        }).addTo(map);
    });
```

### Method 2: Load from Database (Advanced)

**Step 1:** Store GeoJSON in database
```sql
CREATE TABLE municipality_boundaries (
    id INT PRIMARY KEY,
    municipality VARCHAR(50) UNIQUE,
    geojson TEXT,
    centroid POINT
);
```

**Step 2:** Create API endpoint
```php
Route::get('/api/map/geojson', function() {
    $boundaries = DB::table('municipality_boundaries')->get();
    
    $features = $boundaries->map(function($boundary) {
        return [
            'type' => 'Feature',
            'properties' => [
                'municipality' => $boundary->municipality
            ],
            'geometry' => json_decode($boundary->geojson)
        ];
    });
    
    return response()->json([
        'type' => 'FeatureCollection',
        'features' => $features
    ]);
});
```

---

## INTEGRATE WITH PRODUCTION DATA

### Enhanced JavaScript with Data Coloring

```javascript
async function updateMapWithGeoJSON() {
    const crop = document.getElementById('cropFilter').value;
    const year = document.getElementById('yearFilter').value;
    const view = document.getElementById('viewFilter').value;
    
    try {
        // 1. Load GeoJSON boundaries
        const geoResponse = await fetch('/data/benguet.geojson');
        const geoData = await geoResponse.json();
        
        // 2. Load production data
        const dataResponse = await axios.get('/api/map/data', {
            params: { crop, year, view }
        });
        const productionData = dataResponse.data.data;
        
        // 3. Create lookup for easy access
        const dataLookup = {};
        productionData.forEach(item => {
            dataLookup[item.municipality] = item.value;
        });
        
        // 4. Calculate min/max for color scaling
        const values = Object.values(dataLookup);
        const min = Math.min(...values);
        const max = Math.max(...values);
        
        // 5. Remove existing layer
        if (currentLayer) {
            map.removeLayer(currentLayer);
        }
        
        // 6. Add GeoJSON layer with colors
        currentLayer = L.geoJSON(geoData, {
            style: function(feature) {
                const municipality = feature.properties.NAME_2 || feature.properties.name;
                const value = dataLookup[municipality] || 0;
                
                return {
                    fillColor: getColor(value, min, max),
                    weight: 2,
                    opacity: 1,
                    color: 'white',
                    fillOpacity: 0.7
                };
            },
            onEachFeature: function(feature, layer) {
                const municipality = feature.properties.NAME_2 || feature.properties.name;
                const value = dataLookup[municipality] || 0;
                
                // Tooltip
                layer.bindTooltip(`
                    <b>${municipality}</b><br>
                    ${view}: ${value.toFixed(2)}
                `, {
                    permanent: false,
                    direction: 'top'
                });
                
                // Click event
                layer.on('click', function() {
                    loadMunicipalityDetails(municipality);
                });
                
                // Hover effect
                layer.on('mouseover', function() {
                    layer.setStyle({
                        weight: 4,
                        fillOpacity: 0.9
                    });
                });
                
                layer.on('mouseout', function() {
                    layer.setStyle({
                        weight: 2,
                        fillOpacity: 0.7
                    });
                });
            }
        }).addTo(map);
        
        // 7. Fit map to boundaries
        map.fitBounds(currentLayer.getBounds());
        
        // 8. Update legend
        updateLegend(min, max, view);
        
    } catch (error) {
        console.error('Error loading map:', error);
        alert('Error loading map data');
    }
}

// Helper function for color gradient
function getColor(value, min, max) {
    if (value === 0) return '#cccccc';
    
    const ratio = (value - min) / (max - min);
    
    // Green to Yellow to Red gradient
    if (ratio < 0.5) {
        // Green to Yellow
        const r = Math.floor(255 * (ratio * 2));
        return `rgb(${r}, 255, 0)`;
    } else {
        // Yellow to Red
        const g = Math.floor(255 * (2 - ratio * 2));
        return `rgb(255, ${g}, 0)`;
    }
}

// Update legend with gradient
function updateLegend(min, max, view) {
    const legendScale = document.getElementById('legendScale');
    const legendMin = document.getElementById('legendMin');
    const legendMax = document.getElementById('legendMax');
    
    // Clear existing
    legendScale.innerHTML = '';
    
    // Create gradient
    const steps = 10;
    for (let i = 0; i < steps; i++) {
        const value = min + (max - min) * (i / steps);
        const color = getColor(value, min, max);
        const div = document.createElement('div');
        div.style.flex = '1';
        div.style.backgroundColor = color;
        legendScale.appendChild(div);
    }
    
    // Update labels
    legendMin.textContent = min.toFixed(2);
    legendMax.textContent = max.toFixed(2);
    
    // Update title based on view
    const viewLabels = {
        'production': 'Production (mt)',
        'productivity': 'Productivity (mt/ha)',
        'area': 'Area Planted (ha)'
    };
    document.querySelector('.legend-title').textContent = viewLabels[view];
}
```

---

## SIMPLIFY GEOJSON (IMPORTANT!)

Large GeoJSON files can slow down your map. Use **mapshaper** to simplify:

### Online Tool
1. Visit: https://mapshaper.org/
2. Upload your GeoJSON
3. Click "Simplify"
4. Adjust slider to 10-20% (balance between detail and file size)
5. Export as GeoJSON

### Command Line (Node.js)
```bash
npm install -g mapshaper

mapshaper benguet.geojson \
  -simplify 20% \
  -o benguet_simplified.geojson
```

**Result:** 50-80% smaller file size with minimal visual difference!

---

## MUNICIPALITY NAME MAPPING

Your CSV uses different naming than GeoJSON might. Create a mapping:

```javascript
const municipalityMapping = {
    'LATRINIDAD': 'La Trinidad',
    'LA TRINIDAD': 'La Trinidad',
    'ATOK': 'Atok',
    'BAKUN': 'Bakun',
    'BOKOD': 'Bokod',
    'BUGUIAS': 'Buguias',
    'ITOGON': 'Itogon',
    'KABAYAN': 'Kabayan',
    'KAPANGAN': 'Kapangan',
    'KIBUNGAN': 'Kibungan',
    'MANKAYAN': 'Mankayan',
    'SABLAN': 'Sablan',
    'TUBA': 'Tuba',
    'TUBLAY': 'Tublay'
};

function normalizeMunicipalityName(name) {
    const upper = name.toUpperCase().trim();
    return municipalityMapping[upper] || name;
}
```

---

## COMPLETE WORKING EXAMPLE

**File: `public/data/benguet.geojson`** (simplified structure)

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "ATOK"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.68, 16.58], [120.72, 16.58], [120.72, 16.62], [120.68, 16.62], [120.68, 16.58]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "BAKUN"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.65, 16.77], [120.69, 16.77], [120.69, 16.81], [120.65, 16.81], [120.65, 16.77]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "BOKOD"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.82, 16.45], [120.86, 16.45], [120.86, 16.49], [120.82, 16.49], [120.82, 16.45]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "BUGUIAS"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.80, 16.72], [120.84, 16.72], [120.84, 16.76], [120.80, 16.76], [120.80, 16.72]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "ITOGON"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.67, 16.35], [120.71, 16.35], [120.71, 16.39], [120.67, 16.39], [120.67, 16.35]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "KABAYAN"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.83, 16.60], [120.87, 16.60], [120.87, 16.64], [120.83, 16.64], [120.83, 16.60]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "KAPANGAN"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.57, 16.57], [120.61, 16.57], [120.61, 16.61], [120.57, 16.61], [120.57, 16.57]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "KIBUNGAN"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.62, 16.69], [120.66, 16.69], [120.66, 16.73], [120.62, 16.73], [120.62, 16.69]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "LATRINIDAD"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.58, 16.44], [120.62, 16.44], [120.62, 16.48], [120.58, 16.48], [120.58, 16.44]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "MANKAYAN"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.79, 16.85], [120.83, 16.85], [120.83, 16.89], [120.79, 16.89], [120.79, 16.85]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "SABLAN"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.52, 16.47], [120.56, 16.47], [120.56, 16.51], [120.52, 16.51], [120.52, 16.47]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "TUBA"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.55, 16.30], [120.59, 16.30], [120.59, 16.34], [120.55, 16.34], [120.55, 16.30]]]
      }
    },
    {
      "type": "Feature",
      "properties": {"name": "TUBLAY"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[120.60, 16.50], [120.64, 16.50], [120.64, 16.54], [120.60, 16.54], [120.60, 16.50]]]
      }
    }
  ]
}
```

**Note:** This is a simplified rectangular approximation. Download actual boundaries from GADM for accurate shapes!

---

## TESTING YOUR GEOJSON

### Quick Test in Browser Console

```javascript
// Load and display
fetch('/data/benguet.geojson')
    .then(r => r.json())
    .then(data => {
        console.log('Features:', data.features.length);
        console.log('Municipalities:', data.features.map(f => f.properties.name));
    });
```

### Validate GeoJSON

Use online validator:
- https://geojson.io/ (visualize and edit)
- https://geojsonlint.com/ (validate syntax)

---

## PERFORMANCE TIPS

1. **Simplify geometry**: Use mapshaper to reduce points
2. **Cache GeoJSON**: Save to browser localStorage
3. **Use TopoJSON**: Smaller file size than GeoJSON
4. **Lazy load**: Only load when needed
5. **CDN hosting**: Faster download times

### Convert to TopoJSON (smaller files)

```bash
npm install -g topojson

geo2topo benguet.geojson > benguet.topojson
```

Then use with:
```javascript
import * as topojson from 'topojson-client';

fetch('/data/benguet.topojson')
    .then(r => r.json())
    .then(topology => {
        const geojson = topojson.feature(topology, topology.objects.collection);
        L.geoJSON(geojson).addTo(map);
    });
```

---

## TROUBLESHOOTING

### Problem: GeoJSON not showing
**Solution:** Check coordinate order - GeoJSON uses [longitude, latitude], not [lat, lng]

### Problem: Municipality names don't match
**Solution:** Use the municipalityMapping object above

### Problem: Polygons overlap or look wrong
**Solution:** Download accurate boundaries from GADM, don't use approximations

### Problem: Map is slow
**Solution:** Simplify GeoJSON using mapshaper (reduce to 10-20% of original detail)

### Problem: Wrong projection
**Solution:** Ensure GeoJSON uses WGS84 (EPSG:4326) - the standard for web maps

---

## NEXT STEPS

1. ✅ Download actual GeoJSON from GADM
2. ✅ Save to `public/data/benguet.geojson`
3. ✅ Replace map markers with GeoJSON polygons
4. ✅ Test with production data coloring
5. ✅ Simplify if file is too large
6. ✅ Add hover effects and click events

**You're ready to build a professional-looking map!** 🗺️

---

## RESOURCES

- **GADM**: https://gadm.org/
- **PhilGIS**: https://philgis.org/
- **Mapshaper**: https://mapshaper.org/
- **GeoJSON.io**: https://geojson.io/
- **Leaflet Choropleth Tutorial**: https://leafletjs.com/examples/choropleth/
- **TopoJSON**: https://github.com/topojson/topojson

---

Generated: November 2, 2025
For: BenguetCropMap Project
