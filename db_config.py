"""
Database Configuration for ML API
Connects to the Laravel/MySQL database
"""
import os

# Database configuration - Matches Laravel .env
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', '127.0.0.1'),
    'port': int(os.environ.get('DB_PORT', 3306)),
    'database': os.environ.get('DB_DATABASE', 'pasyadatabase'),
    'user': os.environ.get('DB_USERNAME', 'root'),
    'password': os.environ.get('DB_PASSWORD', ''),
    'charset': 'utf8mb4'
}

# Table names (matching Laravel migration)
TABLES = {
    'crop_productions': 'crop_productions',
    'forecasts': 'forecasts',
    'prediction_logs': 'prediction_logs',
    'model_metadata': 'model_metadata'
}

# Column mappings (database column -> CSV column)
COLUMN_MAPPING = {
    'municipality': 'MUNICIPALITY',
    'farm_type': 'FARM TYPE',
    'year': 'YEAR',
    'month': 'MONTH',
    'crop': 'CROP',
    'area_planted_ha': 'Area planted(ha)',
    'area_harvested_ha': 'Area harvested(ha)',
    'production_mt': 'Production(mt)',
    'productivity_mt_ha': 'Productivity(mt/ha)'
}
