"""
Database Configuration for ML API
Supports MySQL/MariaDB and PostgreSQL (Render, Railway, etc.)
"""
import os
from urllib.parse import parse_qs, unquote, urlparse


def _as_int(value, default_value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default_value


def _parse_database_url():
    """Parse DATABASE_URL/DB_URL when provided by hosting platforms."""
    database_url = os.environ.get('DATABASE_URL') or os.environ.get('DB_URL')
    if not database_url:
        return None

    parsed = urlparse(database_url)
    scheme = (parsed.scheme or '').lower()

    if scheme.startswith('postgres'):
        db_type = 'postgresql'
        default_port = 5432
        default_sslmode = 'require'
    elif scheme.startswith('mysql'):
        db_type = 'mysql'
        default_port = 3306
        default_sslmode = ''
    else:
        return None

    query = parse_qs(parsed.query)
    return {
        'type': db_type,
        'host': parsed.hostname or '',
        'port': parsed.port or default_port,
        'database': (parsed.path or '').lstrip('/'),
        'user': unquote(parsed.username or ''),
        'password': unquote(parsed.password or ''),
        'charset': os.environ.get('DB_CHARSET', 'utf8mb4'),
        'sslmode': os.environ.get('DB_SSLMODE', query.get('sslmode', [default_sslmode])[0]),
    }


def _build_db_config():
    parsed_config = _parse_database_url()
    if parsed_config:
        return parsed_config

    db_type = os.environ.get('DB_CONNECTION', os.environ.get('DB_TYPE', 'mysql')).lower()
    if db_type.startswith('postgres'):
        default_port = 5432
        default_sslmode = 'require'
    else:
        db_type = 'mysql'
        default_port = 3306
        default_sslmode = ''

    return {
        'type': db_type,
        'host': os.environ.get('DB_HOST', os.environ.get('PGHOST', '127.0.0.1')),
        'port': _as_int(os.environ.get('DB_PORT', os.environ.get('PGPORT', default_port)), default_port),
        'database': os.environ.get('DB_DATABASE', os.environ.get('PGDATABASE', 'pasyadatabase')),
        'user': os.environ.get('DB_USERNAME', os.environ.get('DB_USER', os.environ.get('PGUSER', 'root'))),
        'password': os.environ.get('DB_PASSWORD', os.environ.get('PGPASSWORD', '')),
        'charset': os.environ.get('DB_CHARSET', 'utf8mb4'),
        'sslmode': os.environ.get('DB_SSLMODE', default_sslmode),
    }

# Database configuration sourced from DATABASE_URL or DB_* variables
DB_CONFIG = _build_db_config()

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
