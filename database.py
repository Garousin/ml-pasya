"""
Database Manager for ML API
Handles connection to MySQL/MariaDB database
"""
import pandas as pd
import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from db_config import DB_CONFIG, TABLES, COLUMN_MAPPING

class DatabaseManager:
    """Manages database connections and queries"""
    
    def __init__(self):
        self.config = DB_CONFIG
        self._connection = None
    
    @contextmanager
    def get_connection(self, use_dict_cursor=False):
        """Context manager for database connections
        
        Args:
            use_dict_cursor: If True, use DictCursor (for cursor.execute queries).
                            If False, use default cursor (for pd.read_sql queries).
        """
        conn = None
        try:
            connect_args = {
                'host': self.config['host'],
                'port': self.config['port'],
                'user': self.config['user'],
                'password': self.config['password'],
                'database': self.config['database'],
                'charset': self.config['charset']
            }
            # Only use DictCursor for cursor-based queries, not for pd.read_sql
            if use_dict_cursor:
                connect_args['cursorclass'] = DictCursor
            conn = pymysql.connect(**connect_args)
            yield conn
        except pymysql.Error as e:
            print(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def test_connection(self):
        """Test if database connection works"""
        try:
            with self.get_connection(use_dict_cursor=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def get_crop_production_data(self, filters=None):
        """
        Fetch crop production data from database
        
        Args:
            filters: dict with optional filters like:
                - municipality: str or list
                - crop: str or list  
                - year_from: int
                - year_to: int
                - month: str or list
        
        Returns:
            pandas DataFrame with crop production data
        """
        query = f"""
            SELECT 
                municipality as MUNICIPALITY,
                farm_type as FARM_TYPE,
                year as YEAR,
                month as MONTH,
                crop as CROP,
                area_planted_ha as AREA_PLANTED,
                area_harvested_ha as AREA_HARVESTED,
                production_mt as PRODUCTION,
                productivity_mt_ha as PRODUCTIVITY
            FROM {TABLES['crop_productions']}
            WHERE 1=1
        """
        params = []
        
        if filters:
            if 'municipality' in filters:
                if isinstance(filters['municipality'], list):
                    placeholders = ','.join(['%s'] * len(filters['municipality']))
                    query += f" AND municipality IN ({placeholders})"
                    params.extend(filters['municipality'])
                else:
                    query += " AND municipality = %s"
                    params.append(filters['municipality'])
            
            if 'crop' in filters:
                if isinstance(filters['crop'], list):
                    placeholders = ','.join(['%s'] * len(filters['crop']))
                    query += f" AND crop IN ({placeholders})"
                    params.extend(filters['crop'])
                else:
                    query += " AND crop = %s"
                    params.append(filters['crop'])
            
            if 'year_from' in filters:
                query += " AND year >= %s"
                params.append(filters['year_from'])
            
            if 'year_to' in filters:
                query += " AND year <= %s"
                params.append(filters['year_to'])
            
            if 'month' in filters:
                if isinstance(filters['month'], list):
                    placeholders = ','.join(['%s'] * len(filters['month']))
                    query += f" AND month IN ({placeholders})"
                    params.extend(filters['month'])
                else:
                    query += " AND month = %s"
                    params.append(filters['month'])
        
        query += " ORDER BY year, month"
        
        try:
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn, params=params if params else None)
                return df
        except Exception as e:
            print(f"Error fetching crop production data: {e}")
            return pd.DataFrame()
    
    def get_all_data_for_training(self):
        """Fetch all crop production data for model training"""
        return self.get_crop_production_data()
    
    def get_historical_data_range(self):
        """Get min and max years from historical data"""
        query = f"""
            SELECT MIN(year) as min_year, MAX(year) as max_year 
            FROM {TABLES['crop_productions']}
        """
        try:
            with self.get_connection(use_dict_cursor=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    result = cursor.fetchone()
                    if result:
                        return result['min_year'], result['max_year']
                    return None, None
        except Exception as e:
            print(f"Error getting data range: {e}")
            return None, None
    
    def get_available_crops(self):
        """Get list of unique crops in database"""
        query = f"SELECT DISTINCT crop FROM {TABLES['crop_productions']} ORDER BY crop"
        try:
            with self.get_connection(use_dict_cursor=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    return [row['crop'] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting crops: {e}")
            return []
    
    def get_available_municipalities(self):
        """Get list of unique municipalities in database"""
        query = f"SELECT DISTINCT municipality FROM {TABLES['crop_productions']} ORDER BY municipality"
        try:
            with self.get_connection(use_dict_cursor=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    return [row['municipality'] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting municipalities: {e}")
            return []
    
    def log_prediction(self, prediction_data):
        """Log a prediction request to the database"""
        query = f"""
            INSERT INTO {TABLES['prediction_logs']} 
            (municipality, farm_type, year, month, crop, area_planted_ha, 
             predicted_production_mt, request_ip, processing_time_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.get_connection(use_dict_cursor=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (
                        prediction_data.get('municipality'),
                        prediction_data.get('farm_type'),
                        prediction_data.get('year'),
                        prediction_data.get('month'),
                        prediction_data.get('crop'),
                        prediction_data.get('area_planted'),
                        prediction_data.get('predicted_production'),
                        prediction_data.get('request_ip'),
                        prediction_data.get('processing_time_ms')
                    ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error logging prediction: {e}")
            return False
    
    def save_forecast(self, forecast_data):
        """Save or update a forecast in the database"""
        query = f"""
            INSERT INTO {TABLES['forecasts']} 
            (crop, municipality, year, production_mt, confidence_lower, 
             confidence_upper, trend_direction, growth_rate_percent)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                production_mt = VALUES(production_mt),
                confidence_lower = VALUES(confidence_lower),
                confidence_upper = VALUES(confidence_upper),
                trend_direction = VALUES(trend_direction),
                growth_rate_percent = VALUES(growth_rate_percent),
                generated_at = CURRENT_TIMESTAMP
        """
        try:
            with self.get_connection(use_dict_cursor=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (
                        forecast_data.get('crop'),
                        forecast_data.get('municipality'),
                        forecast_data.get('year'),
                        forecast_data.get('production_mt'),
                        forecast_data.get('confidence_lower'),
                        forecast_data.get('confidence_upper'),
                        forecast_data.get('trend_direction'),
                        forecast_data.get('growth_rate_percent')
                    ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving forecast: {e}")
            return False
    
    def get_forecasts(self, crop=None, municipality=None, year=None):
        """Retrieve forecasts from database"""
        query = f"SELECT * FROM {TABLES['forecasts']} WHERE 1=1"
        params = []
        
        if crop:
            query += " AND crop = %s"
            params.append(crop)
        if municipality:
            query += " AND municipality = %s"
            params.append(municipality)
        if year:
            query += " AND year = %s"
            params.append(year)
        
        query += " ORDER BY crop, municipality, year"
        
        try:
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn, params=params if params else None)
                return df
        except Exception as e:
            print(f"Error getting forecasts: {e}")
            return pd.DataFrame()


# Global database manager instance
db_manager = DatabaseManager()


def load_data_from_database():
    """
    Load crop production data from database
    Falls back to CSV if database is not available
    """
    if db_manager.test_connection():
        print("[DB] Loading data from database...")
        df = db_manager.get_all_data_for_training()
        if not df.empty:
            print(f"[DB] Loaded {len(df)} records from database")
            return df
        print("[DB] Database is empty, falling back to CSV")
    else:
        print("[DB] Database not available, falling back to CSV")
    
    # Fallback to CSV
    try:
        df = pd.read_csv('fulldataset.csv')
        print(f"[CSV] Loaded {len(df)} records from CSV")
        return df
    except Exception as e:
        print(f"[ERROR] Could not load data: {e}")
        return pd.DataFrame()
