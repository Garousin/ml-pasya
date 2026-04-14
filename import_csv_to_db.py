"""
Import CSV data into MySQL database
Run this once to populate the database with historical crop production data
"""
import pandas as pd
import pymysql
from db_config import DB_CONFIG, TABLES

def create_tables(cursor):
    """Create the required tables if they don't exist"""
    
    # Crop Productions Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['crop_productions']} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            municipality VARCHAR(100) NOT NULL,
            farm_type VARCHAR(50) NOT NULL,
            year INT NOT NULL,
            month VARCHAR(10) NOT NULL,
            crop VARCHAR(100) NOT NULL,
            area_planted_ha DECIMAL(10,2) NOT NULL,
            area_harvested_ha DECIMAL(10,2),
            productivity_mt_ha DECIMAL(10,2),
            production_mt DECIMAL(10,2) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_municipality (municipality),
            INDEX idx_crop (crop),
            INDEX idx_year (year),
            INDEX idx_crop_muni_year (crop, municipality, year)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    
    # Forecasts Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['forecasts']} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            crop VARCHAR(100) NOT NULL,
            municipality VARCHAR(100) NOT NULL,
            year INT NOT NULL,
            production_mt DECIMAL(10,2) NOT NULL,
            confidence_lower DECIMAL(10,2),
            confidence_upper DECIMAL(10,2),
            trend_direction VARCHAR(20),
            growth_rate_percent DECIMAL(5,2),
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_forecast (crop, municipality, year),
            INDEX idx_crop_muni (crop, municipality)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    
    # Prediction Logs Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['prediction_logs']} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            municipality VARCHAR(100),
            farm_type VARCHAR(50),
            year INT,
            month VARCHAR(10),
            crop VARCHAR(100),
            area_planted_ha DECIMAL(10,2),
            predicted_production_mt DECIMAL(10,2),
            request_ip VARCHAR(50),
            user_agent VARCHAR(255),
            processing_time_ms DECIMAL(10,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_created (created_at),
            INDEX idx_crop (crop)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    
    # Model Metadata Table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['model_metadata']} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50),
            version VARCHAR(20),
            r2_score DECIMAL(5,4),
            mae DECIMAL(10,2),
            mape DECIMAL(10,2),
            training_samples INT,
            is_active BOOLEAN DEFAULT FALSE,
            trained_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata_json TEXT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    
    print("Tables created successfully!")

def import_csv_data(cursor, conn):
    """Import data from CSV file into database"""
    
    # Read CSV
    print("Reading CSV file...")
    df = pd.read_csv('fulldataset.csv')
    print(f"Found {len(df)} records in CSV")
    
    # Check if data already exists
    cursor.execute(f"SELECT COUNT(*) as count FROM {TABLES['crop_productions']}")
    existing = cursor.fetchone()['count']
    
    if existing > 0:
        print(f"Database already has {existing} records.")
        response = input("Do you want to clear and reimport? (y/n): ")
        if response.lower() != 'y':
            print("Import cancelled.")
            return
        
        cursor.execute(f"TRUNCATE TABLE {TABLES['crop_productions']}")
        print("Existing data cleared.")
    
    # Prepare insert query
    insert_query = f"""
        INSERT INTO {TABLES['crop_productions']} 
        (municipality, farm_type, year, month, crop, area_planted_ha, 
         area_harvested_ha, productivity_mt_ha, production_mt)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    # Clean the data first - convert to numeric, coerce errors to NaN
    print("Cleaning data...")
    df['Area planted(ha)'] = pd.to_numeric(df['Area planted(ha)'], errors='coerce')
    df['Area harvested(ha)'] = pd.to_numeric(df['Area harvested(ha)'], errors='coerce')
    df['Productivity(mt/ha)'] = pd.to_numeric(df['Productivity(mt/ha)'], errors='coerce')
    df['Production(mt)'] = pd.to_numeric(df['Production(mt)'], errors='coerce')
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    
    # Drop rows with missing essential values
    before_clean = len(df)
    df = df.dropna(subset=['MUNICIPALITY', 'CROP', 'YEAR', 'Area planted(ha)', 'Production(mt)'])
    after_clean = len(df)
    print(f"  Removed {before_clean - after_clean} rows with invalid data")
    
    # Insert in batches
    batch_size = 1000
    total = len(df)
    inserted = 0
    skipped = 0
    
    print("Importing data...")
    for i in range(0, total, batch_size):
        batch = df.iloc[i:i+batch_size]
        
        values = []
        for _, row in batch.iterrows():
            try:
                values.append((
                    str(row['MUNICIPALITY']),
                    str(row['FARM TYPE']) if pd.notna(row['FARM TYPE']) else 'UNKNOWN',
                    int(row['YEAR']),
                    str(row['MONTH']) if pd.notna(row['MONTH']) else 'UNKNOWN',
                    str(row['CROP']),
                    float(row['Area planted(ha)']),
                    float(row['Area harvested(ha)']) if pd.notna(row['Area harvested(ha)']) else None,
                    float(row['Productivity(mt/ha)']) if pd.notna(row['Productivity(mt/ha)']) else None,
                    float(row['Production(mt)'])
                ))
            except (ValueError, TypeError) as e:
                skipped += 1
                continue
        
        if values:
            cursor.executemany(insert_query, values)
            conn.commit()
        inserted += len(values)
        print(f"  Imported {inserted}/{total} records ({100*inserted/total:.1f}%)")
    
    print(f"\nSuccessfully imported {inserted} records!")

def main():
    print("="*60)
    print("CSV to Database Import Tool")
    print("="*60)
    print(f"\nDatabase: {DB_CONFIG['database']}@{DB_CONFIG['host']}")
    print()
    
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database'],
            charset=DB_CONFIG['charset'],
            cursorclass=pymysql.cursors.DictCursor
        )
        print("[OK] Connected to database")
        
        with conn.cursor() as cursor:
            # Create tables
            print("\nCreating tables...")
            create_tables(cursor)
            conn.commit()
            
            # Import data
            print("\nImporting CSV data...")
            import_csv_data(cursor, conn)
        
        conn.close()
        print("\n[DONE] Import complete!")
        
    except pymysql.Error as e:
        print(f"\n[ERROR] Database error: {e}")
        print("\nMake sure:")
        print("1. MySQL/MariaDB is running (check XAMPP)")
        print("2. Database exists (create it first if needed)")
        print("3. Credentials in db_config.py are correct")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
