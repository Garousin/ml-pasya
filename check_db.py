"""Check database contents"""
import pymysql
from db_config import DB_CONFIG

conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
cursor = conn.cursor()

# Check for header rows
cursor.execute("SELECT COUNT(*) as cnt FROM crop_productions WHERE year = 'YEAR' OR municipality = 'MUNICIPALITY'")
print(f"Header-like rows: {cursor.fetchone()['cnt']}")

# Check total rows
cursor.execute("SELECT COUNT(*) as cnt FROM crop_productions")
print(f"Total rows: {cursor.fetchone()['cnt']}")

# Get sample data
cursor.execute("SELECT * FROM crop_productions WHERE municipality = 'ATOK' AND crop = 'CABBAGE' LIMIT 3")
for row in cursor.fetchall():
    print(row)

conn.close()
