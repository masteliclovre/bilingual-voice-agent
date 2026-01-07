import psycopg
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

# Provjeri zadnjih 5 poziva
cur.execute("""
    SELECT vapi_call_id, phone_number, started_at, status
    FROM calls
    ORDER BY started_at DESC
    LIMIT 5
""")

print("=== ZADNJIH 5 POZIVA ===")
rows = cur.fetchall()
if rows:
    for row in rows:
        print(f"ID: {row[0]}, Broj: {row[1]}, Vrijeme: {row[2]}, Status: {row[3]}")
else:
    print("Nema poziva u bazi!")

# Provjeri transcripte
cur.execute("""
    SELECT COUNT(*) FROM transcripts
""")
count = cur.fetchone()[0]
print(f"\n=== UKUPNO TRANSKRIPATA: {count} ===")

cur.close()
conn.close()
