import sqlite3
import pandas as pd

# Path to your orbit.db
db_path = "/home/mvu9/datasets/glomeruli/orbit.db"  # Update this path

# Connect to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 1: List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
table_names = [t[0] for t in tables]
print("Tables found in orbit.db:")
for name in table_names:
    print(f" - {name}")

# Step 2: Print column names for each table and convert to DataFrame
dfs = {}
for table in table_names:
    print(f"\n--- Columns in table: {table} ---")
    cursor.execute(f"PRAGMA table_info({table});")
    columns = cursor.fetchall()
    col_names = [col[1] for col in columns]
    print(col_names)

    # Step 3: Convert to DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    dfs[table] = df
    print(f"Loaded {len(df)} rows into DataFrame.")

# Optional: Save to CSV or preview DataFrame
# dfs["label"].head().to_csv("label_preview.csv", index=False)

# Close connection
conn.close()
