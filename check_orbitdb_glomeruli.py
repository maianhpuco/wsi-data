import sqlite3
import pandas as pd
import os
import base64

def serialize_bytes(obj):
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')  # Convert bytes → base64 → utf8 str
    return obj
 
# Path to your Orbit SQLite database
db_path = "/home/mvu9/datasets/glomeruli/orbit.db"
output_dir = "/home/mvu9/datasets/glomeruli/orbit_json_exports"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Configure pandas to show full columns and cell contents
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_colwidth", None)

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 1: List all tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
table_names = [t[0] for t in tables]
print("Tables found in orbit.db:")
for name in table_names:
    print(f" - {name}")

# Step 2–5: Inspect and export each table
dfs = {}
for table in table_names:
    print(f"\n--- Columns in table: {table} ---")
    
    # Step 2: Get column names
    cursor.execute(f"PRAGMA table_info({table});")
    columns = cursor.fetchall()
    col_names = [col[1] for col in columns]
    print(col_names)

    # Step 3: Load the table into a DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    dfs[table] = df
    print(f"Loaded {len(df)} rows into DataFrame for table '{table}'.")

    # Step 4: Print first 3 rows
    # print("Sample rows:")
    # print(df.head(3))

    # Step 5: Save to JSON
    json_path = os.path.join(output_dir, f"{table}.json")
    # df.to_json(json_path, orient="records", lines=True)
    records = df.to_dict(orient="records")

    # Save safely using built-in json with error handling
    import json 
    with open(json_path, "w", encoding="utf-8") as f:
        for record in records:
            try:
                # Convert all byte fields in the record
                clean_record = {k: serialize_bytes(v) for k, v in record.items()}
                json.dump(clean_record, f, ensure_ascii=False)
                f.write("\n")
            except Exception as e:
                print(f" Skipped a record in {table} due to: {e}")

    print(f" Exported to {json_path}")

# Step 6: Close the connection
conn.close()
print("\n Done! All tables exported and inspected.")
