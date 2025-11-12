import pandas as pd
import os

print("Fixing CSV files...")

data_dir = 'data'
fixed = 0
failed = 0

for filename in os.listdir(data_dir):
    if filename.endswith('.csv') and filename != 'download_summary.csv':
        filepath = os.path.join(data_dir, filename)
        try:
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Find date column
            date_col = None
            for col in ['Date', 'date', 'Unnamed: 0', 'Datetime']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                # Convert to datetime and set as index
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                df.index.name = 'Date'
                
                # Ensure numeric columns
                for col in df.columns:
                    if col not in ['Date']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Save back
                df.to_csv(filepath)
                print(f"✓ Fixed {filename}")
                fixed += 1
            else:
                print(f"✗ No date column in {filename}")
                failed += 1
                
        except Exception as e:
            print(f"✗ Error with {filename}: {e}")
            failed += 1

print(f"\nFixed: {fixed}, Failed: {failed}")