import os
import shutil
import glob
import pandas as pd

# --- Configuration ---
base_dir = '.' 
images_source_dir = os.path.join(base_dir, 'images', 'stitched_comparisons')
base_dest_dir = os.path.join(base_dir, 'images', 'anomaly_comparisons')

anomaly_reports = glob.glob('*_anomaly_report.csv')

if not anomaly_reports:
    print("No anomaly reports found. Please run the anomaly detection script first.")
    exit()

total_copied = 0
not_found = 0

print(f"Found {len(anomaly_reports)} anomaly report(s). Starting extraction...\n")

for report in anomaly_reports:
    # Extract the polygon name from the report filename to create a specific subfolder
    poly_name = report.replace('_anomaly_report.csv', '')
    poly_dest_dir = os.path.join(base_dest_dir, poly_name)
    
    # Ensure this specific polygon's destination folder exists
    os.makedirs(poly_dest_dir, exist_ok=True)
    
    print(f"Reading {report} -> Saving to {poly_dest_dir}/")
    df = pd.read_csv(report)
    
    poly_copied_count = 0
    
    for index, row in df.iterrows():
        raw_time = row['Time']
        dt_obj = pd.to_datetime(raw_time)
        time_code = dt_obj.strftime('%Y%m%d%H%M%S')
        
        search_pattern = os.path.join(images_source_dir, f"*{time_code}*.jpg")
        matches = glob.glob(search_pattern)
        
        if matches:
            for match in matches:
                filename = os.path.basename(match)
                destination_path = os.path.join(poly_dest_dir, filename)
                
                if not os.path.exists(destination_path):
                    shutil.copy2(match, destination_path)
                    poly_copied_count += 1
                    total_copied += 1
        else:
            print(f"  -> Warning: No image found for timestamp {raw_time}")
            not_found += 1
            
    print(f"  Successfully copied {poly_copied_count} images for {poly_name}.\n")

print("--- Extraction Complete ---")
print(f"Total isolated images safely copied: {total_copied}")
if not_found > 0:
    print(f"Missing images: {not_found}")
