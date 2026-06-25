import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
base_dir = 'flood_events' 
csv_pattern = os.path.join(base_dir, '*', '*_accessibility_time_series.csv')
csv_files = glob.glob(csv_pattern)

LOCAL_Z_THRESHOLD = -1.0  
RESIDUAL_THRESHOLD = -15.0 

print(f"Found {len(csv_files)} CSV files. Grouping by polygon...")

polygon_files = {}
for file_path in csv_files:
    basename = os.path.basename(file_path)
    poly_name = basename.replace('_accessibility_time_series.csv', '')
    if poly_name not in polygon_files:
        polygon_files[poly_name] = []
    polygon_files[poly_name].append(file_path)

for poly_name, files in polygon_files.items():
    print(f"\n--- Processing Polygon: {poly_name} ---")
    
    all_data = []
    for f in files:
        df = pd.read_csv(f)
        df['Coverage'] = (df['FloodedPixels'] / df['TotalPixels']) * 100
        df['Event_Folder'] = os.path.basename(os.path.dirname(f))
        all_data.append(df[['Event_Folder', 'Time', 'MaxDepth', 'Coverage']])

    global_df = pd.concat(all_data, ignore_index=True)
    fit_mask = global_df['MaxDepth'] > 0.05 
    fit_data = global_df[fit_mask].copy()

    if fit_data.empty:
        continue

    # --- Build the Localized Baseline & Local Variance ---
    fit_data['Depth_Bin'] = (fit_data['MaxDepth'] * 100).astype(int) / 100

    spine = fit_data.groupby('Depth_Bin').agg(
        Core_Coverage=('Coverage', lambda x: x.quantile(0.75)),
        Local_Std=('Coverage', 'std')
    ).reset_index()

    # Smooth the core curve to trace the visual center
    spine['Smoothed_Core'] = spine['Core_Coverage'].rolling(window=10, center=True, min_periods=1).mean()
    
    # RE-INTRODUCE PLATEAU LOGIC: Force the core tracker to never dip
    spine['Smoothed_Core'] = spine['Smoothed_Core'].cummax()

    spine['Local_Std'] = spine['Local_Std'].bfill().ffill().clip(lower=5.0)
    spine['Smoothed_Std'] = spine['Local_Std'].rolling(window=10, center=True, min_periods=1).mean()

    # --- Apply to Global Data ---
    global_df['Expected_Coverage'] = np.interp(
        global_df['MaxDepth'], spine['Depth_Bin'], spine['Smoothed_Core']
    )
    global_df['Local_Std'] = np.interp(
        global_df['MaxDepth'], spine['Depth_Bin'], spine['Smoothed_Std']
    )

    global_df['Residual'] = global_df['Coverage'] - global_df['Expected_Coverage']
    global_df['Local_Z'] = global_df['Residual'] / global_df['Local_Std']

    # --- Flag Anomalies ---
    mask = (global_df['Local_Z'] < LOCAL_Z_THRESHOLD) #& (global_df['Residual'] < RESIDUAL_THRESHOLD)
    anomalies = global_df[mask].copy()
    anomalies = anomalies.sort_values(by='Local_Z', ascending=True)

    # --- Output ---
    if not anomalies.empty:
        out_csv = f'{poly_name}_anomaly_report.csv'
        report_df = anomalies[['Event_Folder', 'Time', 'MaxDepth', 'Coverage', 'Expected_Coverage', 'Local_Z']].round(3)
        report_df.to_csv(out_csv, index=False)
        print(f"Found {len(anomalies)} local anomalies. Saved to {out_csv}")
    else:
        print("No severe local anomalies found.")

    # --- Diagnostic Plot ---
    plt.figure(figsize=(12, 8), dpi=300)
    
    normal_pts = global_df[~global_df.index.isin(anomalies.index)]
    plt.scatter(normal_pts['MaxDepth'], normal_pts['Coverage'], color='gray', alpha=0.3, label='Normal Data')
    
    if not anomalies.empty:
        plt.scatter(anomalies['MaxDepth'], anomalies['Coverage'], color='red', alpha=0.8, edgecolor='black', label=f'Anomalies (Local Z < {LOCAL_Z_THRESHOLD})')
    
    plt.plot(spine['Depth_Bin'], spine['Smoothed_Core'], color='blue', linewidth=3, label='Visual Core Tracker (75th Pct + CumMax)')
    
    plt.xlabel('Maximum Event Depth on Roadway (m)')
    plt.ylabel('Roadway Percentage Covered (%)')
    plt.title(f'Local Density Anomaly Detection: {poly_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    out_plot = f'{poly_name}_local_anomaly_diagnostic.jpg'
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"Saved diagnostic visual to {out_plot}")
