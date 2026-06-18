import os
import glob
import argparse
import pandas as pd
import numpy as np

def summarize_roadway_stats(main_dir, transect_name, resolution=0.05):
    """
    Aggregates roadway accessibility statistics from all flood events.
    """
    search_pattern = os.path.join(main_dir, "*", f"{transect_name}_accessibility_time_series.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No summary CSVs found for ROI '{transect_name}' in {main_dir}")
        return None

    print(f"Found {len(csv_files)} event files. Processing...")

    event_stats = []
    # Calculate area of a single pixel (0.05 * 0.05 = 0.0025 m^2)
    pixel_area_m2 = resolution ** 2

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values('Time')

            # --- DYNAMIC DURATION CALCULATION ---
            time_diffs = df['Time'].diff().dt.total_seconds().div(60) 
            if len(time_diffs) > 1:
                time_diffs.iloc[0] = time_diffs.iloc[1]
            else:
                time_diffs.iloc[0] = 0 
            
            df['duration_min'] = time_diffs
            
            # --- IMPASSABLE CALCULATIONS ---
            impassable_min = df.loc[df['Impassable'] == 1, 'duration_min'].sum()
            total_event_min = df['duration_min'].sum()
            prop_impassable = impassable_min / total_event_min if total_event_min > 0 else 0
            
            # --- NEW: SPATIAL EXTENT CALCULATIONS ---
            # Check if this CSV has the new polygon spatial data
            if 'FloodedPixels' in df.columns and 'TotalPixels' in df.columns:
                df['FloodedArea_m2'] = df['FloodedPixels'] * pixel_area_m2
                df['PercentCovered'] = df['FloodedPixels'] / df['TotalPixels']
                
                max_flooded_area = df['FloodedArea_m2'].max()
                mean_flooded_area = df['FloodedArea_m2'].mean()
                mean_percent_covered = df['PercentCovered'].mean()
            else:
                max_flooded_area = np.nan
                mean_flooded_area = np.nan
                mean_percent_covered = np.nan

            # --- DEPTH STATS ---
            mean_depth = df['MeanDepth'].mean()
            max_depth = df['MaxDepth'].max()
            min_depth = df['MinDepth'].min()

            event_stats.append({
                'impassable_minutes': impassable_min,
                'prop_impassable': prop_impassable,
                'mean_depth': mean_depth,
                'max_depth': max_depth,
                'min_depth': min_depth,
                'max_flooded_area': max_flooded_area,
                'mean_flooded_area': mean_flooded_area,
                'mean_percent_covered': mean_percent_covered
            })
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    stats_df = pd.DataFrame(event_stats)

    if stats_df.empty:
        print("No valid data found.")
        return

    # --- CALCULATE FINAL SUMMARY ---
    summary = {
        "Total Events Processed": len(stats_df),
        
        "Total Impassable Time (Minutes)": stats_df['impassable_minutes'].sum(),
        "Total Impassable Time (Hours)": stats_df['impassable_minutes'].sum() / 60,
        
        "Mean Impassable Time per Event (Minutes)": stats_df['impassable_minutes'].mean(),
        "Mean Impassable Time per Event (Hours)": stats_df['impassable_minutes'].mean() / 60,
        
        "Median Impassable Time (Minutes)": stats_df['impassable_minutes'].median(),
        "Median Impassable Time (Hours)": stats_df['impassable_minutes'].median() / 60,
        
        "Avg Proportion of Time Impassable": stats_df['prop_impassable'].mean(),
        
        "Absolute Maximum Depth on Road (m)": stats_df['max_depth'].max(),
        
        "Mean of Mean Depths (m)": stats_df['mean_depth'].mean(),
        "Mean of Max Depths (m)": stats_df['max_depth'].mean(),
        "Mean of Min Depths (m)": stats_df['min_depth'].mean(),
    }

    # Only append spatial stats if the data was actually present
    if not stats_df['max_flooded_area'].isna().all():
        summary["Peak Spatial Extent (m^2)"] = stats_df['max_flooded_area'].max()
        summary["Average Spatial Extent (m^2)"] = stats_df['mean_flooded_area'].mean()
        summary["Average Roadway Coverage (%)"] = stats_df['mean_percent_covered'].mean()

    # Print nicely formatted output
    print(f"\n=== Roadway Accessibility Summary ===")
    print(f"Location: {main_dir}")
    print(f"Transect/ROI: {transect_name}\n")
    
    for key, val in summary.items():
        if "Total Events" in key:
            print(f"{key}: {int(val)}")
        elif "Proportion" in key or "Coverage (%)" in key:
            print(f"{key}: {val:.2%}") 
        else:
            print(f"{key}: {val:.4f}")
            
    return pd.DataFrame([summary])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize roadway accessibility stats.")
    parser.add_argument("--event_dir", type=str, required=True, help="Path to flood events directory")
    parser.add_argument("--transect_name", type=str, required=True, help="Base name of the transect/polygon")
    # Added an optional argument so you can easily change the grid size if you ever run this on different data
    parser.add_argument("--resolution", type=float, default=0.05, help="Grid resolution in meters")
    
    args = parser.parse_args()
    summarize_roadway_stats(args.event_dir, args.transect_name, args.resolution)
