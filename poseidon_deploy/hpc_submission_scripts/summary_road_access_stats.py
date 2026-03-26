import os
import glob
import pandas as pd
import numpy as np

def summarize_roadway_stats(main_dir):
    """
    Aggregates roadway accessibility statistics from all flood events using dynamic time intervals.
    """
    # Find all CSV files recursively
    search_pattern = os.path.join(main_dir, "*", "canal_dr_transect_EOR_revised_accessibility_time_series.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No summary CSVs found in {main_dir}")
        return None

    print(f"Found {len(csv_files)} event files. Processing...")

    event_stats = []

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            
            # Ensure Time is datetime
            df['Time'] = pd.to_datetime(df['Time'])
            
            # Sort just in case
            df = df.sort_values('Time')

            # --- DYNAMIC DURATION CALCULATION ---
            # Calculate the time difference from the PREVIOUS row to CURRENT row.
            # We fill the first NaN with the median interval to avoid losing the first data point's weight,
            # or you can fill with 0 (conservative) or forward fill.
            # Here we use backfill (duration of first step is assumed to be same as second step)
            time_diffs = df['Time'].diff().dt.total_seconds().div(60) # Minutes
            
            # Fill the first row (which is NaT/NaN) with the next valid interval
            # This assumes the first image represents a duration similar to the others.
            if len(time_diffs) > 1:
                first_interval = time_diffs.iloc[1]
                time_diffs.iloc[0] = first_interval
            else:
                time_diffs.iloc[0] = 0 # Single point has no duration
            
            df['duration_min'] = time_diffs
            
            # --- IMPASSABLE CALCULATIONS ---
            # Sum the duration of intervals where Impassable == 1
            impassable_min = df.loc[df['Impassable'] == 1, 'duration_min'].sum()
            
            # Total event duration
            total_event_min = df['duration_min'].sum()
            
            # Proportion of time this event was impassable
            prop_impassable = impassable_min / total_event_min if total_event_min > 0 else 0
            
            # --- DEPTH STATS ---
            mean_depth = df['MeanDepth'].mean()
            max_depth = df['MaxDepth'].max()
            min_depth = df['MinDepth'].min()

            event_stats.append({
                'impassable_minutes': impassable_min,
                'prop_impassable': prop_impassable,
                'mean_depth': mean_depth,
                'max_depth': max_depth,
                'min_depth': min_depth
            })
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Convert to DataFrame for aggregation
    stats_df = pd.DataFrame(event_stats)

    if stats_df.empty:
        print("No valid data found.")
        return

    # --- CALCULATE FINAL SUMMARY ---
    summary = {
        "Total Events Processed": len(stats_df),
        
        # Time Sums (Total across all events)
        "Total Impassable Time (Minutes)": stats_df['impassable_minutes'].sum(),
        "Total Impassable Time (Hours)": stats_df['impassable_minutes'].sum() / 60,
        
        # Time Averages (Per event)
        "Avg Proportion of Time Impassable": stats_df['prop_impassable'].mean(),
        
        # Medians
        "Median Impassable Time (Minutes)": stats_df['impassable_minutes'].median(),
        "Median Impassable Time (Hours)": stats_df['impassable_minutes'].median() / 60,
        
        # Depth Averages
        "Mean of Mean Depths (m)": stats_df['mean_depth'].mean(),
        "Mean of Max Depths (m)": stats_df['max_depth'].mean(),
        "Mean of Min Depths (m)": stats_df['min_depth'].mean(),
    }

    # Print nicely formatted output
    print("\n=== Roadway Accessibility Summary (Dynamic Intervals) ===")
    for key, val in summary.items():
        if "Total Events" in key:
            print(f"{key}: {int(val)}")
        elif "Proportion" in key:
            print(f"{key}: {val:.2%}") # Format as percentage
        else:
            print(f"{key}: {val:.4f}")
            
    return pd.DataFrame([summary])

# --- RUN BLOCK ---
if __name__ == "__main__":
    # Update this to your actual data path
    MAIN_DIR = "/rsstu/users/k/kanarde/NASA-Sunnyverse/rmccune/poseidon/data/carolina_beach/flood_events"
    
    summarize_roadway_stats(MAIN_DIR)
