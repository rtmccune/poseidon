import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# --- 1. Directory Configuration & Arguments ---
parser = argparse.ArgumentParser(description="Plot POSEIDON accessibility time-series.")
parser.add_argument('--labels', type=str, default=None, 
                    help="Path to the segmentation_labels.csv to color code the plots.")
args = parser.parse_args()

base_dir = 'flood_events'
out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)

# Load the good segmentations and extract their timestamps
good_timestamps = set()
if args.labels and os.path.exists(args.labels):
    print(f"Loading segmentation labels from: {args.labels}")
    labels_df = pd.read_csv(args.labels)
    # Filter to only the images marked as good (1)
    good_images = labels_df[labels_df['Segmentation_Score'] == 1]['Filename'].dropna()
    
    for fname in good_images:
        # Extract the 14-digit timestamp (YYYYMMDDHHMMSS) from the filename
        match = re.search(r'(\d{14})', fname)
        if match:
            good_timestamps.add(match.group(1))
            
    print(f"Found {len(good_timestamps)} 'good' timestamps to highlight.")
else:
    print("No valid segmentation labels provided. Using default colors.")

# Find all time-series CSVs within the flood event subdirectories
csv_pattern = os.path.join(base_dir, '*', '*_accessibility_time_series.csv')
csv_files = glob.glob(csv_pattern)

# Group files by polygon name to handle multiple areas dynamically
polygon_files = {}
for file_path in csv_files:
    basename = os.path.basename(file_path)
    poly_name = basename.replace('_accessibility_time_series.csv', '')
    if poly_name not in polygon_files:
        polygon_files[poly_name] = []
    polygon_files[poly_name].append(file_path)

# --- 2. Data Processing and Plotting ---
for poly_name, files in polygon_files.items():
    peak_depths = []
    peak_coverages = []
    peak_colors = []

    # Initialize Plot 1 (Evolution Plot)
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    for f in files:
        # Read data
        df = pd.read_csv(f)
        
        # Calculate Percentage Covered
        df['Coverage'] = (df['FloodedPixels'] / df['TotalPixels']) * 100
        
        # Determine the time column dynamically
        time_col = None
        for col in ['time', 'datetime', 'timestamp', 'date', 'Time', 'Date', 'Datetime']:
            if col in df.columns:
                time_col = col
                break
        
        # Match timestamps to color individual points
        if time_col and len(good_timestamps) > 0:
            # Convert CSV timestamps to YYYYMMDDHHMMSS format string
            csv_times = pd.to_datetime(df[time_col], errors='coerce').dt.strftime('%Y%m%d%H%M%S')
            
            # Create an array of colors: green if the timestamp is in our good set, else blue
            colors = np.where(csv_times.isin(good_timestamps), 'green', 'blue')
            
            # If any point in this event is green, mark the whole event as 'green' for Plot 2
            event_has_good = 'green' in colors
        else:
            colors = 'blue'  # Default color array behavior for matplotlib scatter
            event_has_good = False
        
        # Plot 1: Scatter all points for the current event
        ax1.scatter(df['MaxDepth'], df['Coverage'], alpha=0.5, c=colors)
        
        # Extract peak values for this specific event for Plots 2-5
        peak_depths.append(df['MaxDepth'].max())
        peak_coverages.append(df['Coverage'].max())
        peak_colors.append('green' if event_has_good else 'blue')

    # Finalize Plot 1
    ax1.set_xlabel('Maximum Depth on Roadway (m)')
    ax1.set_ylabel('Roadway Percentage Covered (%)')
    ax1.set_title(f'Roadway Coverage vs Maximum Depth\n({poly_name})')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend if we are coloring points
    if len(good_timestamps) > 0:
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8)]
        ax1.legend(custom_lines, ['Good Segmentation', 'Standard/Poor Segmentation'])

    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, f'{poly_name}_1_evolution_plot.jpg'))
    plt.close(fig1)

    # Convert peaks to numpy arrays and sort for statistical plots
    peak_depths = np.array(peak_depths)
    peak_coverages = np.array(peak_coverages)
    peak_colors = np.array(peak_colors)
    
    sorted_indices = np.argsort(peak_depths)
    sorted_depths = peak_depths[sorted_indices]
    n = len(sorted_depths)

    # --- Plot 2: Summary Scatter ---
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
    ax2.scatter(peak_depths, peak_coverages, c=peak_colors, alpha=0.6, edgecolors='black', s=40)
    ax2.set_xlabel('Maximum Event Depth on Roadway (m)')
    ax2.set_ylabel('Maximum Roadway Percentage Covered (%)')
    ax2.set_title(f'Peak Roadway Coverage vs Peak Depth per Event\n({poly_name})')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    if len(good_timestamps) > 0:
        ax2.legend(custom_lines, ['Event Contains Good Segmentation', 'Standard/Poor Event'])
        
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, f'{poly_name}_2_summary_scatter.png'))
    plt.close(fig2)

    # --- Plot 3: Cumulative Distribution Function (CDF) ---
    cdf_probs = np.arange(1, n + 1) / n
    fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)
    ax3.step(sorted_depths, cdf_probs, where='post', color='blue', linewidth=2)
    ax3.plot(sorted_depths, cdf_probs, 'bo', alpha=0.7)
    ax3.set_xlabel('Maximum Event Depth on Roadway (m)')
    ax3.set_ylabel('Cumulative Probability P(X <= x)')
    ax3.set_title(f'Cumulative Distribution Function (CDF) of Peak Depths\n({poly_name})')
    ax3.grid(True, linestyle='--', alpha=0.7)
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, f'{poly_name}_3_depth_cdf.png'))
    plt.close(fig3)

    # --- Plot 4: Probability Density Function (PDF) ---
    fig4, ax4 = plt.subplots(figsize=(10, 6), dpi=300)
    ax4.hist(peak_depths, bins=5, density=True, color='skyblue', edgecolor='black', alpha=0.9)
    ax4.set_xlabel('Maximum Event Depth on Roadway (m)')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Probability Density Function (PDF) of Peak Depths\n({poly_name})')
    ax4.grid(True, linestyle='--', alpha=0.7)
    fig4.tight_layout()
    fig4.savefig(os.path.join(out_dir, f'{poly_name}_4_depth_pdf.png'))
    plt.close(fig4)

    # --- Plot 5: Exceedance Probability ---
    exceedance_probs = np.arange(n, 0, -1) / n 
    fig5, ax5 = plt.subplots(figsize=(10, 6), dpi=300)
    ax5.step(sorted_depths, exceedance_probs, where='post', color='red', linewidth=2)
    ax5.plot(sorted_depths, exceedance_probs, 'ro', alpha=0.7)
    ax5.set_xlabel('Maximum Event Depth on Roadway (m)')
    ax5.set_ylabel('Exceedance Probability P(X >= x)')
    ax5.set_title(f'Exceedance Probability of Peak Event Depths\n({poly_name})')
    ax5.grid(True, linestyle='--', alpha=0.7)
    fig5.tight_layout()
    fig5.savefig(os.path.join(out_dir, f'{poly_name}_5_depth_exceedance.png'))
    plt.close(fig5)

print(f"Processing complete. Plots saved to the '{out_dir}' directory.")
