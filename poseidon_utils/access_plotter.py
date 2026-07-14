import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, minimize
from matplotlib.colors import LinearSegmentedColormap
import cmocean

# --- Define Mathematical Models ---
def logistic_curve(x, L, k, x0):
    """
    Forward Sigmoid: Predicts Coverage based on Depth.
    """
    exponent = np.clip(-k * (x - x0), -100, 100)
    return L / (1 + np.exp(exponent))

def logit_curve(c, L, k, x0):
    """
    Inverse Sigmoid (Logit): Predicts Depth based on Coverage.
    Used to calculate RMSE in meters.
    """
    epsilon = 1e-5
    L_eff = np.maximum(L, np.max(c) + epsilon) 
    c_clipped = np.clip(c, epsilon, L_eff - epsilon)
    return x0 - (1 / k) * np.log((L_eff / c_clipped) - 1)

def pinball_loss(params, x, y, q):
    """Custom loss function for quantile regression."""
    y_pred = logistic_curve(x, *params)
    err = y - y_pred
    return np.sum(np.maximum(q * err, (q - 1) * err))

def calculate_r2(y_true, y_pred):
    """Calculates the R-squared value."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def calculate_rmse(y_true, y_pred):
    """Calculates the Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# --- 1. Directory Configuration & Arguments ---
parser = argparse.ArgumentParser(description="Plot POSEIDON accessibility time-series.")
parser.add_argument('--labels', type=str, default=None, 
                    help="Path to the segmentation_labels.csv to color code the plots.")
parser.add_argument('--sensor', type=str, default=None, 
                    help="Path to the sensor_data.csv for 1:1 depth comparison.")
args = parser.parse_args()

base_dir = 'flood_events'
out_dir = 'plots'
os.makedirs(out_dir, exist_ok=True)

# Load Segmentation Labels
good_timestamps = set()
if args.labels and os.path.exists(args.labels):
    print(f"Loading segmentation labels from: {args.labels}")
    labels_df = pd.read_csv(args.labels)
    good_images = labels_df[labels_df['Segmentation_Score'] == 1]['Filename'].dropna()
    
    for fname in good_images:
        match = re.search(r'(\d{14})', fname)
        if match:
            good_timestamps.add(match.group(1))
            
    print(f"Found {len(good_timestamps)} 'good' timestamps to highlight.")
else:
    print("No valid segmentation labels provided. Using default colors.")

# Load Sensor Data for Nearest-Neighbor Matching
sensor_df_clean = None
if args.sensor and os.path.exists(args.sensor):
    print(f"Loading sensor data from: {args.sensor}")
    sensor_df = pd.read_csv(args.sensor)
    
    if 'road_water_level_adj' in sensor_df.columns:
        # Convert depth from feet to meters for 1:1 plotting
        sensor_df['SensorDepth_m'] = sensor_df['road_water_level_adj'] * 0.3048
        
        # Clamp negative sensor depths to 0.0 to match surface-only camera observations
        sensor_df['SensorDepth_m'] = sensor_df['SensorDepth_m'].clip(lower=0.0)
        
        # Parse datetime strictly as UTC to match image times safely
        sensor_df['datetime_utc'] = pd.to_datetime(sensor_df['date'], errors='coerce', utc=True)
        
        # --- FIXED: Calculate WSE BEFORE creating sensor_df_clean ---
        if 'sensor_water_level_adj' in sensor_df.columns: 
            sensor_df['SensorWSE_m'] = sensor_df['sensor_water_level_adj'] * 0.3048
        # ------------------------------------------------------------
        
        # Drop NaNs and ensure the dataframe is sorted by time (required for merge_asof)
        sensor_df_clean = sensor_df.dropna(subset=['datetime_utc', 'SensorDepth_m']).sort_values('datetime_utc')
        print(f"Loaded {len(sensor_df_clean)} valid sensor readings.")
    else:
        print("Warning: 'road_water_level_adj' column not found in sensor data.")

csv_pattern = os.path.join(base_dir, '*', '*_accessibility_time_series.csv')
csv_files = glob.glob(csv_pattern)

polygon_files = {}
for file_path in csv_files:
    basename = os.path.basename(file_path)
    poly_name = basename.replace('_accessibility_time_series.csv', '')
    if poly_name not in polygon_files:
        polygon_files[poly_name] = []
    polygon_files[poly_name].append(file_path)

metrics = ['MaxDepth', 'MeanDepth', 'MedianDepth']

# # Okabe-Ito Color Palette for Research Figures
OI_BLACK = '#000000'
OI_ORANGE = '#E69F00'
OI_DARKBLUE = '#0072B2'
OI_PURPLE = '#CC79A7'
OI_GRAY = '#999999'
OI_VERMILION = '#D55E00'

# --- Coastal Convergence Palette ---
coastal_colors = {
    'oceanic_blue': {'light': '#7DA6C6', 'base': '#1E6091', 'dark': '#0F3048'},
    'estuary_teal': {'light': '#81CFC6', 'base': '#2A9D8F', 'dark': '#154F48'},
    'erosion_rust': {'light': '#F3B6A7', 'base': '#E76F51', 'dark': '#8E3621'},
    'dune_gold':    {'light': '#F4E1B3', 'base': '#E9C46A', 'dark': '#9A7822'},
    'concrete_slate':{'light': '#A3AEB4', 'base': '#5C6B73', 'dark': '#2D3539'}
}

blue_colors = [coastal_colors['oceanic_blue']['light'], 
               coastal_colors['oceanic_blue']['base'], 
               coastal_colors['oceanic_blue']['dark']]
coastal_blues_cmap = LinearSegmentedColormap.from_list('coastal_blues', blue_colors)

# --- 2. Data Processing and Plotting ---
for poly_name, files in polygon_files.items():
    poly_dfs = []
    event_peaks = []
    
    for f in files:
        df = pd.read_csv(f)
        
        wse_csv_path = f.replace('_accessibility_time_series.csv', '_wse_time_series.csv')
        if os.path.exists(wse_csv_path):
            wse_df = pd.read_csv(wse_csv_path)
            if 'Time' in df.columns and 'Time' in wse_df.columns:
                df = pd.merge(df, wse_df[['Time', 'MeanWSE', 'MaxWSE', 'MedianWSE']], on='Time', how='left')
        
        if 'FloodedPixels' in df.columns and 'TotalPixels' in df.columns:
            df['Coverage'] = (df['FloodedPixels'] / df['TotalPixels']) * 100
        else:
            df['Coverage'] = np.nan
        
        time_col = None
        for col in ['time', 'datetime', 'timestamp', 'date', 'Time', 'Date', 'Datetime']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col and len(good_timestamps) > 0:
            # Parse CSV time to a true UTC datetime object
            df['datetime_utc'] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
            
            # Format back to string just for the filename matching logic
            csv_times = df['datetime_utc'].dt.strftime('%Y%m%d%H%M%S')
            colors = np.where(csv_times.isin(good_timestamps), 'green', 'blue')
            event_has_good = 'green' in colors
            
            # Perform nearest-neighbor time alignment with sensor data
            if sensor_df_clean is not None:
                # Sort df by time (required for merge_asof)
                df = df.sort_values('datetime_utc')
                merge_cols = ['datetime_utc', 'SensorDepth_m']
                if 'SensorWSE_m' in sensor_df_clean.columns:
                    merge_cols.append('SensorWSE_m')
                
                df = pd.merge_asof(
                    df, 
                    sensor_df_clean[merge_cols],
                    on='datetime_utc', 
                    direction='nearest',
                    tolerance=pd.Timedelta(minutes=5)
                )
        else:
            colors = 'blue'
            event_has_good = False
            df['SensorDepth_m'] = np.nan
        
        df['PointColor'] = colors
        poly_dfs.append(df)
        
        event_summary = {
            'Coverage_max': df['Coverage'].max(),
            'EventColor': 'green' if event_has_good else 'blue'
        }
        for m in metrics:
            if m in df.columns:
                event_summary[f'{m}_max'] = df[m].max()
        event_peaks.append(event_summary)
        
    if not poly_dfs:
        continue
        
    combined_df = pd.concat(poly_dfs, ignore_index=True)
    peaks_df = pd.DataFrame(event_peaks)
    
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8)]
    
    for metric in metrics:
        if metric not in combined_df.columns:
            print(f"Column '{metric}' not found for {poly_name}. Skipping this metric.")
            continue
            
        print(f"Generating plots for {poly_name} - {metric}...")
            
        # --- Plot 1: Evolution Plot (Original) ---
        fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
        ax1.scatter(combined_df[metric], combined_df['Coverage'], alpha=0.5, c=combined_df['PointColor'])
        
        ax1.set_xlabel(f'{metric} on Roadway (m)')
        ax1.set_ylabel('Roadway Percentage Covered (%)')
        ax1.set_title(f'Roadway Coverage vs {metric}\n({poly_name})')
        ax1.grid(True, linestyle='--', alpha=0.7)
        if len(good_timestamps) > 0:
            ax1.legend(custom_lines, ['Good Segmentation', 'Standard/Poor Segmentation'])

        fig1.tight_layout()
        fig1.savefig(os.path.join(out_dir, f'{poly_name}_{metric}_1_evolution_plot.jpg'))
        plt.close(fig1)

        # --- Plot 1b: Reversed Axes with Logistic Fit (All Data) ---
        fig1b, ax1b = plt.subplots(figsize=(10, 6), dpi=300)
        ax1b.scatter(combined_df['Coverage'], combined_df[metric], alpha=0.5, c=combined_df['PointColor'])
        
        fit_df = combined_df.dropna(subset=['Coverage', metric]).copy()
        
        if len(fit_df) > 3:
            x_plot_depth = np.linspace(fit_df[metric].min(), fit_df[metric].max(), 200)
            p0 = [fit_df['Coverage'].max(), 10, fit_df[metric].median()]
            
            try:
                # 1. Main Logistic Fit
                popt, _ = curve_fit(logistic_curve, fit_df[metric], fit_df['Coverage'], p0=p0, maxfev=5000)
                y_pred_cov = logistic_curve(x_plot_depth, *popt)
                
                # R2 calculated in Coverage Space (to match optimization goal)
                y_pred_cov_actual = logistic_curve(fit_df[metric], *popt)
                r2 = calculate_r2(fit_df['Coverage'], y_pred_cov_actual)
                
                # RMSE calculated in Depth Space (Meters)
                y_pred_depth_actual = logit_curve(fit_df['Coverage'], *popt)
                rmse = calculate_rmse(fit_df[metric], y_pred_depth_actual)
                
                ax1b.plot(y_pred_cov, x_plot_depth, color='black', linewidth=2, label=f'Logistic Fit ($R^2$={r2:.2f}, RMSE={rmse:.2f}m)')
                
                # 2. Quantiles
                q_preds = {}
                for q in [0.1, 0.5, 0.9]:
                    res = minimize(pinball_loss, popt, args=(fit_df[metric].values, fit_df['Coverage'].values, q), method='Nelder-Mead')
                    if res.success:
                        q_preds[q] = logistic_curve(x_plot_depth, *res.x)
                
                if 0.1 in q_preds and 0.9 in q_preds:
                    ax1b.fill_betweenx(x_plot_depth, q_preds[0.1], q_preds[0.9], color='red', alpha=0.15, label='80% Prediction Interval')
                if 0.5 in q_preds:
                    ax1b.plot(q_preds[0.5], x_plot_depth, linestyle='--', color='purple', linewidth=2, label='Median (0.5 Quantile)')
                    
            except Exception as e:
                print(f"Logistic fit failed for {metric} (All): {e}")

        # Updated Car Floatation Line Styling
        # ax1b.axhline(y=0.38, color=OI_ORANGE, linestyle='-', linewidth=1, alpha=0.7, label='Car Floatation (38 cm)')
        ax1b.set_xlabel('Roadway Percentage Covered (%)')
        ax1b.set_ylabel(f'{metric} (m)')
        ax1b.set_title(f'{metric} vs Roadway Coverage (Logistic Fit - All Data)\n({poly_name})')
        ax1b.grid(False)
        
        handles, labels = ax1b.get_legend_handles_labels()
        if len(good_timestamps) > 0:
            handles = custom_lines + handles
            labels = ['Good Seg.', 'Std/Poor Seg.'] + labels
        ax1b.legend(handles, labels)

        fig1b.tight_layout()
        fig1b.savefig(os.path.join(out_dir, f'{poly_name}_{metric}_1b_reversed_fit.jpg'))
        plt.close(fig1b)

        # --- Plot 1c: Reversed Axes (Good Segmentations Only) with Coastal Convergence ---
        good_df = fit_df[fit_df['PointColor'] == 'green'].copy()
        
        if not good_df.empty and len(good_df) > 3:
            fig1c, ax1c = plt.subplots(figsize=(10, 6), dpi=300)
            
            # Slate scatter for data points
            ax1c.scatter(good_df['Coverage'], good_df[metric], alpha=0.6, 
                         c=coastal_colors['concrete_slate']['light'], edgecolors='none', label='Segmentation Data')
            
            x_plot_depth_good = np.linspace(good_df[metric].min(), good_df[metric].max(), 200)
            p0_good = [good_df['Coverage'].max(), 10, good_df[metric].median()]
            
            try:
                # 1. Main Logistic Fit for Good Data
                popt_good, _ = curve_fit(logistic_curve, good_df[metric], good_df['Coverage'], p0=p0_good, maxfev=5000)
                y_pred_cov_good = logistic_curve(x_plot_depth_good, *popt_good)
                
                # R2 calculated in Coverage Space (to match optimization goal)
                y_pred_cov_actual_good = logistic_curve(good_df[metric], *popt_good)
                r2_good = calculate_r2(good_df['Coverage'], y_pred_cov_actual_good)
                
                # RMSE calculated in Depth Space (Meters)
                y_pred_depth_actual_good = logit_curve(good_df['Coverage'], *popt_good)
                rmse_good = calculate_rmse(good_df[metric], y_pred_depth_actual_good)
                
                # Solid Dark Blue Line for main fit
                ax1c.plot(y_pred_cov_good, x_plot_depth_good, color=coastal_colors['oceanic_blue']['dark'], 
                          linewidth=2.5, label=f'Logistic Fit ($R^2$={r2_good:.2f}, RMSE={rmse_good:.2f}m)')
                
                # 2. Logistic Quantile Regression for Good Data
                q_preds_good = {}
                for q in [0.1, 0.5, 0.9]:
                    res_good = minimize(pinball_loss, popt_good, args=(good_df[metric].values, good_df['Coverage'].values, q), method='Nelder-Mead')
                    if res_good.success:
                        q_preds_good[q] = logistic_curve(x_plot_depth_good, *res_good.x)
                
                # Teal Shading & Median Line
                if 0.1 in q_preds_good and 0.9 in q_preds_good:
                    ax1c.fill_betweenx(x_plot_depth_good, q_preds_good[0.1], q_preds_good[0.9], 
                                       color=coastal_colors['estuary_teal']['light'], alpha=0.35, label='80% Prediction Interval')
                if 0.5 in q_preds_good:
                    ax1c.plot(q_preds_good[0.5], x_plot_depth_good, linestyle='--', 
                              color=coastal_colors['estuary_teal']['base'], linewidth=2, label='Median (0.5 Quantile)')
                    
            except Exception as e:
                print(f"Logistic fit failed for {metric} (Good Only): {e}")

            # Warning Threshold: Car Floatation Line in Rust
            # ax1c.axhline(y=0.38, color=coastal_colors['erosion_rust']['base'], linestyle='-', linewidth=2, alpha=0.9, label='Car Floatation (38 cm)')

            ax1c.set_xlabel('Roadway Percentage Covered (%)')
            ax1c.set_ylabel(f'Maximum Depth on Roadway (m)')
            # ax1c.set_title(f'{metric} vs Roadway Coverage (Logistic Fit - Good Seg. Only)\n({poly_name})')
            ax1c.grid(False)
            
            # Clean up the legend 
            handles, labels = ax1c.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1c.legend(by_label.values(), by_label.keys())

            fig1c.tight_layout()
            fig1c.savefig(os.path.join(out_dir, f'{poly_name}_{metric}_1c_reversed_fit_good_only.jpg'))
            plt.close(fig1c)
            
        elif len(good_timestamps) > 0:
            print(f"Not enough 'good' segmentation points to generate plot 1c for {poly_name} - {metric}.")

        # --- Format Prettier Labels ---
        pretty_metric = metric.replace('MaxDepth', 'Maximum Depth') \
                              .replace('MeanDepth', 'Mean Depth') \
                              .replace('MedianDepth', 'Median Depth')

        # --- Plot 6: 1:1 Sensor vs Image Depth (Good Segmentations Only) ---
        if sensor_df_clean is not None and 'SensorDepth_m' in combined_df.columns:
            # Filter for green points, then drop any rows missing Depth, Sensor Depth, or Coverage
            plot6_df = combined_df[combined_df['PointColor'] == 'green'].dropna(subset=[metric, 'SensorDepth_m', 'Coverage']).copy()
            
            if not plot6_df.empty:
                fig6, ax6 = plt.subplots(figsize=(8, 8), dpi=300)
                
                img_depths = plot6_df[metric].values
                sens_depths = plot6_df['SensorDepth_m'].values
                cov = plot6_df['Coverage'].values
                
                # Less extreme size scaling
                sizes = 40 + (cov / 100) * 100
                
                # Dynamic size, cmocean deep colormap
                scatter = ax6.scatter(sens_depths, img_depths, c=cov, s=sizes, cmap=cmocean.cm.deep, alpha=0.8, edgecolors='k', linewidth=0.5)
                
                # Establish boundaries with a small buffer
                min_val = min(img_depths.min(), sens_depths.min()) - 0.05
                max_val = max(img_depths.max(), sens_depths.max()) + 0.05
                
                # 1:1 Line
                ax6.plot([min_val, max_val], [min_val, max_val], color=OI_BLACK, linestyle='--', linewidth=2, label='1:1 Line')
                
                # Linear Trend Line
                if len(plot6_df) > 1:
                    try:
                        m, b = np.polyfit(sens_depths, img_depths, 1)
                        trend_label = f'Linear Fit (y={m:.2f}x+{b:.2f})' if b > 0 else f'Linear Fit (y={m:.2f}x{b:.2f})'
                        ax6.plot(np.array([min_val, max_val]), m * np.array([min_val, max_val]) + b, color=OI_VERMILION, linestyle='-', linewidth=2, label=trend_label)
                    except Exception as e:
                        print(f"Linear fit failed for {metric} 1:1 plot: {e}")
                
                ax6.set_xlim([min_val, max_val])
                ax6.set_ylim([min_val, max_val])
                ax6.set_aspect('equal', adjustable='box')
                
                ax6.set_xlabel('In-Situ Sensor Depth (m)')
                ax6.set_ylabel(f'Image Derived {pretty_metric} (m)')
                ax6.grid(True, linestyle='--', alpha=0.7)
                
                # Add Colorbar for Coverage
                cbar = fig6.colorbar(scatter, ax=ax6, fraction=0.046, pad=0.04)
                cbar.set_label('Roadway Percentage Covered (%)')
                
                ax6.legend(loc='upper left')
                fig6.tight_layout()
                fig6.savefig(os.path.join(out_dir, f'{poly_name}_{metric}_6_sensor_1to1.jpg'))
                plt.close(fig6)

        # --- Plot 7: 1:1 Sensor vs Image WSE (Good Segmentations Only) ---
        if sensor_df_clean is not None and 'SensorWSE_m' in combined_df.columns:
            wse_metric = metric.replace('Depth', 'WSE')
            pretty_wse = pretty_metric.replace('Depth', 'WSE')
            
            if wse_metric in combined_df.columns:
                plot7_df = combined_df[combined_df['PointColor'] == 'green'].dropna(subset=[wse_metric, 'SensorWSE_m', 'Coverage']).copy()
                
                if not plot7_df.empty:
                    fig7, ax7 = plt.subplots(figsize=(8, 8), dpi=300)
                    
                    img_wse = plot7_df[wse_metric].values
                    # Clip the sensor WSE at a lower bound of 0.92m
                    sens_wse = plot7_df['SensorWSE_m'].clip(lower=0.92).values # CB
                    # sens_wse = plot7_df['SensorWSE_m'].clip(lower=0.49).values # DE
                    cov = plot7_df['Coverage'].values
                    
                    # Less extreme size scaling
                    sizes = 40 + (cov / 100) * 100
                    
                    # Dynamic size, cmocean deep colormap
                    scatter = ax7.scatter(sens_wse, img_wse, c=cov, s=sizes, cmap=cmocean.cm.deep, alpha=0.8, edgecolors='k', linewidth=0.5)
                    
                    min_val = min(img_wse.min(), sens_wse.min()) - 0.05
                    max_val = max(img_wse.max(), sens_wse.max()) + 0.05
                    
                    # 1:1 Line
                    ax7.plot([min_val, max_val], [min_val, max_val], color=OI_BLACK, linestyle='--', linewidth=2, label='1:1 Line')
                    
                    # Linear Trend Line
                    if len(plot7_df) > 1:
                        try:
                            m, b = np.polyfit(sens_wse, img_wse, 1)
                            trend_label = f'Linear Fit (y={m:.2f}x+{b:.2f})' if b > 0 else f'Linear Fit (y={m:.2f}x{b:.2f})'
                            ax7.plot(np.array([min_val, max_val]), m * np.array([min_val, max_val]) + b, color=OI_ORANGE, linestyle='-', linewidth=2, label=trend_label)
                        except Exception as e:
                            print(f"Linear fit failed for {wse_metric} 1:1 plot: {e}")
                    
                    ax7.set_xlim([min_val, max_val])
                    ax7.set_ylim([min_val, max_val])
                    ax7.set_aspect('equal', adjustable='box')
                    
                    ax7.set_xlabel('In-Situ Sensor WSE (m)')
                    ax7.set_ylabel(f'Image Derived {pretty_wse} (m)')
                    ax7.grid(True, linestyle='--', alpha=0.7)
                    
                    cbar = fig7.colorbar(scatter, ax=ax7, fraction=0.046, pad=0.04)
                    cbar.set_label('Roadway Percentage Covered (%)')
                    
                    ax7.legend(loc='upper left')
                    fig7.tight_layout()
                    fig7.savefig(os.path.join(out_dir, f'{poly_name}_{wse_metric}_7_sensor_1to1.jpg'))
                    plt.close(fig7)

        # --- Extract and sort Peaks for Event-Level Plots (2-5) ---
        m_max = f'{metric}_max'
        if m_max not in peaks_df.columns:
            continue
            
        peak_metrics = peaks_df[m_max].values
        peak_coverages = peaks_df['Coverage_max'].values
        peak_colors = peaks_df['EventColor'].values
        
        valid_mask = ~np.isnan(peak_metrics) & ~np.isnan(peak_coverages)
        peak_metrics = peak_metrics[valid_mask]
        peak_coverages = peak_coverages[valid_mask]
        peak_colors = peak_colors[valid_mask]
        
        sorted_indices = np.argsort(peak_metrics)
        sorted_metrics = peak_metrics[sorted_indices]
        n = len(sorted_metrics)
        
        if n == 0:
            continue

        # --- Plot 2: Summary Scatter ---
        fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
        ax2.scatter(peak_metrics, peak_coverages, c=peak_colors, alpha=0.6, edgecolors='black', s=40)
        ax2.set_xlabel(f'Peak Event {metric} (m)')
        ax2.set_ylabel('Peak Roadway Percentage Covered (%)')
        ax2.set_title(f'Peak Coverage vs Peak {metric} per Event\n({poly_name})')
        ax2.grid(True, linestyle='--', alpha=0.7)
        if len(good_timestamps) > 0:
            ax2.legend(custom_lines, ['Event Contains Good Segmentation', 'Standard/Poor Event'])
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, f'{poly_name}_{metric}_2_summary_scatter.png'))
        plt.close(fig2)

        # --- Plot 3: Cumulative Distribution Function (CDF) ---
        cdf_probs = np.arange(1, n + 1) / n
        fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)
        ax3.step(sorted_metrics, cdf_probs, where='post', color='blue', linewidth=2)
        ax3.plot(sorted_metrics, cdf_probs, 'bo', alpha=0.7)
        ax3.set_xlabel(f'Peak Event {metric} (m)')
        ax3.set_ylabel('Cumulative Probability P(X <= x)')
        ax3.set_title(f'CDF of Peak {metric}\n({poly_name})')
        ax3.grid(True, linestyle='--', alpha=0.7)
        fig3.tight_layout()
        fig3.savefig(os.path.join(out_dir, f'{poly_name}_{metric}_3_depth_cdf.png'))
        plt.close(fig3)

        # --- Plot 4: Probability Density Function (PDF) ---
        fig4, ax4 = plt.subplots(figsize=(10, 6), dpi=300)
        ax4.hist(peak_metrics, bins=5, density=True, color='skyblue', edgecolor='black', alpha=0.9)
        ax4.set_xlabel(f'Peak Event {metric} (m)')
        ax4.set_ylabel('Density')
        ax4.set_title(f'PDF of Peak {metric}\n({poly_name})')
        ax4.grid(True, linestyle='--', alpha=0.7)
        fig4.tight_layout()
        fig4.savefig(os.path.join(out_dir, f'{poly_name}_{metric}_4_depth_pdf.png'))
        plt.close(fig4)

        # --- Plot 5: Exceedance Probability ---
        exceedance_probs = np.arange(n, 0, -1) / n 
        fig5, ax5 = plt.subplots(figsize=(10, 6), dpi=300)
        ax5.step(sorted_metrics, exceedance_probs, where='post', color='red', linewidth=2)
        ax5.plot(sorted_metrics, exceedance_probs, 'ro', alpha=0.7)
        ax5.set_xlabel(f'Peak Event {metric} (m)')
        ax5.set_ylabel('Exceedance Probability P(X >= x)')
        ax5.set_title(f'Exceedance Probability of Peak {metric}\n({poly_name})')
        ax5.grid(True, linestyle='--', alpha=0.7)
        fig5.tight_layout()
        fig5.savefig(os.path.join(out_dir, f'{poly_name}_{metric}_5_depth_exceedance.png'))
        plt.close(fig5)

print(f"Processing complete. Plots saved to the '{out_dir}' directory.")
