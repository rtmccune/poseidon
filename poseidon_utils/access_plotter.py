import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, minimize

# --- Define Mathematical Models ---
def logistic_curve(x, L, k, x0):
    """
    L: Maximum value (asymptote / max coverage)
    k: Steepness of the curve
    x0: The x-value of the sigmoid's midpoint
    """
    # Use np.clip to prevent RuntimeWarnings from overflow in exp
    exponent = np.clip(-k * (x - x0), -100, 100)
    return L / (1 + np.exp(exponent))

def pinball_loss(params, x, y, q):
    """Custom loss function for quantile regression on non-linear models."""
    y_pred = logistic_curve(x, *params)
    err = y - y_pred
    return np.sum(np.maximum(q * err, (q - 1) * err))

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
    good_images = labels_df[labels_df['Segmentation_Score'] == 1]['Filename'].dropna()
    
    for fname in good_images:
        match = re.search(r'(\d{14})', fname)
        if match:
            good_timestamps.add(match.group(1))
            
    print(f"Found {len(good_timestamps)} 'good' timestamps to highlight.")
else:
    print("No valid segmentation labels provided. Using default colors.")

# Find all time-series CSVs
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

# --- 2. Data Processing and Plotting ---
for poly_name, files in polygon_files.items():
    poly_dfs = []
    event_peaks = []
    
    for f in files:
        df = pd.read_csv(f)
        
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
            csv_times = pd.to_datetime(df[time_col], errors='coerce').dt.strftime('%Y%m%d%H%M%S')
            colors = np.where(csv_times.isin(good_timestamps), 'green', 'blue')
            event_has_good = 'green' in colors
        else:
            colors = 'blue'
            event_has_good = False
        
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
    
    quantiles = [0.1, 0.5, 0.9]
    linestyles = [':', '--', ':']
    colors_q = ['red', 'purple', 'red']
    
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
            
            # Initial guess: L=Max Coverage, k=10 (arbitrary steepness), x0=Median Depth
            p0 = [fit_df['Coverage'].max(), 10, fit_df[metric].median()]
            
            try:
                # 1. Main Logistic Fit (Non-linear Least Squares)
                popt, _ = curve_fit(logistic_curve, fit_df[metric], fit_df['Coverage'], p0=p0, maxfev=5000)
                y_pred_cov = logistic_curve(x_plot_depth, *popt)
                ax1b.plot(y_pred_cov, x_plot_depth, color='black', linewidth=2, label='Logistic S-Curve Fit')
                
                # 2. Logistic Quantile Regression (Using Pinball Loss Custom Minimizer)
                for q, ls, qc in zip(quantiles, linestyles, colors_q):
                    # Use the main fit's parameters as the starting point for the quantile minimizer
                    res = minimize(pinball_loss, popt, args=(fit_df[metric].values, fit_df['Coverage'].values, q), method='Nelder-Mead')
                    if res.success:
                        y_pred_cov_q = logistic_curve(x_plot_depth, *res.x)
                        ax1b.plot(y_pred_cov_q, x_plot_depth, linestyle=ls, color=qc, linewidth=2, label=f'Quantile {q}')
            except Exception as e:
                print(f"Logistic fit failed for {metric} (All): {e}")

        ax1b.set_xlabel('Roadway Percentage Covered (%)')
        ax1b.set_ylabel(f'{metric} (m)')
        ax1b.set_title(f'{metric} vs Roadway Coverage (Logistic Fit - All Data)\n({poly_name})')
        ax1b.grid(True, linestyle='--', alpha=0.7)
        
        handles, labels = ax1b.get_legend_handles_labels()
        if len(good_timestamps) > 0:
            handles = custom_lines + handles
            labels = ['Good Seg.', 'Std/Poor Seg.'] + labels
        ax1b.legend(handles, labels)

        fig1b.tight_layout()
        fig1b.savefig(os.path.join(out_dir, f'{poly_name}_{metric}_1b_reversed_fit.jpg'))
        plt.close(fig1b)

        # --- Plot 1c: Reversed Axes with Logistic Fit (Good Segmentations Only) ---
        good_df = fit_df[fit_df['PointColor'] == 'green'].copy()
        
        if not good_df.empty and len(good_df) > 3:
            fig1c, ax1c = plt.subplots(figsize=(10, 6), dpi=300)
            
            ax1c.scatter(good_df['Coverage'], good_df[metric], alpha=0.6, c='green')
            
            x_plot_depth_good = np.linspace(good_df[metric].min(), good_df[metric].max(), 200)
            p0_good = [good_df['Coverage'].max(), 10, good_df[metric].median()]
            
            try:
                # 1. Main Logistic Fit for Good Data
                popt_good, _ = curve_fit(logistic_curve, good_df[metric], good_df['Coverage'], p0=p0_good, maxfev=5000)
                y_pred_cov_good = logistic_curve(x_plot_depth_good, *popt_good)
                ax1c.plot(y_pred_cov_good, x_plot_depth_good, color='black', linewidth=2, label='Logistic S-Curve Fit')
                
                # 2. Logistic Quantile Regression for Good Data
                for q, ls, qc in zip(quantiles, linestyles, colors_q):
                    res_good = minimize(pinball_loss, popt_good, args=(good_df[metric].values, good_df['Coverage'].values, q), method='Nelder-Mead')
                    if res_good.success:
                        y_pred_cov_q_good = logistic_curve(x_plot_depth_good, *res_good.x)
                        ax1c.plot(y_pred_cov_q_good, x_plot_depth_good, linestyle=ls, color=qc, linewidth=2, label=f'Quantile {q}')
            except Exception as e:
                print(f"Logistic fit failed for {metric} (Good Only): {e}")

            ax1c.set_xlabel('Roadway Percentage Covered (%)')
            ax1c.set_ylabel(f'{metric} (m)')
            ax1c.set_title(f'{metric} vs Roadway Coverage (Logistic Fit - Good Seg. Only)\n({poly_name})')
            ax1c.grid(True, linestyle='--', alpha=0.7)
            
            handles, labels = ax1c.get_legend_handles_labels()
            good_line = Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8)
            handles = [good_line] + handles
            labels = ['Good Seg.'] + labels
            ax1c.legend(handles, labels)

            fig1c.tight_layout()
            fig1c.savefig(os.path.join(out_dir, f'{poly_name}_{metric}_1c_reversed_fit_good_only.jpg'))
            plt.close(fig1c)
        elif len(good_timestamps) > 0:
            print(f"Not enough 'good' segmentation points to generate plot 1c for {poly_name} - {metric}.")

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
