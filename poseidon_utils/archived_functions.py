

    # def gen_grid_pdal_tin(self, resolution, orig_units='feet', point_mask_value=2):
    #     """Generates an interpolated elevation array using a PDAL TIN pipeline.
    #     [... same docstring ...]
    #     """
    #     print(f"Generating grid with PDAL (TIN method, res={resolution}m, class={point_mask_value})...")

    #     # --- Use in-memory virtual file ---
    #     tmp_filename = f"/vsimem/{uuid.uuid4().hex}.tif"
        
    #     try:
    #         # 2. Calculate explicit grid dimensions
    #         width = int(np.ceil((self.max_x_extent - self.min_x_extent) / resolution))
    #         height = int(np.ceil((self.max_y_extent - self.min_y_extent) / resolution))

    #         # 3. Define the filter range string
    #         filter_range_str = (
    #             f"Classification[{point_mask_value}:{point_mask_value}],"
    #             f"X[{self.min_x_extent}:{self.max_x_extent}],"
    #             f"Y[{self.min_y_extent}:{self.max_y_extent}]"
    #         )
            
    #         # 4. - 7. (Pipeline stages are identical to the previous in-memory attempt)
    #         pipeline_stages = [
    #             {"type": "readers.las", "filename": self.filepath}
    #         ]
    #         if orig_units == 'feet':
    #             print("Applying feet-to-meters transformation...")
    #             pipeline_stages.append({
    #                 "type": "filters.transformation",
    #                 "expression": "X = X * 0.3048; Y = Y * 0.3048; Z = Z * 0.3048"
    #             })
    #         pipeline_stages.append({
    #             "type": "filters.range",
    #             "limits": filter_range_str
    #         })
    #         pipeline_stages.append({
    #             "type": "filters.delaunay"
    #         })
    #         nodata_value = -9999.0
    #         pipeline_stages.append({
    #             "type": "filters.faceraster",
    #             "resolution": resolution,
    #             "origin_x": self.min_x_extent,
    #             "origin_y": self.min_y_extent,
    #             "width": width,
    #             "height": height,
    #             "nodata": nodata_value
    #         })
    #         pipeline_stages.append({
    #             "type": "writers.raster",
    #             "filename": tmp_filename,  # This is the /vsimem/ path
    #             "gdaldriver": "GTiff"
    #         })

    #         # 8. Create and execute the pipeline
    #         pipeline_dict = {"pipeline": pipeline_stages}
    #         pipeline_json = json.dumps(pipeline_dict)
            
    #         print("Executing PDAL TIN pipeline...")
    #         pipeline = pdal.Pipeline(pipeline_json)
            
    #         # --- Still include the multithreading optimization ---
    #         pipeline.threads = os.cpu_count() 
            
    #         pipeline.execute()
    #         print("PDAL execution complete.")

    #         # 9. Read the IN-MEMORY raster file
    #         # This is the key test: rasterio.open() tries to read from /vsimem/
    #         with rasterio.open(tmp_filename) as src:
    #             elevation_array = src.read(1).astype(float)
    #             nodata_val = src.nodata
    #             if nodata_val is not None:
    #                 elevation_array[elevation_array == nodata_val] = np.nan
            
    #         return elevation_array

    #     except Exception as e:
    #         print(f"Error during PDAL TIN pipeline processing: {e}")
    #         raise
        
    #     finally:
    #         # 10. Clean up
    #         # We are *not* cleaning up the /vsimem/ file because we
    #         # can't import rasterio.vfs.unlink().
    #         # The file will be automatically destroyed when the script ends.
    #         print(f"In-memory file {tmp_filename} will be cleared on script exit.")
    
import cmocean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# (Your existing plot_elev_grid function would be here)
# ...

def compare_grids(grid_a, grid_b, name_a="Grid A (TIN)", name_b="Grid B (orig)"):
    """
    Provides a statistical and visual comparison between two elevation grids.
    
    Assumes grids are "raster-oriented" (top-down) and handles
    NaN values for all calculations and plots.
    
    Parameters:
    ----------
    grid_a : 2D np.ndarray
        The first grid to compare (e.g., from gen_grid_pdal_tin).
    grid_b : 2D np.ndarray
        The second grid to compare (e.g., from gen_grid_pdal 'idw').
    name_a : str
        Display name for the first grid.
    name_b : str
        Display name for the second grid.
    """
    
    # 1. Calculate Difference
    # We use np.nansum to handle cases where one grid has data and the other has NaN.
    # This isn't perfect, but for subtraction, it's safer than (grid_a - grid_b)
    # which would result in NaN if *either* is NaN.
    # A more robust way is to set NaNs to 0 for subtraction, then set back.
    # Let's do (grid_a - grid_b) and let np.nan... functions handle it.
    difference = grid_a - grid_b

    # 2. Print Statistical Summary
    print("--- Grid Comparison Statistics ---")
    print(f"Stats for: {name_a}")
    print(f"  Min: {np.nanmin(grid_a):.3f} m")
    print(f"  Max: {np.nanmax(grid_a):.3f} m")
    print(f"  Mean: {np.nanmean(grid_a):.3f} m")
    print(f"  Std Dev: {np.nanstd(grid_a):.3f} m")
    print("-" * 20)
    
    print(f"Stats for: {name_b}")
    print(f"  Min: {np.nanmin(grid_b):.3f} m")
    print(f"  Max: {np.nanmax(grid_b):.3f} m")
    print(f"  Mean: {np.nanmean(grid_b):.3f} m")
    print(f"  Std Dev: {np.nanstd(grid_b):.3f} m")
    print("-" * 20)
    
    print("Stats for: Difference (A - B)")
    print(f"  Min Diff: {np.nanmin(difference):.3f} m")
    print(f"  Max Diff: {np.nanmax(difference):.3f} m")
    print(f"  Mean Diff: {np.nanmean(difference):.3f} m")
    print(f"  Std Dev (Abs): {np.nanstd(np.abs(difference)):.3f} m")
    print("------------------------------------")

    
    # 3. Plot the Difference Grid
    
    # Flip it for correct plotting with origin='lower'
    plot_diff_grid = np.flipud(difference)
    
    # Get the diverging colormap 'balance' or 'diff' from cmocean
    cmap = cmocean.cm.balance 
    
    # Find the maximum absolute difference to center the colormap at 0
    max_abs_diff = np.nanmax(np.abs(plot_diff_grid))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(
        plot_diff_grid, 
        origin="lower", 
        cmap=cmap,
        vmin=-max_abs_diff,  # Center colormap at 0
        vmax=max_abs_diff
    )
    plt.colorbar(label=f"Elevation Difference (m)\n({name_a} - {name_b})")
    plt.title("Difference Grid")
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.show()

    
    # 4. Plot Histograms 📊
    plt.figure(figsize=(12, 6))
    
    # Filter NaNs for histogram plotting
    hist_a = grid_a[~np.isnan(grid_a)]
    hist_b = grid_b[~np.isnan(grid_b)]
    
    plt.hist(hist_a, bins=100, alpha=0.7, label=name_a, density=True)
    plt.hist(hist_b, bins=100, alpha=0.7, label=name_b, density=True)
    plt.title("Elevation Distribution")
    plt.xlabel("Elevation (m)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()