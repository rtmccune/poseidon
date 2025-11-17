import os
import zarr
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from cucim.skimage.measure import label
from cucim.skimage.morphology import binary_closing
from skimage.measure import find_contours


def _log(message):
    """Helper function for timestamped logging."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


class DepthMapProcessor:
    """
    A class for identifying, analyzing, and visualizing water depth.

    This class operates on a 2D elevation grid and uses binary flood
    masks to identify pond regions. It performs labeling of connected
    ponds, extracts edge contours and elevation data, computes pond
    water depths, and visualizes edge elevation distributions.

    Attributes
    ----------
    elev_grid : numpy.ndarray
        A 2D NumPy array representing elevation values across a spatial
        grid.
    elev_grid_cp : cupy.ndarray
        A 2D CuPy array (GPU) of the elevation grid.
    plot_edges : bool
        Flag indicating whether to generate and save pond edge
        elevation plots.
    """

    def __init__(
        self,
        elevation_grid,
        plot_edges=True,
    ):
        """Initialize the object with an elevation grid.

        Parameters
        ----------
        elevation_grid : numpy.ndarray
            A 2D array representing elevation values, where each element
            corresponds to an elevation at a specific grid point.
        plot_edges : bool, optional
            If True (default), plots of pond edge elevations will be
            generated and saved.
        """
        self.elev_grid = elevation_grid
        self.elev_grid_cp = cp.array(self.elev_grid)
        self.plot_edges = plot_edges
    
    def process_single_depth_map(self, zarr_store_path, file_name):
        """Processes a Zarr store and computes depth maps.

        Parameters
        ----------
        zarr_store_path : str
            Path to the Zarr store containing the rectified image
            label array.
        file_name : str
            Name of the file being processed.

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing:
            - 'image_name' (str): The name of the processed depth map.
            - 'depth_map' (cp.ndarray): The computed depth map.
        """
        img_store = zarr.open(zarr_store_path)
        array = img_store[:]  # open image array

        gpu_label_array = cp.array(
            array
        )  # convert to cupy array for GPU processing

        # separate ponds
        labeled_data = self._label_ponds(gpu_label_array)

        contour_pixels_per_pond, contour_values_per_pond = (
            self._extract_contours(labeled_data, gpu_label_array)
        )  # extract elevations of pond edges

        # Plotting is conditional and runs before the main depth
        # calculation.
        if self.plot_edges:
            self._plot_pond_edge_elevations(
                labeled_data,
                contour_values_per_pond,
                file_name,
            )

        # Calculate all 8 depth/WSE maps (4 methods, 2 formats) in one
        # pass.
        all_maps = self._calculate_all_depths(
            labeled_data, contour_values_per_pond
        )

        # Format the output dictionary
        depth_data = []
        for map_name_suffix, map_array in all_maps.items():
            depth_data.append(
                {
                    "image_name": f"{file_name}_{map_name_suffix}",
                    "depth_map": map_array,
                }
            )

        return depth_data

    def process_depth_maps(self, 
                           labels_zarr_dir,
                           depth_map_zarr_dir,
                           pond_edge_elev_plot_dir="data/edge_histograms",):
        """Creates and saves depth maps as zarr arrays with logging.

        Given a zarr directory containing rectified labels, this method
        processes each file and saves the resulting depth maps to the
        destination directory. Logs progress and errors.

        Parameters
        ----------
        labels_zarr_dir : str
            Path to zarr directory containing rectified labels.
        depth_map_zarr_dir : str
            Path to the directory where processed depth maps will be
            saved.
        pond_edge_elev_plot_dir : str, optional
            The base directory path to save plots. Defaults to
            'data/edge_histograms'.

        Returns
        -------
        None
        """
        self.pond_edge_elev_plot_dir = pond_edge_elev_plot_dir
        
        _log("\n=== Starting Depth Map Generation ===")
        _log(f"  Source Zarr directory: {labels_zarr_dir}")
        _log(f"  Output Zarr directory: {depth_map_zarr_dir}")

        # Get a list of all files in the source directory
        try:
            all_files = os.listdir(labels_zarr_dir)
        except FileNotFoundError:
            _log(f"  ERROR: Source directory not found at {labels_zarr_dir}.")
            _log("=== Depth Map Generation Aborted ===")
            return
        except Exception as e:
            _log(f"  ERROR: Could not read directory {labels_zarr_dir}. {e}")
            _log("=== Depth Map Generation Aborted ===")
            return

        # Filter for the files we actually need to process
        files_to_process = [f for f in all_files if f.endswith("_rectified")]
        total_files = len(files_to_process)

        if total_files == 0:
            _log(
                f"  WARNING: No '_rectified' files found in {labels_zarr_dir}. "
                f"Nothing to do."
            )
            _log("=== Depth Map Generation Complete ===")
            return

        _log(f"  Found {total_files} rectified label arrays to process.")

        # Determine report interval (print ~10 updates + first/last)
        report_interval = max(1, total_files // 10)
        processed_count = 0

        for i, file_name in enumerate(files_to_process):
            # Log progress periodically
            if (
                (i + 1) % report_interval == 0
                or i == 0
                or (i + 1) == total_files
            ):
                _log(f"  Processing file {i + 1}/{total_files}: {file_name}")

            rectified_label_path = os.path.join(labels_zarr_dir, file_name)

            # Generate depth map
            try:
                depth_data = self.process_single_depth_map(
                    rectified_label_path, file_name
                )
            except Exception as e:
                _log(
                    f"  ERROR: Failed to process depth map for {file_name}. {e}"
                )
                continue  # Skip to the next file

            # Create DataFrame
            try:
                depth_maps = pd.DataFrame(depth_data)
                if depth_maps.empty:
                    _log(
                        f"  WARNING: No depth data generated for {file_name}. "
                        f"Skipping save."
                    )
                    continue
            except Exception as e:
                _log(
                    f"  ERROR: Failed to create DataFrame for {file_name}. {e}"
                )
                continue  # Skip to the next file

            # Save to Zarr
            try:
                self._save_depth_maps(depth_maps, depth_map_zarr_dir)
            except Exception as e:
                _log(
                    f"  ERROR: Failed to save depth maps for {file_name} to "
                    f"{depth_map_zarr_dir}. {e}"
                )
                continue  # Skip to the next file

            processed_count += 1

        # Print success message after all images are processed
        _log(f"  Successfully processed {processed_count}/{total_files} files.")
        _log("=== Depth Map Generation Complete ===")

    def _label_ponds(self, gpu_label_array):
        """Labels and processes pond regions efficiently using a LUT.

        This function takes a binary array indicating pond regions,
        applies morphological operations, labels connected regions,
        and removes small ponds below a size threshold.

        It performs the filtering and relabeling in a single pass
        using a lookup table (LUT) to avoid a second, expensive
        call to `label()`.

        Parameters
        ----------
        gpu_label_array : cp.ndarray
            A CuPy array where pond regions are marked as 1.

        Returns
        -------
        cp.ndarray
            A labeled CuPy array where each connected pond has a unique,
            contiguous integer ID.
        """
        labels_squeezed = gpu_label_array.squeeze()
        water_mask = labels_squeezed == 1
        masked_labels = water_mask.astype(cp.uint8)

        closed_data = binary_closing(
            masked_labels, footprint=cp.ones((3, 3), dtype=cp.uint8)
        )

        # First and only label call
        labeled_data, num_features = label(closed_data, return_num=True)

        # Handle edge case where no ponds are found
        if num_features == 0:
            return labeled_data

        # Use bincount (faster than unique)
        # Get the size (pixel count) for each label ID.
        # Note: labeled_data.ravel() is required as bincount needs a
        # 1D array.
        # We use minlength to ensure the counts array is large enough.
        counts = cp.bincount(labeled_data.ravel(), minlength=num_features + 1)

        # Identify large ponds
        min_size = 55
        # Find the *original* label IDs that we want to keep.
        # We use cp.arange to get the label IDs and exclude label 0
        # (background).
        all_labels = cp.arange(num_features + 1, dtype=cp.int32)
        large_ponds = all_labels[(counts >= min_size) & (all_labels != 0)]

        # Create the Lookup Table (LUT)
        # This array will map old labels to new, contiguous labels.
        # Initialize a mapper array full of zeros.
        # The size is (num_features + 1) to map all possible old labels.
        lut = cp.zeros(num_features + 1, dtype=cp.int32)

        # Create new, contiguous labels [1, 2, 3, ...]
        new_labels = cp.arange(1, len(large_ponds) + 1, dtype=cp.int32)

        # Populate the LUT: map old, large-pond IDs to the new labels.
        # e.g., if large_ponds was [2, 5, 8], this maps:
        # lut[2] = 1, lut[5] = 2, lut[8] = 3
        # All other indices (like small ponds) will remain 0.
        lut[large_ponds] = new_labels

        # Apply the LUT
        # This is a very fast GPU operation.
        # Every pixel in labeled_data has its value "looked up" in the
        # lut.
        # This one step both removes small ponds and relabels
        # contiguously.
        relabeled_data = lut[labeled_data]

        return relabeled_data

    def _extract_contours(self, labeled_data, gpu_label_array):
        """Extracts contour pixels and their elevation values for each
        pond.

        This function identifies contours around labeled pond regions,
        extracts their pixel coordinates, and retrieves
        corresponding label values.

        Parameters
        ----------
        labeled_data : cp.ndarray
            A CuPy array where each pond is labeled with a unique
            integer ID.
        gpu_label_array : cp.ndarray
            A CuPy array representing the original labeled dataset,
            where 1 indicates flooded regions.

        Returns
        -------
        contour_pixels_per_pond : dict[int, np.ndarray]
            A dictionary mapping each pond ID to an array of its contour
            pixel coordinates (Nx2).
        contour_values_per_pond : dict[int, np.ndarray]
            A dictionary mapping each pond ID to an array of elevation
            values at contour pixels.
        """
        # --- GPU Operations ---
        # Get unique pond IDs
        unique_ponds = cp.unique(labeled_data)
        unique_ponds = unique_ponds[
            unique_ponds != 0
        ]  # Exclude background label

        # Create the elevation-water map
        arr = cp.where(gpu_label_array.squeeze() == 1, self.elev_grid_cp, 0)

        # --- GPU-to-CPU Transfers ---
        # Transfer all required data to CPU *once*.
        # These are the main unavoidable bottlenecks, as find_contours
        # runs on the CPU.
        unique_ponds_np = cp.asnumpy(unique_ponds)
        labeled_data_np = cp.asnumpy(labeled_data)
        arr_np = cp.asnumpy(arr)

        # --- CPU Operations ---
        pond_contours = {}

        # Loop 1: Find contours for each pond
        for pond_id in unique_ponds_np:
            # Create a simple binary mask for this pond
            pond_mask = labeled_data_np == pond_id

            # Find contours directly on the 0/1 mask.
            contours = find_contours(pond_mask, level=0.5)
            pond_contours[pond_id] = contours

        # Loop 2: Process contours and look up elevation values
        contour_pixels_per_pond = {}
        contour_values_per_pond = {}

        for pond_id, contours in pond_contours.items():
            if not contours:
                continue  # Skip empty contours

            contour_pixels = np.vstack(
                [np.round(contour).astype(int) for contour in contours]
            )

            # Ensure indices are within bounds
            valid_mask = (
                (0 <= contour_pixels[:, 1])
                & (contour_pixels[:, 1] < arr_np.shape[1])
                & (0 <= contour_pixels[:, 0])
                & (contour_pixels[:, 0] < arr_np.shape[0])
            )

            contour_pixels = contour_pixels[valid_mask]

            # Look up elevation values from the complete map.
            contour_values = arr_np[contour_pixels[:, 0], contour_pixels[:, 1]]

            contour_pixels_per_pond[pond_id] = contour_pixels
            contour_values_per_pond[pond_id] = contour_values

        return contour_pixels_per_pond, contour_values_per_pond

    def _calculate_all_depths(self, labeled_data, contour_values_per_pond):
        """
        Calculates all 8 depth/wse maps (4 methods, 2 formats) in one
        pass.

        This method loops through each pond ID *once*, computes all
        required statistics (mean, median, 90/95th percentile) for its
        contours, generates the 8 map fragments (4 stats * 2 formats)
        for that pond, and collects them. Finally, it combines all
        fragments into 8 complete maps.

        Parameters
        ----------
        labeled_data : cp.ndarray
            A CuPy array where each pond is labeled with a unique
            integer ID.
        contour_values_per_pond : dict[int, np.ndarray]
            A dictionary mapping each pond ID to an array of elevation
            values at contour pixels.

        Returns
        -------
        dict[str, cp.ndarray]
            A dictionary where keys are map name suffixes (e.g.,
            'wse_map_mean', 'depth_map_95_perc') and values are the
            corresponding combined CuPy depth maps.
        """
        # This will hold the map fragments, e.g.:
        # { 'wse_map_mean': {1: pond_1_map, 2: pond_2_map}, ... }
        pond_depth_collectors = {
            "wse_map_mean": {},
            "depth_map_mean": {},
            "wse_map_95_perc": {},
            "depth_map_95_perc": {},
            "wse_map_90_perc": {},
            "depth_map_90_perc": {},
            "wse_map_median": {},
            "depth_map_median": {},
        }

        # --- GPU Data Prep ---
        try:
            gpu_contours = {
                k: cp.array(v) for k, v in contour_values_per_pond.items()
            }
        except Exception as e:
            print(f"Error converting contour values to CuPy: {e}")
            return {}  # Return empty on error

        unique_pond_ids = cp.unique(labeled_data)

        # Handle no ponds case
        if len(unique_pond_ids) == 1 and unique_pond_ids[0] == 0:
            final_maps = {}
            empty_map = cp.full_like(self.elev_grid_cp, cp.nan)
            for map_name in pond_depth_collectors.keys():
                final_maps[map_name] = empty_map
            return final_maps

        unique_pond_ids_cpu = unique_pond_ids.get()

        # --- SINGLE LOOP OVER PONDS ---
        for pond_id_cpu in unique_pond_ids_cpu:
            if pond_id_cpu == 0:  # Skip background
                continue

            if pond_id_cpu not in gpu_contours:
                print(
                    f"Warning: No contours found for pond_id {pond_id_cpu}. "
                    f"Skipping depth calculation for this pond."
                )
                continue

            pond_mask = labeled_data == pond_id_cpu
            masked_elevations = cp.where(pond_mask, self.elev_grid_cp, cp.nan)
            contour_vals_gpu = gpu_contours[pond_id_cpu]

            # --- CALCULATE ALL 4 STATS AT ONCE ---
            stats = {
                "mean": cp.nanmean(contour_vals_gpu),
                "median": cp.nanmedian(contour_vals_gpu),
                "95_perc": cp.percentile(contour_vals_gpu, 95),
                "90_perc": cp.percentile(contour_vals_gpu, 90),
            }

            # --- GENERATE ALL 8 MAP FRAGMENTS FOR THIS POND ---
            for method, max_elevation in stats.items():
                # WSE map
                wse_map = cp.where(pond_mask, max_elevation, cp.nan)
                pond_depth_collectors[f"wse_map_{method}"][
                    pond_id_cpu
                ] = wse_map

                # Depth map
                depth_map = masked_elevations - max_elevation
                depth_map = cp.minimum(depth_map, 0)
                depth_map = cp.abs(depth_map)
                pond_depth_collectors[f"depth_map_{method}"][
                    pond_id_cpu
                ] = depth_map

        # --- COMBINE ALL 8 MAPS ---
        final_maps = {}
        for map_name, pond_depths_dict in pond_depth_collectors.items():
            final_maps[map_name] = self._combine_depth_maps(pond_depths_dict)

        return final_maps

    def _combine_depth_maps(self, pond_depths):
        """Combines multiple pond depth maps into a single depth map.

        Parameters
        ----------
        pond_depths : dict[int, cp.ndarray]
            A dictionary where each key is a pond ID and the value is
            the depth map for that pond.

        Returns
        -------
        cp.ndarray
            A combined depth map. Overlapping pond values are handled
            using `cp.fmax` (the max value wins), and NaN values are
            preserved in the background.
        """
        # Get a list of the actual map arrays from the dictionary
        maps_to_combine = list(pond_depths.values())

        # Handle the edge case where there are no ponds to combine
        if not maps_to_combine:
            # Return a grid full of NaNs with the same shape as the
            # elevation grid
            return cp.full_like(self.elev_grid_cp, cp.nan)

        # Start with the first map in the list as the base
        combined_map = maps_to_combine[0]

        # Iterate through the rest of the maps in the list
        for i in range(1, len(maps_to_combine)):
            current_map = maps_to_combine[i]

            # Use cp.fmax to combine. It handles NaNs nicely:
            # - fmax(val, NaN) -> val
            # - fmax(NaN, val) -> val
            # - fmax(val1, val2) -> the larger value
            combined_map = cp.fmax(combined_map, current_map)

        return combined_map

    def _save_depth_maps(self, depth_maps_dataframe, depth_map_zarr_dir):
        """Saves depth maps from a DataFrame into a Zarr store.

        Parameters
        ----------
        depth_maps_dataframe : pd.DataFrame
            A DataFrame containing depth maps, where each row includes
            an 'image_name' and a 'depth_map'.
        depth_map_zarr_dir : str
            Path to the directory where the depth maps will be stored
            in a Zarr group.

        Returns
        -------
        None
        """
        for _, row in depth_maps_dataframe.iterrows():
            store = zarr.open_group(depth_map_zarr_dir, mode="a")

            image_name = row["image_name"]

            depth_map = row["depth_map"]

            store[image_name] = depth_map.get()

    def _plot_pond_edge_elevations(
        self,
        labeled_data,
        contour_values_per_pond,
        file_name,
    ):
        """
        Orchestrates the plotting of edge elevation histograms.

        This method prepares the data and directories, then calls
        helper methods to plot histograms for individual ponds and
        for all ponds combined.

        Parameters
        ----------
        labeled_data : cp.ndarray
            A CuPy array containing labeled pond segmentation data.
        contour_values_per_pond : dict[int, np.ndarray]
            A dictionary mapping pond IDs to NumPy arrays of
            edge elevation values.
        file_name : str
            Base filename used for saving the plots.

        Returns
        -------
        None
        """
        # Setup directories for all plots and individual pond plots
        all_pond_dir = os.path.join(self.pond_edge_elev_plot_dir, "all_ponds")
        ind_pond_dir = os.path.join(self.pond_edge_elev_plot_dir, "ind_ponds")
        os.makedirs(all_pond_dir, exist_ok=True)
        os.makedirs(ind_pond_dir, exist_ok=True)

        bins = np.linspace(0, 2, 51)

        # Extract and validate data
        # We create a new dict of just the valid NumPy arrays
        edge_elevs_by_pond = {}
        unique_pond_ids = cp.unique(labeled_data).get()  # Get IDs on CPU

        for pond_id in unique_pond_ids:
            if pond_id == 0:  # Skip background
                continue

            pond_id_key = pond_id.item()
            # Only add ponds that have corresponding contour data
            if pond_id_key in contour_values_per_pond:
                edge_elevs_by_pond[pond_id_key] = contour_values_per_pond[
                    pond_id_key
                ]
            else:
                print(
                    f"Warning: No contours for pond_id {pond_id_key}. "
                    f"Skipping plot."
                )

        # Check if there is anything to plot
        if not edge_elevs_by_pond:
            print(
                f"No pond edge elevations found for {file_name}. "
                f"Skipping all histograms."
            )
            return

        # Call helper to plot all individual ponds
        self._plot_individual_histograms(
            edge_elevs_by_pond, bins, ind_pond_dir, file_name
        )

        # Call helper to plot the single combined histogram
        all_edge_elevs = np.concatenate(list(edge_elevs_by_pond.values()))
        self._plot_combined_histogram(
            all_edge_elevs, bins, all_pond_dir, file_name
        )

        return None

    def _plot_individual_histograms(
        self, edge_elevs_by_pond, bins, output_dir, file_name_base
    ):
        """
        Plots and saves one histogram for each pond.

        Parameters
        ----------
        edge_elevs_by_pond : dict[int, np.ndarray]
            A dictionary mapping pond IDs to their edge elevation arrays.
        bins : np.ndarray
            The bins to use for the histogram.
        output_dir : str
            The directory path to save the plots.
        file_name_base : str
            The base name for the output plot files.
        """
        # Loop through each pond and call the generic plotting helper
        for pond_id, elevs_np in edge_elevs_by_pond.items():
            title = f"Elevation Histogram for Pond {pond_id}"
            save_path = f"{output_dir}/{file_name_base}_Pond_{pond_id}"

            self._plot_histogram_helper(
                elevs_np, bins, title, save_path, color="lightblue"
            )

    def _plot_combined_histogram(
        self, all_edge_elevs, bins, output_dir, file_name_base
    ):
        """
        Plots and saves a single combined histogram for all ponds.

        Parameters
        ----------
        all_edge_elevs : np.ndarray
            A 1D array of all edge elevations from all ponds.
        bins : np.ndarray
            The bins to use for the histogram.
        output_dir : str
            The directory path to save the plot.
        file_name_base : str
            The base name for the output plot file.
        """
        # Create the title and save path for the combined plot
        title = "Elevation Histogram - All Ponds Combined"
        save_path = f"{output_dir}/{file_name_base}_All_Ponds_Histogram"

        # Call the generic plotting helper with the combined data
        self._plot_histogram_helper(
            all_edge_elevs, bins, title, save_path, color="lightcoral"
        )

    def _plot_histogram_helper(self, data, bins, title, save_path, color):
        """
        Generic helper to create and save one histogram plot.

        This method calculates statistics (mean, median, 95th percentile)
        and plots the data with statistic lines.

        Parameters
        ----------
        data : np.ndarray
            The 1D array of data to plot.
        bins : np.ndarray
            The bins to use for the histogram.
        title : str
            The title for the plot.
        save_path : str
            The full file path to save the resulting plot.
        color : str
            The color for the histogram bars.
        """
        # Ensure there is data to prevent errors with empty arrays
        if data.size == 0:
            print(f"Warning: No data to plot for '{title}'. Skipping.")
            return

        # Compute all required statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        percentile_95 = np.percentile(data, 95)

        # Initialize the plot figure
        plt.figure(figsize=(8, 6))

        # Plot the main histogram
        plt.hist(
            data,
            bins=bins,
            color=color,
            edgecolor="black",
            alpha=0.7,
        )

        # Overlay vertical lines for each statistic
        plt.axvline(
            mean_val,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean_val:.2f}",
        )
        plt.axvline(
            median_val,
            color="green",
            linestyle="solid",
            linewidth=2,
            label=f"Median: {median_val:.2f}",
        )
        plt.axvline(
            percentile_95,
            color="purple",
            linestyle="dashdot",
            linewidth=2,
            label=f"95th %ile: {percentile_95:.2f}",
        )

        # Set plot limits
        plt.xlim(0, 2)

        # Add labels, title, and legend
        plt.xlabel("Elevation")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save the plot to the specified file path and close the figure
        # to free up memory
        plt.savefig(save_path)
        plt.close()
