import os
import zarr
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import label
from cupyx.scipy.ndimage import binary_closing
from skimage.measure import find_contours


class DepthMapper:
    """
    A class for identifying, analyzing, and visualizing water depth in ponded regions based on elevation data.

    This class operates on a 2D elevation grid and uses binary flood masks to identify pond regions. It performs
    labeling of connected ponds, extracts edge contours and elevation data, computes pond water depths, and
    visualizes edge elevation distributions.

    Attributes:
        elev_grid (numpy.ndarray): A 2D NumPy array representing elevation values across a spatial grid.

    Methods:
        label_ponds(gpu_label_array):
            Labels and cleans connected pond regions in a binary CuPy array.

        extract_contours(labeled_data, gpu_label_array):
            Extracts contour coordinates and their elevation values for each labeled pond.

        calculate_depths(labeled_data, contour_values_per_pond):
            Calculates pond depth maps using the 95th percentile of contour elevations.

        plot_pond_edge_elevations(labeled_data, contour_values_per_pond, pond_edge_elev_plot_dir, file_name):
            Plots and saves elevation histograms for individual ponds and all ponds combined.
    """

    def __init__(self, elevation_grid):
        """Initialize the object with with an elevation grid.

        Args:
            elevation_grid (numpy.ndarray): A 2D array representing elevation values,
            where each element corresponds to an elevation at a specific grid point.
        """

        self.elev_grid = elevation_grid
        # self.pond_edge_elev_plot_dir = pond_edge_elev_plot_dir

    def label_ponds(self, gpu_label_array):
        """Labels and processes pond regions in a binary mask.

        This function takes a binary array indicating pond regions, applies morphological operations
        to clean up the mask, labels connected pond regions, and removes small ponds below a size threshold.

        Args:
            gpu_label_array (cp.ndarray): A CuPy array where pond regions are marked as 1.

        Returns:
            cp.ndarray: A labeled CuPy array where each connected pond has a unique integer ID.
        """
        labels_squeezed = gpu_label_array.squeeze()
        mask = labels_squeezed == 1  # Boolean mask

        # Create binary mask directly as uint8 (avoids redundant cp.where)
        masked_labels = mask.astype(cp.uint8)

        # Apply binary closing (morphological operation)
        closed_data = binary_closing(
            masked_labels, structure=cp.ones((3, 3), dtype=cp.uint8)
        )

        # Label connected components
        labeled_data, num_features = label(closed_data)

        # Remove small ponds
        min_size = 55
        unique, counts = cp.unique(labeled_data, return_counts=True)
        small_ponds = unique[counts < min_size]

        # In-place update (avoid unnecessary memory allocation)
        mask_small_ponds = cp.isin(labeled_data, small_ponds)
        labeled_data[mask_small_ponds] = 0

        # Relabel remaining ponds
        labeled_data, num_features = label(labeled_data)

        return labeled_data

    def extract_contours(self, labeled_data, gpu_label_array):
        """Extracts contour pixels and their elevation values for each pond.

        This function identifies contours around labeled pond regions, extracts their pixel coordinates,
        and retrieves corresponding label values.

        Args:
            labeled_data (cp.ndarray): A CuPy array where each pond is labeled with a unique integer ID.
            gpu_label_array (cp.ndarray): A CuPy array representing the original labeled dataset,
                                        where 1 indicates flooded regions.

        Returns:
            tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
                - A dictionary mapping each pond ID to an array of its contour pixel coordinates (Nx2).
                - A dictionary mapping each pond ID to an array of elevation values at contour pixels.`
        """
        unique_ponds = cp.unique(labeled_data)
        unique_ponds = unique_ponds[unique_ponds != 0]  # Exclude background label

        pond_contours = {}
        arr = cp.where(gpu_label_array.squeeze() == 1, self.elev_grid, 0)

        # Convert to NumPy only once per loop iteration
        labeled_data_np = cp.asnumpy(labeled_data)
        arr_np = cp.asnumpy(arr)

        for pond_id in unique_ponds:
            pond_id_int = int(pond_id.get())  # Convert cupy scalar to Python int
            pond_mask = labeled_data_np == pond_id_int
            pond_arr = np.where(pond_mask, arr_np, 0)

            contours = find_contours(pond_arr, level=0.5)
            pond_contours[pond_id_int] = contours

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
            contour_values = arr_np[contour_pixels[:, 0], contour_pixels[:, 1]]

            contour_pixels_per_pond[pond_id] = contour_pixels
            contour_values_per_pond[pond_id] = contour_values

        return contour_pixels_per_pond, contour_values_per_pond

    def calculate_depths(self, labeled_data, contour_values_per_pond, method, output_format):
        """Calculates the depth of each pond based on elevation differences.

        Args:
            labeled_data (cupy.ndarray): A 2D array where each pond is assigned a unique label.
            contour_values_per_pond (dict of int -> cupy.ndarray): A dictionary mapping
                pond IDs to an array of elevation values along the pond contour.

        Returns:
            dict of int -> cupy.ndarray: A dictionary where each key is a pond ID, and
                the value is a 2D CuPy array representing the depth map of that pond.
                Depth values are computed as the difference between the pond's elevation
                and the 95th percentile of its contour elevations. Areas above this
                threshold are set to zero, ensuring only ponding regions remain.
        """
        pond_depths = {}  # initialize pond depth dictionary

        unique_pond_ids = cp.unique(labeled_data)  # array of unique pond labels

        if len(unique_pond_ids) == 1 and unique_pond_ids[0] == 0:  # If no ponds present
            pond_depths[1] = cp.full_like(
                self.elev_grid, cp.nan
            )  # Fill pond depths with NaN (background, no water)
            return pond_depths

        for pond_id in unique_pond_ids:
            if pond_id == 0:  # Skip background
                continue
            
            if pond_id.item() not in contour_values_per_pond:
                print(f"Warning: No contours found for pond_id {pond_id.item()}. Skipping depth calculation for this pond.")
                continue

            pond_mask = labeled_data == pond_id  # mask to current pond only
            masked_elevations = cp.where(
                pond_mask, self.elev_grid, cp.nan
            )  # replace any other values not belonging to this pond with NaN

            
            if method == 'mean':
                max_elevation = cp.nanmean(
                    cp.array(contour_values_per_pond[pond_id.item()])
                )  # calculate mean of edges
            
            elif method == 'median':
                max_elevation = cp.nanmedian(
                    cp.array(contour_values_per_pond[pond_id.item()])
                )  # calculate median of edges
            
            elif method == '95_perc':
                max_elevation = cp.percentile(
                    cp.array(contour_values_per_pond[pond_id.item()]), 95
                )  # calculate 95th percentile of edges
                
            elif method == '90_perc':
                max_elevation = cp.percentile(
                    cp.array(contour_values_per_pond[pond_id.item()]), 90
                )  # calculate 90th percentile of edges
            
            if output_format == 'wse':
                # Where pond_mask is True, use max_elevation. Everywhere else, use NaN.
                # This correctly preserves the NaN background.
                depth_map = cp.where(pond_mask, max_elevation, cp.nan)
                
            elif output_format == 'depth':
                depth_map = masked_elevations - max_elevation  # calculate depth across pond
                depth_map[depth_map > 0] = (
                    0  # set depths greater than 0 to 0 to handle edges above 95th percentile
                )
                depth_map = cp.abs(depth_map)  # take the absolute value of the depths

            pond_depths[pond_id.item()] = depth_map

        return pond_depths

    def plot_pond_edge_elevations(
        self, labeled_data, contour_values_per_pond, pond_edge_elev_plot_dir, file_name
    ):
        """
        Plots and saves histograms of edge elevations for individual ponds and all ponds combined.

        Args:
            labeled_data : cp.ndarray
                A CuPy array containing labeled pond segmentation data. Each unique non-zero value represents a different pond.

            contour_values_per_pond : dict
                A dictionary mapping pond IDs (integers) to CuPy arrays of edge elevation values for each pond.

            pond_edge_elev_plot_dir : str
                Path to the directory where the output plots will be saved. Subdirectories for individual and combined pond plots will be created.

            file_name : str
                Base filename used for saving the plots.

        Returns:
            None
                This method saves plots to disk and does not return any values.

        Notes:
        - Individual pond elevation histograms include vertical lines for the mean, median, and 95th percentile values.
        - If no ponds are found (i.e., `edge_elevs` is empty), the method exits early without creating a combined plot.
        """
        all_pond_dir = os.path.join(pond_edge_elev_plot_dir, "all_ponds")
        ind_pond_dir = os.path.join(pond_edge_elev_plot_dir, "ind_ponds")
        os.makedirs(all_pond_dir, exist_ok=True)
        os.makedirs(ind_pond_dir, exist_ok=True)

        edge_elevs = []  # initialize pond depth dictionary

        unique_pond_ids = cp.unique(labeled_data)  # array of unique pond labels

        bins = np.linspace(0, 2, 51)

        for pond_id in unique_pond_ids:
            if pond_id == 0:  # Skip background
                continue
            
            if pond_id.item() not in contour_values_per_pond:
                print(f"Warning: No contours found for pond_id {pond_id.item()}. Skipping plotting for this pond.")
                continue

            pond_edge_elevs = cp.array(contour_values_per_pond[pond_id.item()])

            # edge_elevs[pond_id.item()] = pond_edge_elevs
            edge_elevs.append(pond_edge_elevs.get())

            # Compute statistics
            mean_val = np.mean(pond_edge_elevs)
            median_val = np.median(pond_edge_elevs)
            percentile_95 = np.percentile(pond_edge_elevs, 95)

            # Plot histogram
            plt.figure(figsize=(8, 6))
            plt.hist(
                pond_edge_elevs.get(),
                bins=bins,
                color="lightblue",
                edgecolor="black",
                alpha=0.7,
            )

            # Overlay vertical lines for statistics
            plt.axvline(
                mean_val.get(),
                color="red",
                linestyle="dashed",
                linewidth=2,
                label=f"Mean: {mean_val:.2f}",
            )
            plt.axvline(
                median_val.get(),
                color="green",
                linestyle="solid",
                linewidth=2,
                label=f"Median: {median_val:.2f}",
            )
            plt.axvline(
                percentile_95.get(),
                color="purple",
                linestyle="dashdot",
                linewidth=2,
                label=f"95th %ile: {percentile_95:.2f}",
            )

            plt.xlim(0, 2)

            # Labels and legend
            plt.xlabel("Elevation")
            plt.ylabel("Frequency")
            plt.title(f"Elevation Histogram for Pond {pond_id}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)

            # Show plot
            plt.savefig(f"{ind_pond_dir}/{file_name}_Pond_{pond_id}")
            plt.close()

        if not edge_elevs:
            print(
                f"No pond edge elevations found for {file_name}. Skipping combined histogram."
            )
            return  # Exit the function early

        all_edge_elevs = np.concatenate(edge_elevs)

        # Plot histogram for all ponds combined
        plt.figure(figsize=(8, 6))
        plt.hist(
            all_edge_elevs, bins=bins, color="lightcoral", edgecolor="black", alpha=0.7
        )

        # Compute global statistics
        mean_all = np.mean(all_edge_elevs)
        median_all = np.median(all_edge_elevs)
        percentile_95_all = np.percentile(all_edge_elevs, 95)

        # Overlay vertical lines for statistics
        plt.axvline(
            mean_all,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean_all:.2f}",
        )
        plt.axvline(
            median_all,
            color="green",
            linestyle="solid",
            linewidth=2,
            label=f"Median: {median_all:.2f}",
        )
        plt.axvline(
            percentile_95_all,
            color="purple",
            linestyle="dashdot",
            linewidth=2,
            label=f"95th %ile: {percentile_95_all:.2f}",
        )

        plt.xlim(0, 2)

        # Labels and legend
        plt.xlabel("Elevation")
        plt.ylabel("Frequency")
        plt.title("Elevation Histogram - All Ponds Combined")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # Save the combined histogram
        plt.savefig(f"{all_pond_dir}/{file_name}_All_Ponds_Histogram")
        plt.close()

        return None

    # def combine_depth_maps(self, pond_depths, output_format):
    #     """Combines multiple pond maps into a single map.

    #     The combination logic depends on the output format:
    #     - 'wse': Fills NaN values. The first pond to occupy a pixel sets the WSE.
    #     - 'depth': Sums values. Overlapping pond depths are added together.

    #     Args:
    #         pond_depths (dict of int -> cupy.ndarray): A dictionary of maps where each
    #             key is a pond ID and the value is the map for that pond.
    #         output_format (str): The format of the maps, either 'wse' or 'depth'.

    #     Returns:
    #         cupy.ndarray: A combined map.
    #     """
    #     # Get a list of the map arrays from the dictionary
    #     maps_to_combine = list(pond_depths.values())

    #     if not maps_to_combine:
    #         # Return an empty or NaN grid if there are no ponds
    #         return cp.full_like(self.elev_grid, cp.nan)

    #     # Start with the first map in the list
    #     combined_map = maps_to_combine[0]

    #     # Iterate through the rest of the maps
    #     for i in range(1, len(maps_to_combine)):
    #         current_map = maps_to_combine[i]

    #         if output_format == 'wse':
    #             # For WSE, fill in the blanks. Where the combined map is NaN,
    #             # use the value from the current map. Otherwise, keep the existing value.
    #             combined_map = cp.where(
    #                 cp.isnan(combined_map),
    #                 current_map,
    #                 combined_map
    #             )
    #         elif output_format == 'depth':
    #             # For depth, add the values, treating NaNs as 0 for the sum.
    #             combined_map = cp.nansum(cp.array([combined_map, current_map]), axis=0)
    #             # After summing, where both were originally NaN, it will be 0.
    #             # We need to restore NaN for the background.
    #             combined_map = cp.where(cp.isnan(maps_to_combine[0]) & cp.isnan(current_map), cp.nan, combined_map)


    #     return combined_map
    
    #def combine_depth_maps(self, pond_depths):
    #    """Combines multiple pond depth maps into a single depth map.
#
     #   Args:
      #      pond_depths (list of cupy.ndarray): A list of depth maps where each element
 #               represents the depth values for a specific pond.

  #      Returns:
   #         cupy.ndarray: A combined depth map where overlapping pond depths are summed,
    #        and NaN values are handled appropriately.
     #   """
      #  combined_depth_map = pond_depths[1]
#
 #       for i in range(2, len(pond_depths) + 1):
  #          combined_depth_map = cp.where(
   #             cp.isnan(combined_depth_map),
    #            pond_depths[i],  # If combined_depth_map has NaN, use pond_depths[i]
     #           cp.where(
      #              cp.isnan(pond_depths[i]),
       #             combined_depth_map,  # If pond_depths[i] has NaN, keep combined_depth_map
        #            combined_depth_map + pond_depths[i],
         #       ),
          #  )  # Otherwise, sum the values

        #return combined_depth_map
    def combine_depth_maps(self, pond_depths):
        """Combines multiple pond depth maps into a single depth map.

        Args:
            pond_depths (dict of int -> cupy.ndarray): A dictionary where each key is a
                pond ID and the value is the depth map for that pond.

        Returns:
            cupy.ndarray: A combined depth map where overlapping pond depths are handled,
            and NaN values are preserved in the background.
        """
        # Get a list of the actual map arrays from the dictionary
        maps_to_combine = list(pond_depths.values())

        # Handle the edge case where there are no ponds to combine
        if not maps_to_combine:
           # Return a grid full of NaNs with the same shape as the elevation grid
            return cp.full_like(self.elev_grid, cp.nan)

        # Start with the first map in the list as the base
        combined_map = maps_to_combine[0]

        # Iterate through the rest of the maps in the list
        for i in range(1, len(maps_to_combine)):
            current_map = maps_to_combine[i]

            # Use cp.fmax to combine. It handles NaNs nicely:
            # - fmax(val, NaN) -> val
            # - fmax(NaN, val) -> val
            # - fmax(val1, val2) -> the larger value
            # This effectively overlays the ponds without needing complex 'where' clauses.
            # For 'depth', this means the deeper value wins at overlaps.
            # For 'wse', this also works as all pond WSEs should be the same.
            # If you need to SUM the depths at overlaps, the nansum approach is better.
            combined_map = cp.fmax(combined_map, current_map)

        return combined_map

    def process_file(self, zarr_store_path, file_name, pond_edge_elev_plot_dir):
        """Processes a Zarr store containing rectified labeled image data and computes depth maps.

        Args:
            zarr_store_path (str): Path to the Zarr store containing the rectified image label array.
            file_name (str): Name of the file being processed.

        Returns:
            list[dict]: A list containing a dictionary with:
                - 'image_name' (str): The name of the processed depth map.
                - 'depth_map' (cupy.ndarray or similar): The computed depth map.
        """
        depth_data = []  # intialize depth data list
        img_store = zarr.open(zarr_store_path)
        array = img_store[:]  # open image array

        gpu_label_array = cp.array(array)  # convert to cupy array for GPU processing
        labeled_data = self.label_ponds(gpu_label_array)  # separate ponds

        contour_pixels_per_pond, contour_values_per_pond = self.extract_contours(
            labeled_data, gpu_label_array
        )  # extract elevations of pond edges

        self.plot_pond_edge_elevations(
            labeled_data, contour_values_per_pond, pond_edge_elev_plot_dir, file_name
        )
        
        methods = ['mean', '95_perc', '90_perc', 'median']
        output_formats = ['wse', 'depth']
        
        for method in methods:
            for output_format in output_formats:
                pond_depths = self.calculate_depths(
                    labeled_data, contour_values_per_pond, method, output_format
                )  # calculate depths based on extracted edge elevations

                combined_depth_map = self.combine_depth_maps(
                    pond_depths
                )  # combine separate ponds depth maps into one map
                depth_data.append(
                    {
                        "image_name": f"{file_name}_{output_format}_map_{method}",
                        "depth_map": combined_depth_map,
                    }
                )
        return depth_data

    def process_depth_maps(
        self, labels_zarr_dir, depth_map_zarr_dir, pond_edge_elev_plot_dir
    ):
        """Creates and saves depth maps as zarr arrays at the provided destination,
            given the zarr direcotry containing rectified labels.

        Args:
            labels_zarr_dir (str): Path to zarr directory containing rectified labels.
            depth_map_zarr_dir (str): Path to the directory where processed depth maps will be saved.
        """
        for file_name in os.listdir(labels_zarr_dir):  # for each rectified label array
            if file_name.endswith("_rectified"):  # confirm that it has been rectified
                rectified_label_array = os.path.join(
                    labels_zarr_dir, file_name
                )  # combine file path
                depth_data = self.process_file(
                    rectified_label_array, file_name, pond_edge_elev_plot_dir
                )  # generate depth map

                depth_maps = pd.DataFrame(
                    depth_data
                )  # create dataframe from dictionary output

                self.save_depth_maps(depth_maps, depth_map_zarr_dir)  # save to zarr

    def save_depth_maps(self, depth_maps_dataframe, depth_map_zarr_dir):
        """Saves depth maps from a DataFrame into a Zarr store.

        Args:
            depth_maps_dataframe (pd.DataFrame): A DataFrame containing depth maps,
                where each row includes an 'image_name' and a 'depth_map'.
            depth_map_zarr_dir (str): Path to the directory where the depth maps will be stored in a Zarr group.
        """
        for _, row in depth_maps_dataframe.iterrows():
            store = zarr.open_group(depth_map_zarr_dir, mode="a")

            image_name = row["image_name"]

            depth_map = row["depth_map"]

            store[image_name] = depth_map.get()
