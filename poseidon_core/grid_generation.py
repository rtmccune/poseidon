import os
import zarr
import laspy
import pdal
import json
import numpy as np
from scipy.interpolate import griddata

# Added imports for the new PDAL method
import rasterio
import tempfile


class GridGenerator:
    """A class for generating and processing LiDAR data grids.

    This class allows for loading LiDAR data from a specified file, creating point arrays
    based on classification values, and generating grids of points in specified spatial extents
    and resolutions. The class also provides methods for saving the generated grids as
    compressed Zarr files.

    Attributes:
        filepath (str): The path to the LiDAR data file.
        filename (str): The name of the LiDAR data file.
        lidar (laspy.LasData): The loaded LiDAR data.
        min_x_extent (float): The minimum x-coordinate extent for grid generation.
        max_x_extent (float): The maximum x-coordinate extent for grid generation.
        min_y_extent (float): The minimum y-coordinate extent for grid generation.
        max_y_extent (float): The maximum y-coordinate extent for grid generation.
        point_mask_val (int): The classification value used to filter points.

    Methods:
        load_lidar(): Load LiDAR data from the specified file.
        create_point_array(point_mask_value=2): Create an array of filtered LiDAR points based on classification.
        gen_grid_pdal(...): Generates a grid using an efficient PDAL pipeline.
    """

    def __init__(
        self,
        file_path,
        min_x_extent=None,
        max_x_extent=None,
        min_y_extent=None,
        max_y_extent=None,
    ):
        """Initialize the LiDAR processing object.

        This constructor initializes the LiDAR processing object by loading the LiDAR data
        from the specified file path. It sets the spatial extents for the LiDAR data, which
        can be specified or automatically determined from the loaded data.

        Args:
            file_path (str): The path to the LiDAR data file to be loaded.
            min_x_extent (float, optional): The minimum x-coordinate extent for the grid.
                                            Defaults to None, which will set it to the minimum
                                            x-coordinate of the LiDAR data converted to meters.
            max_x_extent (float, optional): The maximum x-coordinate extent for the grid.
                                            Defaults to None, which will set it to the maximum
                                            x-coordinate of the LiDAR data converted to meters.
            min_y_extent (float, optional): The minimum y-coordinate extent for the grid.
                                            Defaults to None, which will set it to the minimum
                                            y-coordinate of the LiDAR data converted to meters.
            max_y_extent (float, optional): The maximum y-coordinate extent for the grid.
                                            Defaults to None, which will set it to the maximum
                                            y-coordinate of the LiDAR data converted to meters.
        """
        self.filepath = file_path
        self.filename = os.path.basename(file_path)
        self.lidar = self.load_lidar()
        
        # Determine extents, converting from feet to meters if no extents are provided
        # This assumes the base lidar.x/y/z are in feet if extents are None
        if min_x_extent is None:
            # Assuming units are feet if not specified, matching create_point_array logic
            scale_factor = 0.3048 
            self.min_x_extent = np.min(self.lidar.x) * scale_factor
            self.max_x_extent = np.max(self.lidar.x) * scale_factor
            self.min_y_extent = np.min(self.lidar.y) * scale_factor
            self.max_y_extent = np.max(self.lidar.y) * scale_factor
        else:
            # If extents are provided, assume they are already in the target units (meters)
            self.min_x_extent = min_x_extent
            self.max_x_extent = max_x_extent
            self.min_y_extent = min_y_extent
            self.max_y_extent = max_y_extent


    def load_lidar(self):
        """Load LiDAR data from the specified file.

        This function reads the LiDAR data from the file located at `self.filepath`
        and returns the LiDAR object for further processing.

        Returns:
            laspy.LasData: The LiDAR data loaded from the file.
        """
        lidar = laspy.read(self.filepath)

        return lidar

    def create_point_array(self, orig_units='feet', point_mask_value=2):
        """Create an array of LiDAR points based on classification.

        This function filters the LiDAR data to select points that match the
        specified classification value, converts their coordinates from feet to meters,
        and applies extent filtering based on predefined minimum and maximum extents.

        Args:
            orig_units (str, optional): The original units of the LAS file ('feet' or 'meters').
                                        Defaults to 'feet'.
            point_mask_value (int, optional): The classification value to filter points. Defaults to 2.

        Returns:
            np.ndarray: An array of filtered LiDAR points in meters, within the specified extents.
        """
        self.point_mask_val = point_mask_value

        point_mask = (
            self.lidar.classification == point_mask_value
        )  # Mask points based on classification

        if orig_units=='feet':
            # Convert feet to meters and stack into an array
            xyz_m = np.vstack([
                self.lidar.x[point_mask] * 0.3048,
                self.lidar.y[point_mask] * 0.3048,
                self.lidar.z[point_mask] * 0.3048
            ])
        else:
            # Data is already in meters
            xyz_m = np.vstack([
                self.lidar.x[point_mask],
                self.lidar.y[point_mask],
                self.lidar.z[point_mask]
            ])
        
        # Apply extent filtering
        # Assumes self.extents are in meters
        extent_mask = (
            (xyz_m[0] >= self.min_x_extent)
            & (xyz_m[0] <= self.max_x_extent)
            & (xyz_m[1] >= self.min_y_extent)
            & (xyz_m[1] <= self.max_y_extent)
        )

        return xyz_m[:, extent_mask]  # Keep only the points within the extents

    def gen_grid(self, resolution, z=0, dir="data/generated_grids"):
        """Generate a grid of points in the specified extent and resolution.

        This function creates a grid based on the specified resolution and z-value. It can
        generate a uniform grid at a specified elevation (z) or interpolate a set of points
        provided in a 2D array. The generated grid arrays are saved as compressed Zarr files
        in the specified directory.

        Args:
            resolution (float): The spacing between grid points in the x and y dimensions.
            z (int, float, np.ndarray, optional): The elevation value to use for the z dimension. Defaults to 0.
            dir (str, optional): The directory where the generated grid files will be saved. Defaults to 'generated_grids'.

        Returns:
            tuple: A tuple containing three numpy arrays:
                - grid_x (np.ndarray): The x-coordinates of the grid points.
                - grid_y (np.ndarray): The y-coordinates of the grid points.
                - grid_z (np.ndarray): The z-coordinates of the grid points.
        """
        # Check that the path exists
        if not os.path.exists(dir):
            os.makedirs(dir)  # Create the directory if not
            print(f"Directory to store grids created: {dir}")
        else:
            print(f"Directory to store grids already exists: {dir}")

        # Generate structured grid for extents and resolution
        grid_x, grid_y = np.mgrid[
            self.min_x_extent : self.max_x_extent : resolution,
            self.min_y_extent : self.max_y_extent : resolution,
        ]

        # If it or float provided for z, create set elevation grid
        if isinstance(z, int) or isinstance(z, float):
            grid_z = np.full_like(grid_x, z)

        # If points array is provided, fit points to structured grid
        if isinstance(z, np.ndarray):
            x = z[0]
            y = z[1]
            z = z[2]
            grid_z = griddata((x, y), z, (grid_x, grid_y), method="linear")

        grid_x = grid_x.T
        grid_y = grid_y.T
        grid_z = grid_z.T

        # Save the grid arrays to a compressed Zarr files
        zarr.save(os.path.join(dir, f"grid_x_{resolution}m.zarr"), grid_x)
        zarr.save(os.path.join(dir, f"grid_y_{resolution}m.zarr"), grid_y)
        zarr.save(os.path.join(dir, f"grid_z_{resolution}m.zarr"), grid_z)

        return grid_x, grid_y, grid_z
    
    # --------------------------------------------------------------------
    # PDAL IDW/MEAN/MAX METHOD (Corrected)
    # --------------------------------------------------------------------
    
    def gen_grid_pdal(self, resolution, orig_units='feet', point_mask_value=2, 
                      interpolation_method='idw', radius=None):
        """Generates an interpolated elevation array using a PDAL pipeline.

        This method provides an efficient way to generate a Digital Elevation Model (DEM)
        by streaming the source file, filtering, and writing directly to a raster.
        It uses a temporary file to store the intermediate raster, which is then
        read into a NumPy array.

        Args:
            resolution (float): The desired output resolution of the grid in the
                                target units (meters).
            orig_units (str, optional): The original units of the LAS file ('feet' or 'meters').
                                        If 'feet', a transformation will be applied.
                                        Defaults to 'feet'.
            point_mask_value (int, optional): The classification value to use for
                                              interpolation (e.g., 2 for ground).
                                              Defaults to 2.
            interpolation_method (str, optional): The interpolation algorithm to use.
                                                  Common values: 'idw' (default), 'mean', 
                                                  'min', 'max', 'count'.
                                                  Defaults to 'idw'.
            radius (float, optional): The search radius (in meters) to find points
                                      for interpolation. If None, PDAL uses a
                                      very small default (res * sqrt(2)) which
                                      often results in an empty grid.
                                      **A value of 0.5 or 1.0 is recommended.**

        Returns:
            np.ndarray: A 2D NumPy array containing the interpolated elevation values.
                        Areas with no data are filled with np.nan.
        
        Raises:
            Exception: If the PDAL pipeline execution fails.
        """
        print(f"Generating grid with PDAL (res={resolution}m, class={point_mask_value}, method={interpolation_method})...")

        tmp_filename = None  # Initialize to None
        try:
            # 1. Create a temporary file for the intermediate raster output
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                tmp_filename = f.name
            
            # 2. Calculate explicit grid dimensions
            width = int(np.ceil((self.max_x_extent - self.min_x_extent) / resolution))
            height = int(np.ceil((self.max_y_extent - self.min_y_extent) / resolution))

            # 3. Define the filter range string
            filter_range_str = (
                f"Classification[{point_mask_value}:{point_mask_value}],"
                f"X[{self.min_x_extent}:{self.max_x_extent}],"
                f"Y[{self.min_y_extent}:{self.max_y_extent}]"
            )
            
            # 4. Define the PDAL pipeline stages
            pipeline_stages = [
                {
                    "type": "readers.las",
                    "filename": self.filepath
                }
            ]

            # 5. Add unit transformation if necessary
            if orig_units == 'feet':
                print("Applying feet-to-meters transformation...")
                pipeline_stages.append({
                    "type": "filters.transformation",
                    "expression": "X = X * 0.3048; Y = Y * 0.3048; Z = Z * 0.3048"
                })

            # 6. Add filtering stage
            pipeline_stages.append({
                "type": "filters.range",
                "limits": filter_range_str
            })

            # 7. Define the writer stage (raster/gridding)
            nodata_value = -9999.0
            
            # --- FIX IS HERE ---
            # Set a default radius if one isn't provided, otherwise
            # the grid will be empty.
            if radius is None:
                # Set a more sensible default than PDAL's
                # Let's use 10x the resolution
                radius = resolution * 10
                print(f"Warning: No radius specified. Defaulting to {radius:.2f}m.")
            # --- END FIX ---
                
            pipeline_stages.append({
                "type": "writers.gdal",
                "filename": tmp_filename,
                "gdaldriver": "GTiff",
                "output_type": interpolation_method,
                "resolution": resolution,
                "nodata": nodata_value,
                "origin_x": self.min_x_extent,
                "origin_y": self.min_y_extent,
                "width": width,
                "height": height,
                "radius": radius  # <-- ADDED THE RADIUS
            })

            # 8. Create and execute the pipeline
            pipeline_dict = {"pipeline": pipeline_stages}
            pipeline_json = json.dumps(pipeline_dict)
            
            print("Executing PDAL pipeline...")
            pipeline = pdal.Pipeline(pipeline_json)
            pipeline.execute()
            print("PDAL execution complete.")

            # 9. Read the temporary raster file back into a NumPy array
            with rasterio.open(tmp_filename) as src:
                elevation_array = src.read(1).astype(float)
                nodata_val = src.nodata
                if nodata_val is not None:
                    elevation_array[elevation_array == nodata_val] = np.nan
            
            return elevation_array

        except Exception as e:
            print(f"Error during PDAL pipeline processing: {e}")
            raise  # Re-raise the exception to signal failure
        
        finally:
            # 10. Clean up the temporary file
            if tmp_filename and os.path.exists(tmp_filename):
                os.remove(tmp_filename)
                print(f"Cleaned up temporary file: {tmp_filename}")
                
    # --------------------------------------------------------------------
    # NEW PDAL TIN METHOD
    # --------------------------------------------------------------------
    
    def gen_grid_pdal_tin(self, resolution, orig_units='feet', point_mask_value=2):
        """Generates an interpolated elevation array using a PDAL TIN pipeline.

        This method is for 'linear' interpolation and is the direct PDAL
        equivalent to scipy's 'linear' (TIN) method. It first builds a
        Delaunay Triangulation (a 2D mesh) and then creates a raster
        by interpolating values from that mesh.

        Args:
            resolution (float): The desired output resolution of the grid in the
                                target units (meters).
            orig_units (str, optional): The original units of the LAS file ('feet' or 'meters').
                                        If 'feet', a transformation will be applied.
                                        Defaults to 'feet'.
            point_mask_value (int, optional): The classification value to use for
                                              interpolation (e.g., 2 for ground).
                                              Defaults to 2.

        Returns:
            np.ndarray: A 2D NumPy array containing the interpolated elevation values.
                        Areas with no data are filled with np.nan.
        
        Raises:
            Exception: If the PDAL pipeline execution fails.
        """
        print(f"Generating grid with PDAL (TIN method, res={resolution}m, class={point_mask_value})...")

        tmp_filename = None  # Initialize to None
        try:
            # 1. Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
                tmp_filename = f.name
            
            # 2. Calculate explicit grid dimensions
            width = int(np.ceil((self.max_x_extent - self.min_x_extent) / resolution))
            height = int(np.ceil((self.max_y_extent - self.min_y_extent) / resolution))

            # 3. Define the filter range string
            filter_range_str = (
                f"Classification[{point_mask_value}:{point_mask_value}],"
                f"X[{self.min_x_extent}:{self.max_x_extent}],"
                f"Y[{self.min_y_extent}:{self.max_y_extent}]"
            )
            
            # 4. Define the PDAL pipeline stages
            pipeline_stages = [
                {
                    "type": "readers.las",
                    "filename": self.filepath
                }
            ]

            # 5. Add unit transformation if necessary
            if orig_units == 'feet':
                print("Applying feet-to-meters transformation...")
                pipeline_stages.append({
                    "type": "filters.transformation",
                    "expression": "X = X * 0.3048; Y = Y * 0.3048; Z = Z * 0.3048"
                })

            # 6. Add filtering stage
            pipeline_stages.append({
                "type": "filters.range",
                "limits": filter_range_str
            })

            # 7. --- PDAL TIN WORKFLOW ---
            # First, create the Delaunay Triangulation (mesh)
            pipeline_stages.append({
                "type": "filters.delaunay"
            })
            
            # Second, create the raster from the mesh
            nodata_value = -9999.0
            pipeline_stages.append({
                "type": "filters.faceraster",
                "resolution": resolution,
                "origin_x": self.min_x_extent,
                "origin_y": self.min_y_extent,
                "width": width,
                "height": height,
                "nodata": nodata_value
            })
            
            # Third, *write* the raster created by faceraster to our file
            pipeline_stages.append({
                "type": "writers.raster",
                "filename": tmp_filename,
                "gdaldriver": "GTiff"
                # Note: No 'output_type' or 'resolution' here.
                # 'writers.raster' just saves what 'faceraster' created.
            })
            # --- END PDAL TIN WORKFLOW ---

            # 8. Create and execute the pipeline
            pipeline_dict = {"pipeline": pipeline_stages}
            pipeline_json = json.dumps(pipeline_dict)
            
            print("Executing PDAL TIN pipeline...")
            pipeline = pdal.Pipeline(pipeline_json)
            pipeline.execute()
            print("PDAL execution complete.")

            # 9. Read the temporary raster file
            with rasterio.open(tmp_filename) as src:
                elevation_array = src.read(1).astype(float)
                nodata_val = src.nodata
                if nodata_val is not None:
                    elevation_array[elevation_array == nodata_val] = np.nan
            
            return elevation_array

        except Exception as e:
            print(f"Error during PDAL TIN pipeline processing: {e}")
            raise
        
        finally:
            # 10. Clean up the temporary file
            if tmp_filename and os.path.exists(tmp_filename):
                os.remove(tmp_filename)
                print(f"Cleaned up temporary file: {tmp_filename}")