import os
import cv2
import zarr
import numpy as np
import cupy as cp
from datetime import datetime
from cupyx.scipy.interpolate import RegularGridInterpolator as reg_interp_gpu
from scipy.interpolate import RegularGridInterpolator as reg_interp

def _log(message):
    """Helper function for timestamped logging."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

class ImageRectifier:
    """
    Manages image rectification and distortion correction.

    This class uses intrinsic and extrinsic camera parameters, along
    with a 3D real-world coordinate grid (X, Y, Z), to compute the
    mapping from world coordinates to distorted image coords (U, V).

    This mapping is then used to rectify raw images.

    Parameters
    ----------
    intrinsics : np.ndarray or cp.ndarray
        11-element array of camera intrinsic parameters, including focal
        lengths and a principal point.
    extrinsics : np.ndarray or cp.ndarray
        6-element array of camera extrinsic parameters (rotation and
        translation).
    grid_x : np.ndarray or cp.ndarray
        2D array representing the real-world X coordinates.
    grid_y : np.ndarray or cp.ndarray
        2D array representing the real-world Y coordinates.
    grid_z : np.ndarray or cp.ndarray
        2D array representing the real-world Z coordinates (elevation).
    use_gpu : bool
        If True, use CuPy for GPU-accelerated computations. If False,
        use NumPy for CPU-based computation.

    Attributes
    ----------
    Ud : np.ndarray or cp.ndarray
        The computed distorted U (column) coordinates, rounded for
        indexing.
    Vd : np.ndarray or cp.ndarray
        The computed distorted V (row) coordinates, rounded for indexing.
    azimuth : float
        Azimuth angle (in radians) extracted from extrinsics.
    tilt : float
        Tilt angle (in radians) extracted from extrinsics.
    swing : float
        Swing angle (in radians) extracted from extrinsics.
    grid_shape : tuple
        The (row, col) shape of the input grids.

    Notes
    -----
    The class internally computes the camera projection matrix (P),
    applies distortion corrections, and calculates the `Ud` and `Vd`
    mapping upon initialization.

    The primary public methods for rectification are `merge_rectify`
    (for a single image) and `merge_rectify_folder`
    (for batch processing).
    Helper methods like `build_K`, `distort_UV`, and `xyz_to_dist_UV`
    are used internally during this process.

    """

    def __init__(
        self, intrinsics, extrinsics, grid_x, grid_y, grid_z, use_gpu=False
    ):
        """Initialize the object with camera parameters and grid data.

        Parameters
        ----------
        intrinsics : np.ndarray
            An 11 element numpy array representing the camera intrinsic
            parameters, including focal lengths and optical center.
        extrinsics : np.ndarray
            A 6-element numpy array containing the camera extrinsic
            parameters, specifying the rotation and translation of the
            camera in space.
        grid_x : np.ndarray
            A 2D numpy array representing the X coordinates of the grid
            points.
        grid_y : np.ndarray
            A 2D numpy array representing the Y coordinates of the grid
            points.
        grid_z : np.ndarray
            A 2D numpy array representing the Z coordinates (elevation)
            of the grid points.
        use_gpu : bool, optional
            A flag indicating whether to use GPU for computations.
            Default is False.
        """
        _log("--- Initializing ImageRectifier ---")

        self.use_gpu = use_gpu
        _log(f"  Mode: {'GPU (CuPy)' if use_gpu else 'CPU (NumPy)'}")

        # Convert numpy arrays to cupy arrays for GPU acceleration
        if use_gpu:
            self.intrinsics = cp.array(intrinsics)
            self.extrinsics = cp.array(extrinsics)
            self.grid_x = cp.array(grid_x)
            self.grid_y = cp.array(grid_y)
            self.grid_z = cp.array(grid_z)
        else:  # Else,set values to inputted numpy array format
            self.intrinsics = intrinsics
            self.extrinsics = extrinsics
            self.grid_x = grid_x
            self.grid_y = grid_y
            self.grid_z = grid_z
        
        _log(f"  Input grid shape: {grid_x.shape}")

        self.azimuth = self.extrinsics[3]  # Set azimuth from extrinsics
        self.tilt = self.extrinsics[4]  # Set tilt from extrinsics
        self.swing = self.extrinsics[5]  # Set swing from extrinsics
        self.grid_shape = grid_x.shape  # Set grid shape

        _log("  Starting distortion map computation (Ud, Vd)...")
        self.Ud, self.Vd = (
            # Compute list of distorted U and V coord. corresponding to
            # world xyz coord.
            self._compute_Ud_Vd()
        )
        _log("  ...Distortion map computation complete.")
        _log("--- ImageRectifier Initialization Complete ---")

    def _load_image(self, image_path, labels):
        """Load an image from the specified path.

        Parameters
        ----------
        image_path : str
            The file path to the image to be loaded.
        labels : bool
            A flag indicating whether to load the image in grayscale
            (True) or in color (False). If True, the image will be
            loaded as a single-channel grayscale image; if False, the
            image will be loaded in RGB format.

        Returns
        -------
        np.ndarray or cp.ndarray
            The loaded image as a numpy array (if use_gpu is False)
            or as a cupy array (if use_gpu is True).

        Notes
        -----
        The function uses OpenCV to read the image. If `self.use_gpu`
        is True, the image is converted to a cupy array for GPU
        processing. If `labels` is True, the image is loaded in
        grayscale; otherwise, it is loaded in RGB format.
        """
        # If grayscale image of labels, read as grayscale image
        if labels:
            I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:  # Else, read in as RGB
            I = cv2.imread(image_path)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

        # Convert numpy array from OpenCV to cupy array for GPU process.
        if self.use_gpu:
            return cp.asarray(I)
        else:
            return I

    def _reshape_grids(self):
        """Reshape the grid arrays into a single 2D array of XYZ coord.

        Returns
        -------
        np.ndarray or cp.ndarray
            A 2D array where each row represents a point in
            3D space (X, Y, Z). The shape of the array will
            be (N, 3), where N is the total number of grid points.

        Notes
        -----
        The function takes the transposed grid arrays for X, Y, and Z
        coordinates, reshapes them into column vectors, and concatenates
        them along the second axis. If `self.use_gpu` is True, the
        resulting array will be a cupy array for GPU processing;
        otherwise, it will be a numpy array for CPU processing.
        """
        # Reshape grid arrays into vectors
        x_vec = self.grid_x.T.reshape(-1, 1)
        y_vec = self.grid_y.T.reshape(-1, 1)
        z_vec = self.grid_z.T.reshape(-1, 1)

        if (
            self.use_gpu
        ):  # Use cupy concatenate function if processing with a GPU
            xyz = cp.concatenate([x_vec, y_vec, z_vec], axis=1)
        else:  # Else, use numpy concatenate function
            xyz = np.concatenate([x_vec, y_vec, z_vec], axis=1)

        return xyz

    def _get_pixels(self, image):
        """Extract pixel values from an image at specified grid points.

        Parameters
        ----------
        image : np.ndarray or cp.ndarray
            The input image from which pixel values are to be extracted.
            The image can be a 2D (grayscale) or 3D (color) array.

        Returns
        -------
        np.ndarray or cp.ndarray
            A 3D array containing the interpolated pixel values
            at the specified grid points (self.Vd, self.Ud).
            The shape of the output will be (grid_shape[0],
            grid_shape[1], number of channels) for multi-channel images
            or (grid_shape[0], grid_shape[1], 1) for single-channel
            images.

        Notes
        -----
        The function uses regular grid interpolation to map pixel values
        from the input image to the specified grid points.
        If `self.use_gpu` is True, the function utilizes cupy for GPU
        processing; otherwise, it uses numpy for CPU processing. Values
        that fall outside the image boundaries are masked and assigned
        NaN (for GPU) or 0 (for CPU) to indicate invalid pixels.
        """
        # Set the array module (numpy or cupy) and interpolator function
        if self.use_gpu:
            xp = cp
            reg_interp_func = reg_interp_gpu
        else:
            xp = np
            reg_interp_func = reg_interp

        im_s = image.shape

        # Unify 2D/3D logic by adding a channel dimension to 2D images
        # This allows same logic for RGB and grayscale images
        if len(im_s) == 2:
            image = image[:, :, xp.newaxis]
            im_s = image.shape  # Update shape to (H, W, 1)

        # Define the grid axes
        points = (xp.arange(im_s[0]), xp.arange(im_s[1]))

        # Create interpolator
        rgi = reg_interp_func(
            points,
            image,  # Pass the (H, W, C) image directly
            bounds_error=False,
            fill_value=xp.nan,
        )

        # Call the interpolator to get all channel values
        ir = rgi((self.Vd, self.Ud))

        # Apply custom boundary mask and set fill values
        if self.use_gpu:
            # GPU/CuPy path (no 'errstate')
            mask_u = xp.logical_or(self.Ud <= 1, self.Ud >= im_s[1])
            mask_v = xp.logical_or(self.Vd <= 1, self.Vd >= im_s[0])
            mask = xp.logical_or(mask_u, mask_v)

            # Apply mask (fills with NaN)
            ir = xp.where(mask[:, :, xp.newaxis], xp.nan, ir)

        else:
            # CPU/NumPy path (use 'errstate' to suppress warnings)
            with xp.errstate(invalid="ignore"):
                mask_u = xp.logical_or(self.Ud <= 1, self.Ud >= im_s[1])
                mask_v = xp.logical_or(self.Vd <= 1, self.Vd >= im_s[0])
                mask = xp.logical_or(mask_u, mask_v)

            # Apply mask (fills with 0)
            ir[mask] = 0

        return ir

    def _CIRN_angles_to_R(self):
        """Calculate the rotation matrix from CIRN angles
        (azimuth, tilt, swing).

        Returns
        -------
        np.ndarray or cp.ndarray
            A 3x3 rotation matrix that represents the orientation
            in 3D space based on the specified azimuth, tilt, and swing
            angles.

        Notes
        -----
        The rotation matrix is constructed using the provided angles:
        - Azimuth: The angle of rotation around the vertical axis.
        - Tilt: The angle of elevation from the horizontal plane.
        - Swing: The angle of rotation around the forward axis.
        If `self.use_gpu` is True, the function returns the rotation
        matrix as a cupy array for GPU processing; otherwise, it returns
        a numpy array for CPU processing.
        """
        R = np.empty((3, 3))  # Initialize empty 3x3 rotation matrix

        # Pre-calculate all 6 trig values once
        ca = np.cos(self.azimuth)
        sa = np.sin(self.azimuth)
        ct = np.cos(self.tilt)
        st = np.sin(self.tilt)
        cs = np.cos(self.swing)
        ss = np.sin(self.swing)

        # Calculate rotation matrix values
        # Note: using numpy rather than cupy even for GPU processing as
        # it is a small matrix that is only calculated once.
        # Porting to GPU would only add time.
        R[0, 0] = -ca * cs - sa * ct * ss
        R[0, 1] = cs * sa - ss * ct * ca
        R[0, 2] = -ss * st
        R[1, 0] = -ss * ca + cs * ct * sa
        R[1, 1] = ss * sa + cs * ct * ca
        R[1, 2] = cs * st
        R[2, 0] = st * sa
        R[2, 1] = st * ca
        R[2, 2] = -ct

        if self.use_gpu:  # Return as a cupy array if GPU processing
            return cp.array(R)
        else:
            return R

    def _build_K(self):
        """Constructs the camera intrinsic matrix K.

        This matrix is used to go from camera coordinates to undistorted
        UV coordinates.

        Returns
        -------
        np.ndarray or cp.ndarray
            A 3x3 camera intrinsic matrix that defines the
            mapping from 3D world coordinates to 2D image coordinates.

        Notes
        -----
        The intrinsic matrix K is constructed using the parameters from
        the `self.intrinsics` array:
        - K[0,0]: Negative focal length in the x-direction.
        - K[1,1]: Negative focal length in the y-direction.
        - K[0,2]: x-coordinate of the principal point.
        - K[1,2]: y-coordinate of the principal point.
        - K[2,2]: Set to 1 for homogeneous coordinates.
        If `self.use_gpu` is True, the function returns the intrinsic
        matrix as a cupy array for GPU processing; otherwise, it returns
        a numpy array for CPU processing.
        """
        # Initialize empty 3x3 intrinsic matrix
        if self.use_gpu:
            K = cp.zeros((3, 3))
        else:
            K = np.zeros((3, 3))

        # Set focal length in x direction
        K[0, 0] = -(self.intrinsics[4])

        # Set focal length in y direction
        K[1, 1] = -(self.intrinsics[5])

        # Set x-coordinate of principal point
        K[0, 2] = self.intrinsics[2]

        # Set y-coordinate of principal point
        K[1, 2] = self.intrinsics[3]
        K[2, 2] = 1

        return K

    def _intrinsics_extrinsics_to_P(self):
        """Computes the camera projection matrix P.

        Combines intrinsics and extrinsics to form the 4x4 projection
        matrix P, along with its component matrices K, R, and IC.

        Returns
        -------
        P : np.ndarray or cp.ndarray
            3x4 camera projection matrix.
        K : np.ndarray or cp.ndarray
            3x3 intrinsic matrix.
        R : np.ndarray or cp.ndarray
            3x3 rotation matrix.
        IC : np.ndarray or cp.ndarray
            4x3 matrix combining identity and camera translation.

        Notes
        -----
        Output P is normalized for homogenous coordinates. The
        projection matrix P is constructed as follows:
        - K is the intrinsic matrix built from camera parameters.
        - R is the rotation matrix calculated from the azimuth, tilt,
        and swing angles.
        - The extrinsics (translation) are applied as a column vector
        [-x, -y, -z].
        The resulting matrix P transforms 3D world coordinates into 2D
        image coordinates.
        If `self.use_gpu` is True, the function utilizes cupy for GPU
        processing; otherwise, it uses numpy for CPU processing. The
        last row of the projection matrix is normalized by dividing by
        P[2, 3].
        """
        # Set the array module (numpy or cupy)
        xp = cp if self.use_gpu else np

        # Generate K and R matrices
        K = self._build_K()
        R = self._CIRN_angles_to_R()

        x = self.extrinsics[0]
        y = self.extrinsics[1]
        z = self.extrinsics[2]

        # Create translation matrix
        column_vec = xp.array([-x, -y, -z]).reshape(-1, 1)
        IC = xp.concatenate([xp.eye(3), column_vec], axis=1)

        # Combine K, R, and IC into P
        P = xp.dot(K, xp.dot(R, IC))

        # Normalize for homogeneous coordinates
        P /= P[2, 3]

        return P, K, R, IC

    def _distort_UV(self, UV):
        """Distort UV coordinates based on camera intrinsic parameters.

        Parameters
        ----------
        UV : np.ndarray or cp.ndarray
            A 2xN array containing UV coordinates,
            where U is in the first row and V in the second.

        Returns
        -------
        Ud : np.ndarray or cp.ndarray
            The distorted U coordinates.
        Vd : np.ndarray or cp.ndarray
            The distorted V coordinates.
        flag : np.ndarray or cp.ndarray
            Array indicating invalid (0) or valid (1) coordinates.

        Notes
        -----
        The distortion is calculated using radial and tangential
        distortion models based on the camera's intrinsic parameters.
        The function applies corrections to the input UV coordinates and
        ensures that any resulting coordinates that fall outside the
        valid image boundaries are set to zero. The tangential
        distortion is also computed at the corners of the image for
        comparison. If `self.use_gpu` is True, the function utilizes
        cupy for GPU processing; otherwise, it uses numpy for CPU
        processing.
        """
        # Set the array module (numpy or cupy)
        xp = cp if self.use_gpu else np

        # Assign coefficients out of intrinsic matrix
        NU = self.intrinsics[0]
        NV = self.intrinsics[1]
        c0U = self.intrinsics[2]
        c0V = self.intrinsics[3]
        fx = self.intrinsics[4]
        fy = self.intrinsics[5]
        d1 = self.intrinsics[6]
        d2 = self.intrinsics[7]
        d3 = self.intrinsics[8]
        t1 = self.intrinsics[9]
        t2 = self.intrinsics[10]

        # Separate UV into vectors U and V
        U = UV[0, :]
        V = UV[1, :]

        # Normalize distances
        x = (U - c0U) / fx
        y = (V - c0V) / fy

        # Radial distortion
        r2 = x**2 + y**2
        fr = 1 + d1 * r2 + d2 * r2**2 + d3 * r2**3

        # Tangential distortion
        dx = 2 * t1 * x * y + t2 * (r2 + 2 * x**2)
        dy = t1 * (r2 + 2 * y**2) + 2 * t2 * x * y

        # Apply correction
        xd = x * fr + dx
        yd = y * fr + dy
        Ud = xd * fx + c0U
        Vd = yd * fy + c0V

        # Find and mask negative UV coordinates
        flag_mask = (Ud < 0) | (Ud > NU) | (Vd < 0) | (Vd > NV)
        Ud[flag_mask] = 0
        Vd[flag_mask] = 0

        # Define maximum possible tangential distortion at the corners
        Um = xp.array([0, 0, NU.item(), NU.item()])
        Vm = xp.array([0, NV.item(), NV.item(), 0])

        # Normalization
        xm = (Um - c0U) / fx
        ym = (Vm - c0V) / fy
        r2m = xm**2 + ym**2

        # Tangential Distortion at corners
        dxm = 2 * t1 * xm * ym + t2 * (r2m + 2 * xm**2)
        dym = t1 * (r2m + 2 * ym**2) + 2 * t2 * xm * ym

        # Find max values at corners
        max_dym = xp.max(xp.abs(dym))
        max_dxm = xp.max(xp.abs(dxm))

        # Use direct boolean masking (more efficient than np.where)
        mask_dy = xp.abs(dy) > max_dym
        mask_dx = xp.abs(dx) > max_dxm

        # Initialize and set flag array
        flag = xp.ones_like(Ud)
        flag[mask_dy] = 0.0
        flag[mask_dx] = 0.0

        return Ud, Vd, flag

    def _xyz_to_dist_UV(self):
        """Maps 3D world (XYZ) coordinates to distorted 2D (UV) image
        coordinates.

        This method uses the full camera model (P matrix) and distortion
        parameters to project the 3D grid points onto the image plane.

        Returns
        -------
        DU : np.ndarray or cp.ndarray
            The distorted U coordinates, reshaped to match the grid.
        DV : np.ndarray or cp.ndarray
            The distorted V coordinates, reshaped to match the grid.

        Notes
        -----
        The function computes the projection of 3D coordinates onto the
        image plane using camera intrinsic and extrinsic parameters. It
        first transforms the 3D coordinates into homogeneous coordinates
        and then calculates the UV coordinates. Distortion is applied to
        these UV coordinates based on the camera's intrinsic parameters.
        The function also checks for negative Z camera coordinates and
        updates a flag to indicate valid points. If `self.use_gpu` is
        True, cupy is used for GPU processing; otherwise, numpy is used
        for CPU processing. The final distorted U and V coordinates
        are returned, multiplied by the flag to indicate valid points.
        """

        # Set the array module (numpy or cupy)
        xp = cp if self.use_gpu else np

        # Create matrix P and its components
        P, K, R, IC = self._intrinsics_extrinsics_to_P()

        # Reshape grids
        xyz = self._reshape_grids()  # (N, 3)

        # Create homogeneous coordinates and project to image plane
        xyz_homogeneous = xp.vstack((xyz.T, xp.ones(xyz.shape[0])))
        UV_homogeneous = xp.dot(P, xyz_homogeneous)

        # Perspective divide
        UV = UV_homogeneous[:2, :] / UV_homogeneous[2, :]

        # Apply distortion
        Ud, Vd, flag = self._distort_UV(UV)  # (N,) (N,) (N,)
        DU = Ud.reshape(self.grid_shape, order="F")
        DV = Vd.reshape(self.grid_shape, order="F")

        # Compute camera coordinates
        xyzC = xp.dot(xp.dot(R, IC), xyz_homogeneous)  # (3, N)

        # Find negative Zc coordinates (Z <= 0) and update the flag
        # negative_z_indices = xp.where(xyzC[2, :] <= 0.0)
        mask_z = xyzC[2, :] <= 0.0

        # Apply the mask to the flag
        # flag[negative_z_indices] = 0.0
        flag[mask_z] = 0.0
        flag = flag.reshape(self.grid_shape, order="F")

        return DU * flag, DV * flag

    def _compute_Ud_Vd(self):
        """Compute the distorted U and V coordinates from 3D points.

        Returns
        -------
        Ud : np.ndarray or cp.ndarray
            The final, rounded, integer-cast U coordinates.
        Vd : np.ndarray or cp.ndarray
            The final, rounded, integer-cast V coordinates.

        Notes
        -----
        This function computes the distorted U and V coordinates by
        calling the `xyz_to_dist_UV` method, which converts 3D
        coordinates to 2D distorted image coordinates. The resulting
        coordinates are then rounded to the nearest integer and cast to
        integers to represent pixel indices. The function uses either
        NumPy for CPU processing or CuPy for GPU processing based on the
        value of `self.use_gpu`.
        """
        # Set the array module
        xp = cp if self.use_gpu else np

        Ud, Vd = (
            self._xyz_to_dist_UV()
        )  # Calculate distorted U and V coordinates

        # Round to the nearest integer to represent pixel indices
        Ud = xp.round(Ud).astype(int)
        Vd = xp.round(Vd).astype(int)

        return Ud, Vd

    def merge_rectify(self, image_path, labels=False, verbose=False):
        """Merge and rectify an image based on the provided path.

        Parameters
        ----------
        image_path : str
            The path to the image file to be loaded and rectified.
        labels : bool, optional
            A flag indicating whether to load the image in grayscale
            (True) or in color (False). Default is False.
        verbose : bool, optional
            If True, print a log message when rectifying this image.
            Default is False, to avoid spamming logs during batch jobs.

        Returns
        -------
        np.ndarray or cp.ndarray
            The rectified image as an array of type uint8.

        Notes
        -----
        This function loads an image from the specified path using the
        `load_image` method, processes it to obtain rectified pixel
        values using the `get_pixels` method, and returns the rectified
        image. The function utilizes CuPy for GPU processing if
        `self.use_gpu` is set to True; otherwise, it uses NumPy for CPU
        processing. The output is cast to an array of type uint8 to
        represent pixel values.
        """
        if verbose:
            _log(f"Rectifying single image: {image_path}")

        # Load image from path
        image = self._load_image(image_path, labels)

        rectified_image = self._get_pixels(
            image
        )  # Collect pixel values for real world coordinates

        if self.use_gpu:  # If GPU processing, return image as cupy arr
            return cp.array(rectified_image, dtype=np.uint8)

        else:  # Else, return as numpy array
            return np.array(rectified_image, dtype=np.uint8)

        
    def merge_rectify_folder(self, folder_path, zarr_store_path, labels=False):
        """Merge and rectify all images in a specified folder and save
        them to a Zarr store.

        Parameters
        ----------
        folder_path : str
            The path to the folder containing images to be rectified.
        zarr_store_path : str
            The path to the Zarr store where rectified images will be
            saved.
        labels : bool, optional
            A flag indicating whether to load the image in grayscale
            (True) or in color (False). Default is False.

        Returns
        -------
        None

        Notes
        -----
        This function iterates over all image files in the specified
        folder, rectifies each image using the `merge_rectify` method,
        and saves the resulting rectified images to the specified Zarr
        store. The dataset names in the Zarr store are created by
        appending '_rectified' to the original image names. The Zarr
        store is opened in append mode before processing the images,
        allowing for multiple images to be saved efficiently. CuPy is
        utilized for GPU processing if `self.use_gpu` is set to True;
        otherwise, standard arrays are used.
        """
        _log("\n=== Starting Batch Rectification ===")
        _log(f"  Source folder: {folder_path}")
        _log(f"  Output Zarr store: {zarr_store_path}")

        # Open the Zarr store once before the loop
        try:
            store = zarr.open_group(zarr_store_path, mode="a")
        except Exception as e:
            _log(f"  ERROR: Could not open Zarr store at {zarr_store_path}. {e}")
            _log("=== Batch Rectification Aborted ===")
            return

        # Get a list of all images in the folder
        try:
            image_names = os.listdir(folder_path)
        except FileNotFoundError:
            _log(f"  ERROR: Source folder not found at {folder_path}.")
            _log("=== Batch Rectification Aborted ===")
            return
            
        total_images = len(image_names)

        if total_images == 0:
            _log("  WARNING: No images found in source folder. Nothing to do.")
            _log("=== Batch Rectification Complete ===")
            return

        _log(f"  Found {total_images} images to process.")

        # Determine report interval (print ~10 updates + first/last)
        report_interval = max(1, total_images // 10)

        for i, image_name in enumerate(image_names):
            # Log progress periodically
            if (i + 1) % report_interval == 0 or i == 0 or (i + 1) == total_images:
                _log(f"  Processing image {i + 1}/{total_images}: {image_name}")

            image_path = os.path.join(folder_path, image_name)
            
            try:
                rectified_image = self.merge_rectify(
                    image_path, labels, verbose=False  # Keep this quiet
                )
            except Exception as e:
                _log(f"  ERROR: Failed to rectify image {image_name}. {e}")
                continue # Skip to the next image

            # Create a dataset name by appending 'rectified' to the
            # original image name
            dataset_name = f"{os.path.splitext(image_name)[0]}_rectified"

            # Save the rectified image array to the Zarr store
            try:
                if self.use_gpu:  # If GPU processing
                    store[dataset_name] = (
                        rectified_image.get()
                    )  # Port GPU array to CPU and save
                else:  # Else, save array
                    store[dataset_name] = rectified_image
            except Exception as e:
                _log(f"  ERROR: Failed to save {dataset_name} to Zarr store. {e}")


        # Print success message after all images are processed
        _log(f"  Successfully processed {total_images} images.")
        _log("=== Batch Rectification Complete ===")
