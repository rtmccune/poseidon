import os
import cv2
import zarr
import numpy as np
import cupy as cp
from tqdm import tqdm
from cupyx.scipy.interpolate import RegularGridInterpolator as reg_interp_gpu
from scipy.interpolate import RegularGridInterpolator as reg_interp


class ImageRectifier:
    """
    A class for rectifying images and handling distortion correction using intrinsic and extrinsic camera parameters
    and a 3D grid representing real-world coordinates.

    This class provides methods to build the camera projection matrix, apply distortion corrections, and rectify images.
    It can operate on both CPU and GPU (using CuPy) to enhance performance when processing large datasets.

    Attributes:
        intrinsics (cp.ndarray or np.ndarray): An 11-element array representing the camera intrinsic parameters,
                                               including focal lengths and optical center.
        extrinsics (cp.ndarray or np.ndarray): A 6-element array containing the camera extrinsic parameters,
                                               specifying the rotation and translation of the camera in space.
        grid_x (cp.ndarray or np.ndarray): A 2D array representing the X coordinates of the grid points.
        grid_y (cp.ndarray or np.ndarray): A 2D array representing the Y coordinates of the grid points.
        grid_z (cp.ndarray or np.ndarray): A 2D array representing the Z coordinates (elevation) of the grid points.
        use_gpu (bool): Indicates whether to use GPU for computations.
        azimuth (float): The azimuth angle extracted from the extrinsic parameters.
        tilt (float): The tilt angle extracted from the extrinsic parameters.
        swing (float): The swing angle extracted from the extrinsic parameters.
        grid_shape (tuple): The shape of the grid used for rectification.
        Ud (cp.ndarray or np.ndarray): The distorted U coordinates computed from the world XYZ grid.
        Vd (cp.ndarray or np.ndarray): The distorted V coordinates computed from the world XYZ grid.

    Methods:
        load_image(image_path, labels): Load an image from the specified path.
        reshape_grids(): Reshape the grid arrays into a single 2D array of XYZ coordinates.
        get_pixels(image): Extract pixel values from an image at specified grid points.
        CIRN_angles_to_R(): Converts azimuth, swing, and tilt angles into a rotation matrix R.
        build_K(): Constructs the intrinsic matrix K from the intrinsic parameters.
        intrinsics_extrinsics_to_P(): Combines intrinsic and extrinsic parameters to form the camera projection matrix.
        distort_UV(UV): Applies distortion correction to UV coordinates.
        xyz_to_dist_UV(): Converts 3D points to distorted UV coordinates.
        compute_Ud_Vd(): Computes the distorted UV coordinates and rounds them for indexing.
        merge_rectify(image_path, labels=False): Loads an image and rectifies it, optionally using label information.
        merge_rectify_folder(folder_path, zarr_store_path, labels=False): Processes and rectifies all images in a folder, saving results to a Zarr store.
    """

    def __init__(self, intrinsics, extrinsics, grid_x, grid_y, grid_z, use_gpu=False):
        """Initialize the object with camera parameters and grid data.

        Args:
            intrinsics (np.ndarray): An 11 element numpy array representing the camera intrinsic parameters,
            including focal lengths and optical center.
            extrinsics (np.ndarray): A 6-element numpy array containing the camera extrinsic parameters,
            specifying the rotation and translation of the camera in space.
            grid_x (np.ndarray): A 2D numpy array representing the X coordinates of the grid points.
            grid_y (np.ndarray): A 2D numpy array representing the Y coordinates of the grid points.
            grid_z (np.ndarray): A 2D numpy array representing the Z coordinates (elevation) of the grid points.
            use_gpu (bool, optional): A flag indicating whether to use GPU for computations. Defaults to False.
        """
        self.use_gpu = use_gpu

        if use_gpu:  # Convert numpy arrays to cupy arrays for GPU acceleration
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

        self.azimuth = self.extrinsics[3]  # Set azimuth from extrinsics
        self.tilt = self.extrinsics[4]  # Set tilt from extrinsics
        self.swing = self.extrinsics[5]  # Set swing from extrinsics
        self.grid_shape = grid_x.shape  # Set grid shape
        self.Ud, self.Vd = (
            self.compute_Ud_Vd()
        )  # Compute list of distorted U and V coordinates corresponding to world xyz coordinates

    def load_image(self, image_path, labels):
        """ "Load an image from the specified path.

        Args:
            image_path (str): The file path to the image to be loaded.
            labels (bool): A flag indicating whether to load the image in grayscale (True)
            or in color (False). If True, the image will be loaded as a
            single-channel grayscale image; if False, the image will be
            loaded in RGB format.

        Returns:
            np.ndarry or cp.ndarray: The loaded image as a numpy array (if use_gpu is False)
            or as a cupy array (if use_gpu is True).
        Notes:
            The function uses OpenCV to read the image. If `self.use_gpu` is True, the image
            is converted to a cupy array for GPU processing. If `labels` is True, the image
            is loaded in grayscale; otherwise, it is loaded in RGB format.
        """
        if labels:  # If grayscale image of labels, read as grayscale image
            I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:  # Else, read in as RGB
            I = cv2.imread(image_path)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        if self.use_gpu:
            return cp.asarray(
                I
            )  # Convert numpy array from OpenCV to cupy array for GPU processing.
        else:
            return I

    def reshape_grids(self):
        """Reshape the grid arrays into a single 2D array of XYZ coordinates.

        Returns:
            np.ndarray or cp.ndarray: A 2D array where each row represents a point in
            3D space (X, Y, Z). The shape of the array will
            be (N, 3), where N is the total number of grid points.

        Notes:
            The function takes the transposed grid arrays for X, Y, and Z coordinates,
            reshapes them into column vectors, and concatenates them along the second axis.
            If `self.use_gpu` is True, the resulting array will be a cupy array for GPU processing;
            otherwise, it will be a numpy array for CPU processing.
        """
        # Reshape grid arrays into vectors
        x_vec = self.grid_x.T.reshape(-1, 1)
        y_vec = self.grid_y.T.reshape(-1, 1)
        z_vec = self.grid_z.T.reshape(-1, 1)

        if self.use_gpu:  # Use cupy concatenate function if processing with a GPU
            xyz = cp.concatenate([x_vec, y_vec, z_vec], axis=1)
        else:  # Else, use numpy concatenate function
            xyz = np.concatenate([x_vec, y_vec, z_vec], axis=1)

        return xyz

    def get_pixels(self, image):
        """Extract pixel values from an image at specified grid points.

        Args:
            image (np.ndarray or cp.ndarray): The input image from which pixel values are to be extracted.
            The image can be a 2D (grayscale) or 3D (color) array.

        Returns:
            np.ndarray or cp.ndarray: A 3D array containing the interpolated pixel values
                    at the specified grid points (self.Vd, self.Ud).
                    The shape of the output will be (grid_shape[0], grid_shape[1],
                    number of channels) for multi-channel images or
                    (grid_shape[0], grid_shape[1], 1) for single-channel images.

        Notes:
            The function uses regular grid interpolation to map pixel values from the input image
            to the specified grid points. If `self.use_gpu` is True, the function utilizes cupy for
            GPU processing; otherwise, it uses numpy for CPU processing. Values that fall outside the
            image boundaries are masked and assigned NaN (for GPU) or 0 (for CPU) to indicate invalid pixels.
        """

        im_s = image.shape

        # Use regular grid interpolator to grab points
        if self.use_gpu:  # Use cupy library if GPU processing
            if len(im_s) > 2:
                ir = cp.full((self.grid_shape[0], self.grid_shape[1], im_s[2]), cp.nan)
                for i in range(im_s[2]):
                    rgi = reg_interp_gpu(
                        (cp.arange(0, image.shape[0]), cp.arange(0, image.shape[1])),
                        image[:, :, i],
                        bounds_error=False,
                        fill_value=cp.nan,
                    )
                    ir[:, :, i] = rgi((self.Vd, self.Ud))
            else:
                ir = cp.full((self.grid_shape[0], self.grid_shape[1], 1), cp.nan)
                rgi = reg_interp_gpu(
                    (cp.arange(0, image.shape[0]), cp.arange(0, image.shape[1])),
                    image,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                ir[:, :, 0] = rgi((self.Vd, self.Ud))

            # Mask out values out of range
            mask_u = cp.logical_or(self.Ud <= 1, self.Ud >= image.shape[1])
            mask_v = cp.logical_or(self.Vd <= 1, self.Vd >= image.shape[0])
            mask = cp.logical_or(mask_u, mask_v)

            # Use cp.where to assign NaN where the mask is True
            if len(im_s) > 2:
                mask = mask[
                    :, :, None
                ]  # Adding a channel dimension (matching ir's shape)
                ir = cp.where(mask, cp.nan, ir)  # For multi-channel data
            else:
                ir[mask] = cp.nan  # For 2D data

        else:  # Else, use numpy library for CPU processing
            if len(im_s) > 2:
                ir = np.full((self.grid_shape[0], self.grid_shape[1], im_s[2]), np.nan)
                for i in range(im_s[2]):
                    rgi = reg_interp(
                        (np.arange(0, image.shape[0]), np.arange(0, image.shape[1])),
                        image[:, :, i],
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    ir[:, :, i] = rgi((self.Vd, self.Ud))
            else:
                ir = np.full((self.grid_shape[0], self.grid_shape[1], 1), np.nan)
                rgi = reg_interp(
                    (np.arange(0, image.shape[0]), np.arange(0, image.shape[1])),
                    image,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                ir[:, :, 0] = rgi((self.Vd, self.Ud))

            # Mask out values out of range
            with np.errstate(invalid="ignore"):
                mask_u = np.logical_or(self.Ud <= 1, self.Ud >= image.shape[1])
                mask_v = np.logical_or(self.Vd <= 1, self.Vd >= image.shape[0])
            mask = np.logical_or(mask_u, mask_v)
            if len(im_s) > 2:
                ir[mask, :] = 0
            else:
                ir[mask] = 0

        return ir

    def CIRN_angles_to_R(self):
        """Calculate the rotation matrix from CIRN angles (azimuth, tilt, swing).

        Returns:
            np.ndarray or cp.ndarray: A 3x3 rotation matrix that represents the orientation
            in 3D space based on the specified azimuth, tilt, and swing angles.

        Notes:
            The rotation matrix is constructed using the provided angles:
            - Azimuth: The angle of rotation around the vertical axis.
            - Tilt: The angle of elevation from the horizontal plane.
            - Swing: The angle of rotation around the forward axis.
            If `self.use_gpu` is True, the function returns the rotation matrix as a cupy array for
            GPU processing; otherwise, it returns a numpy array for CPU processing.
        """
        R = np.empty((3, 3))  # Initialize empty 3x3 rotation matrix

        # Calculate rotation matrix values
        # Note: using numpy rather than cupy even for GPU processing as it is a small
        # matrix that is only calculated once. Porting to GPU would only add time.
        R[0, 0] = -np.cos(self.azimuth) * np.cos(self.swing) - np.sin(
            self.azimuth
        ) * np.cos(self.tilt) * np.sin(self.swing)
        R[0, 1] = np.cos(self.swing) * np.sin(self.azimuth) - np.sin(
            self.swing
        ) * np.cos(self.tilt) * np.cos(self.azimuth)
        R[0, 2] = -np.sin(self.swing) * np.sin(self.tilt)
        R[1, 0] = -np.sin(self.swing) * np.cos(self.azimuth) + np.cos(
            self.swing
        ) * np.cos(self.tilt) * np.sin(self.azimuth)
        R[1, 1] = np.sin(self.swing) * np.sin(self.azimuth) + np.cos(
            self.swing
        ) * np.cos(self.tilt) * np.cos(self.azimuth)
        R[1, 2] = np.cos(self.swing) * np.sin(self.tilt)
        R[2, 0] = np.sin(self.tilt) * np.sin(self.azimuth)
        R[2, 1] = np.sin(self.tilt) * np.cos(self.azimuth)
        R[2, 2] = -np.cos(self.tilt)

        if self.use_gpu:  # Return as a cupy array if GPU processing
            return cp.array(R)
        else:
            return R

    def build_K(self):
        """Construct the camera intrinsic matrix K. This matrix is used to go from camera coordinates to undistorted UV coordinates.

        Returns:
            np.ndarray or cp.ndarray: A 3x3 camera intrinsic matrix that defines the
            mapping from 3D world coordinates to 2D image coordinates.

        Notes:
            The intrinsic matrix K is constructed using the parameters from the
            `self.intrinsics` array:
            - K[0,0]: Negative focal length in the x-direction.
            - K[1,1]: Negative focal length in the y-direction.
            - K[0,2]: x-coordinate of the principal point.
            - K[1,2]: y-coordinate of the principal point.
            - K[2,2]: Set to 1 for homogeneous coordinates.
            If `self.use_gpu` is True, the function returns the intrinsic matrix as a cupy array
            for GPU processing; otherwise, it returns a numpy array for CPU processing.
        """
        # Initialize empty 3x3 intrinsic matrix
        if self.use_gpu:
            K = cp.zeros((3, 3))
        else:
            K = np.zeros((3, 3))
        K[0, 0] = -(self.intrinsics[4])  # Set focal length in x direction
        K[1, 1] = -(self.intrinsics[5])  # Set focal length in y direction
        K[0, 2] = self.intrinsics[2]  # Set x-coordinate of principal point
        K[1, 2] = self.intrinsics[3]  # Set y-coordinate of principal point
        K[2, 2] = 1

        return K

    def intrinsics_extrinsics_to_P(self):
        """Compute the camera projection matrix P and comprising matrices K, R and IC from intrinsic and extrinsic parameters.

        Returns:
            np.ndarray or cp.ndarray: A 4x4 camera projection matrix P that combines the intrinsic
            matrix K and the extrinsic parameters (rotation and translation).
            np.ndarray or cp.ndarray: The 3x3 intrinsic matrix K.
            np.ndarray or cp.ndarray: The 3x3 rotation matrix R derived from CIRN angles.
            np.ndarray or cp.ndarray: The 4x3 matrix IC that includes the identity matrix and
            the camera translation vector.

        Notes:
            Output P is normalized for homogenous coordinates. The projection matrix P is constructed as follows:
            - K is the intrinsic matrix built from camera parameters.
            - R is the rotation matrix calculated from the azimuth, tilt, and swing angles.
            - The extrinsics (translation) are applied as a column vector [-x, -y, -z].
            The resulting matrix P transforms 3D world coordinates into 2D image coordinates.
            If `self.use_gpu` is True, the function utilizes cupy for GPU processing; otherwise,
            it uses numpy for CPU processing. The last row of the projection matrix is normalized
            by dividing by P[2, 3].
        """
        # Generate K and R matrices
        K = self.build_K()
        R = self.CIRN_angles_to_R()

        x = self.extrinsics[0]
        y = self.extrinsics[1]
        z = self.extrinsics[2]

        if self.use_gpu:  # If GPU processing, create translation matrix with cupy
            column_vec = cp.array([-x, -y, -z]).reshape(-1, 1)
            IC = cp.concatenate([cp.eye(3), column_vec], axis=1)
            P = cp.dot(K, cp.dot(R, IC))  # Combine K, R, and IC into P

        else:  # Else, create translation matrix with numpy
            column_vec = np.array([-x, -y, -z]).reshape(-1, 1)
            IC = np.concatenate([np.eye(3), column_vec], axis=1)
            P = np.dot(K, np.dot(R, IC))  # Combine K, R, and IC into P

        P /= P[2, 3]  # Normalize for homogeneous coordinates

        return P, K, R, IC

    def distort_UV(self, UV):
        """Distort UV coordinates based on camera intrinsic parameters.

        Args:
            UV (np.ndarray or cp.ndarray): A 2xN array containing UV coordinates,
            where U is in the first row and V in the second.

        Returns:
            tuple: A tuple containing:
                - Ud (np.ndarray or cp.ndarray): The distorted U coordinates.
                - Vd (np.ndarray or cp.ndarray): The distorted V coordinates.
                - flag (np.ndarray or cp.ndarray): An array indicating which distorted coordinates
                exceed the image bounds or are invalid.

        Notes:
            The distortion is calculated using radial and tangential distortion models based on
            the camera's intrinsic parameters. The function applies corrections to the input UV
            coordinates and ensures that any resulting coordinates that fall outside the valid
            image boundaries are set to zero. The tangential distortion is also computed at the
            corners of the image for comparison.
            If `self.use_gpu` is True, the function utilizes cupy for GPU processing; otherwise,
            it uses numpy for CPU processing.
        """
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

        # Find negative UV coordinates
        flag_mask = (Ud < 0) | (Ud > NU) | (Vd < 0) | (Vd > NV)
        Ud[flag_mask] = 0
        Vd[flag_mask] = 0

        # Define maximum possible tangential distortion at the corners
        if self.use_gpu:
            Um = cp.array([0, 0, NU.item(), NU.item()])
            Vm = cp.array([0, NV.item(), NV.item(), 0])
        else:
            Um = np.array([0, 0, NU.item(), NU.item()])
            Vm = np.array([0, NV.item(), NV.item(), 0])

        # Normalization
        xm = (Um - c0U) / fx
        ym = (Vm - c0V) / fy
        r2m = xm**2 + ym**2

        # Tangential Distortion at corners
        dxm = 2 * t1 * xm * ym + t2 * (r2m + 2 * xm**2)
        dym = t1 * (r2m + 2 * ym**2) + 2 * t2 * xm * ym

        if self.use_gpu:
            # Find values larger than those at corners
            max_dym = cp.max(cp.abs(dym))
            max_dxm = cp.max(cp.abs(dxm))

            # Indices where distortion values are larger than those at corners
            exceeds_dy = cp.where(cp.abs(dy) > max_dym)
            exceeds_dx = cp.where(cp.abs(dx) > max_dxm)

            # Initialize flag array (assuming it’s previously defined)
            flag = cp.ones_like(Ud)
        else:
            # Find values larger than those at corners
            max_dym = np.max(np.abs(dym))
            max_dxm = np.max(np.abs(dxm))

            # Indices where distortion values are larger than those at corners
            exceeds_dy = np.where(np.abs(dy) > max_dym)
            exceeds_dx = np.where(np.abs(dx) > max_dxm)

            # Initialize flag array (assuming it’s previously defined)
            flag = np.ones_like(Ud)

        flag[exceeds_dy] = 0.0
        flag[exceeds_dx] = 0.0

        return Ud, Vd, flag

    def xyz_to_dist_UV(self):
        """Computes the distorted UV coordinates (UVd)  that correspond to a set of world xyz points
        for given extrinsics and intrinsics. Function also produces a flag variable to indicate
        if the UVd point is valid.

        Returns:
            tuple: A tuple containing:
                - DU (np.ndarray or cp.ndarray): The distorted U coordinates reshaped to the grid.
                - DV (np.ndarray or cp.ndarray): The distorted V coordinates reshaped to the grid.

        Notes:
            The function computes the projection of 3D coordinates onto the image plane using
            camera intrinsic and extrinsic parameters. It first transforms the 3D coordinates
            into homogeneous coordinates and then calculates the UV coordinates. Distortion is
            applied to these UV coordinates based on the camera's intrinsic parameters.
            The function also checks for negative Z camera coordinates and updates a flag to
            indicate valid points. If `self.use_gpu` is True, cupy is used for GPU processing;
            otherwise, numpy is used for CPU processing. The final distorted U and V coordinates
            are returned, multiplied by the flag to indicate valid points.
        """
        P, K, R, IC = self.intrinsics_extrinsics_to_P()  # Create matrix P

        xyz = self.reshape_grids()  # Reshape grids to single array of XYZ coordinates

        if self.use_gpu:
            xyz_homogeneous = cp.vstack((xyz.T, cp.ones(xyz.shape[0])))
            UV_homogeneous = cp.dot(P, xyz_homogeneous)
        else:
            xyz_homogeneous = np.vstack((xyz.T, np.ones(xyz.shape[0])))
            UV_homogeneous = np.dot(P, xyz_homogeneous)

        UV = UV_homogeneous[:2, :] / UV_homogeneous[2, :]
        Ud, Vd, flag = self.distort_UV(UV)
        DU = Ud.reshape(self.grid_shape, order="F")
        DV = Vd.reshape(self.grid_shape, order="F")

        # Compute camera coordinates
        if self.use_gpu:
            xyzC = cp.dot(cp.dot(R, IC), xyz_homogeneous)
            # Find negative Zc coordinates (Z <= 0) and update the flag
            negative_z_indices = cp.where(xyzC[2, :] <= 0.0)
        else:
            xyzC = np.dot(np.dot(R, IC), xyz_homogeneous)
            # Find negative Zc coordinates (Z <= 0) and update the flag
            negative_z_indices = np.where(xyzC[2, :] <= 0.0)

        flag[negative_z_indices] = 0.0
        flag = flag.reshape(self.grid_shape, order="F")

        return DU * flag, DV * flag

    def compute_Ud_Vd(self):
        """Compute the distorted U and V coordinates from 3D points.

        Returns:
            tuple: A tuple containing:
                - Ud (np.ndarray or cp.ndarray): The rounded and converted U coordinates.
                - Vd (np.ndarray or cp.ndarray): The rounded and converted V coordinates.

        Notes:
            This function computes the distorted U and V coordinates by calling the
            `xyz_to_dist_UV` method, which converts 3D coordinates to 2D distorted image
            coordinates. The resulting coordinates are then rounded to the nearest integer
            and cast to integers to represent pixel indices. The function uses either
            NumPy for CPU processing or CuPy for GPU processing based on the value of
            `self.use_gpu`.
        """
        Ud, Vd = self.xyz_to_dist_UV()  # Calculate distorted U and V coordinates

        # Round to the nearest integer to represent pixel indices
        if self.use_gpu:
            Ud = cp.round(Ud).astype(int)
            Vd = cp.round(Vd).astype(int)
        else:
            Ud = np.round(Ud).astype(int)
            Vd = np.round(Vd).astype(int)

        return Ud, Vd

    def merge_rectify(self, image_path, labels=False):
        """Merge and rectify an image based on the provided path.

        Args:
            image_path (str): The path to the image file to be loaded and rectified.
            labels (bool, optional): A flag indicating whether to load the image in grayscale (True)
                                    or in color (False). If True, the image will be loaded as a
                                    single-channel grayscale image; if False, the image will be
                                    loaded in RGB format. Defaults to False.

        Returns:
            np.ndarray or cp.ndarray: The rectified image as an array of type uint8.

        Notes:
        This function loads an image from the specified path using the `load_image` method,
        processes it to obtain rectified pixel values using the `get_pixels` method,
        and returns the rectified image. The function utilizes CuPy for GPU processing if
        `self.use_gpu` is set to True; otherwise, it uses NumPy for CPU processing. The
        output is cast to an array of type uint8 to represent pixel values.
        """
        image = self.load_image(image_path, labels)  # Load image from path

        rectified_image = self.get_pixels(
            image
        )  # Collect pixel values for real world coordinates

        if self.use_gpu:  # If GPU processing, return image as cupy array
            return cp.array(rectified_image, dtype=np.uint8)

        else:  # Else, return as numpy array
            return np.array(rectified_image, dtype=np.uint8)

    def merge_rectify_folder(self, folder_path, zarr_store_path, labels=False):
        """Merge and rectify all images in a specified folder and save them to a Zarr store.

        Args:
            folder_path (str): The path to the folder containing images to be rectified.
            zarr_store_path (str): The path to the Zarr store where rectified images will be saved.
            labels (bool, optional): A flag indicating whether to load the image in grayscale (True)
                                    or in color (False). If True, the image will be loaded as a
                                    single-channel grayscale image; if False, the image will be
                                    loaded in RGB format. Defaults to False.

        Returns:
            None: The function saves rectified images to the specified Zarr store but does not return any value.

        Notes:
            This function iterates over all image files in the specified folder, rectifies each
            image using the `merge_rectify` method, and saves the resulting rectified images to
            the specified Zarr store. The dataset names in the Zarr store are created by appending
            '_rectified' to the original image names. The Zarr store is opened in append mode before
            processing the images, allowing for multiple images to be saved efficiently.
            CuPy is utilized for GPU processing if `self.use_gpu` is set to True; otherwise,
            standard arrays are used.
        """
        # Open the Zarr store once before the loop
        store = zarr.open_group(zarr_store_path, mode="a")

        # Get a list of all images in the folder
        image_names = os.listdir(folder_path)

        for image_name in tqdm(
            image_names, desc="Processing images", unit="image"
        ):  # For each image in the provided folder path
            image_path = os.path.join(folder_path, image_name)
            rectified_image = self.merge_rectify(
                image_path, labels
            )  # Generate a rectified image

            # Create a dataset name by appending 'rectified' to the original image name
            dataset_name = f"{os.path.splitext(image_name)[0]}_rectified"

            # Save the rectified image array to the Zarr store
            if self.use_gpu:  # If GPU processing
                store[dataset_name] = (
                    rectified_image.get()
                )  # Port GPU array to CPU and save
            else:  # Else, save array
                store[dataset_name] = rectified_image

        # Print success message after all images are processed
        print("All images have been successfully saved to the Zarr store.")
