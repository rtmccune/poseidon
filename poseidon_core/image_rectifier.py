"""Rectify oblique camera images onto a real-world elevation grid.

The camera model follows the CIRN (Coastal Imaging Research Network)
conventions: an 11-element intrinsics vector (image size, principal
point, focal lengths, radial and tangential distortion) and a 6-element
extrinsics vector (camera position, azimuth, tilt, swing). Each grid
cell's (X, Y, Z) world coordinate is projected once into distorted
image pixel coordinates; every image is then rectified by sampling the
pixel at each cell's projected location.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from poseidon_utils.logging_utils import progress_interval, should_report
from poseidon_utils.zarr_io import write_array, write_zip_group

from .backend import get_xp, resolve_use_gpu, to_numpy

logger = logging.getLogger(__name__)

#: File extensions treated as images by :meth:`merge_rectify_folder`.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class ImageRectifier:
    """
    Project grid cells into a camera image and sample their pixels.

    Parameters
    ----------
    intrinsics : array_like
        11-element CIRN intrinsics: ``[NU, NV, c0U, c0V, fx, fy, d1, d2,
        d3, t1, t2]`` (image columns, image rows, principal point,
        focal lengths, radial and tangential distortion).
    extrinsics : array_like
        6-element extrinsics: ``[x, y, z, azimuth, tilt, swing]`` in
        the grid CRS (meters) and radians.
    grid_x, grid_y, grid_z : array_like
        2-D world coordinate grids of identical ``(rows, cols)`` shape.
    use_gpu : bool or None, optional
        True to run on the GPU with CuPy, False for NumPy, or None to
        use the GPU only when available. Default is False.

    Attributes
    ----------
    use_gpu : bool
        Whether CuPy arrays are used.
    grid_shape : tuple of int
        ``(rows, cols)`` shape of the grids and of rectified images.
    u_distorted, v_distorted : numpy.ndarray or cupy.ndarray
        Integer image column (U) and row (V) sampled for each grid cell.
        Zero marks cells that do not project into the image.
    """

    def __init__(self, intrinsics, extrinsics, grid_x, grid_y, grid_z,
                 use_gpu=False):
        """
        Store the camera model and precompute the grid-to-pixel map.

        Parameters
        ----------
        intrinsics : array_like
            11-element CIRN intrinsics vector.
        extrinsics : array_like
            6-element extrinsics vector.
        grid_x, grid_y, grid_z : array_like
            2-D world coordinate grids of identical shape.
        use_gpu : bool or None, optional
            Backend choice; see the class docstring. Default is False.

        Raises
        ------
        ValueError
            If the parameter vectors or grid shapes are invalid.
        """
        self.use_gpu = resolve_use_gpu(use_gpu)
        xp = get_xp(self.use_gpu)

        # Camera parameters stay on the host; they are tiny scalars
        self.intrinsics = np.asarray(intrinsics, dtype=float)
        self.extrinsics = np.asarray(extrinsics, dtype=float)
        if self.intrinsics.shape != (11,):
            raise ValueError("intrinsics must have 11 elements")
        if self.extrinsics.shape != (6,):
            raise ValueError("extrinsics must have 6 elements")

        self.grid_x = xp.asarray(grid_x)
        self.grid_y = xp.asarray(grid_y)
        self.grid_z = xp.asarray(grid_z)
        if not self.grid_x.shape == self.grid_y.shape == self.grid_z.shape:
            raise ValueError("grid_x, grid_y and grid_z must share a shape")
        self.grid_shape = self.grid_x.shape

        self.azimuth, self.tilt, self.swing = self.extrinsics[3:6]

        logger.info("Initializing ImageRectifier (%s) for a %s grid.",
                    "GPU" if self.use_gpu else "CPU", self.grid_shape)
        self.u_distorted, self.v_distorted = self._compute_distorted_uv()

    def merge_rectify(self, image_path, labels=False):
        """
        Rectify one image onto the grid.

        Parameters
        ----------
        image_path : str or os.PathLike
            Image to rectify.
        labels : bool, optional
            True to read the image as single-channel grayscale (label
            masks), False to read it as RGB. Default is False.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            ``uint8`` array of shape ``(rows, cols, channels)`` with
            ``channels`` 1 for labels and 3 for RGB. Cells outside the
            camera view are 0.
        """
        image = self._load_image(image_path, labels)
        return self._get_pixels(image)

    def rectify_array(self, image):
        """
        Rectify an in-memory image or mask onto the grid.

        Uses the same grid-to-pixel lookup as :meth:`merge_rectify`, so
        a mask drawn on a camera image lands on exactly the grid cells
        that sample those pixels.

        Parameters
        ----------
        image : array_like
            ``(H, W)`` or ``(H, W, C)`` array the size of the camera
            image (``NV`` rows by ``NU`` columns).

        Returns
        -------
        numpy.ndarray
            ``uint8`` array of shape ``(rows, cols, C)`` in grid
            orientation (row 0 south); 0 outside the camera view.

        Raises
        ------
        ValueError
            If `image` is not 2-D or 3-D.
        """
        image = np.asarray(image)
        if image.ndim not in (2, 3):
            raise ValueError("image must be 2-D or 3-D")
        xp = get_xp(self.use_gpu)
        return to_numpy(self._get_pixels(xp.asarray(image)))

    def merge_rectify_folder(self, folder_path, zarr_store_path,
                             labels=False):
        """
        Rectify every image in a folder into one ZIP Zarr store.

        Each image is stored as ``<image stem>_rectified``. Any existing
        store is replaced once all images are written. Images that fail
        to load are logged and skipped.

        Parameters
        ----------
        folder_path : str or os.PathLike
            Folder of images to rectify.
        zarr_store_path : str or os.PathLike
            Output store. ``.zip`` is appended if not already present.
        labels : bool, optional
            Read images as grayscale label masks. Default is False.

        Returns
        -------
        int
            Number of images written to the store.
        """
        folder = Path(folder_path)
        zip_path = Path(zarr_store_path)
        if zip_path.suffix != ".zip":
            zip_path = zip_path.with_name(zip_path.name + ".zip")

        if not folder.is_dir():
            logger.error("Source folder not found: %s", folder)
            return 0
        image_paths = sorted(p for p in folder.iterdir()
                             if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_paths:
            logger.warning("No images found in %s; nothing written.", folder)
            return 0

        total = len(image_paths)
        interval = progress_interval(total)
        logger.info("Rectifying %d images from %s -> %s",
                    total, folder, zip_path)

        written = 0
        with write_zip_group(zip_path) as group:
            for i, path in enumerate(image_paths):
                if should_report(i, total, interval):
                    logger.info("  Image %d/%d: %s", i + 1, total, path.name)
                try:
                    rectified = self.merge_rectify(path, labels)
                except (OSError, ValueError) as err:
                    logger.error("  Skipping %s: %s", path.name, err)
                    continue
                write_array(group, f"{path.stem}_rectified",
                            to_numpy(rectified))
                written += 1

        logger.info("Wrote %d/%d rectified images to %s",
                    written, total, zip_path)
        return written

    def _load_image(self, image_path, labels):
        """
        Read an image from disk in the layout used for sampling.

        Parameters
        ----------
        image_path : str or os.PathLike
            Image file to read.
        labels : bool
            True for single-channel grayscale, False for RGB.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            ``(H, W)`` grayscale or ``(H, W, 3)`` RGB ``uint8`` array on
            the active backend.

        Raises
        ------
        FileNotFoundError
            If OpenCV cannot read the file.
        """
        flag = cv2.IMREAD_GRAYSCALE if labels else cv2.IMREAD_COLOR
        image = cv2.imread(str(image_path), flag)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: "
                                    f"{image_path}")
        if not labels:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return get_xp(self.use_gpu).asarray(image)

    def _reshape_grids(self):
        """
        Flatten the grids into an ``(N, 3)`` array of XYZ points.

        Points are ordered column-major (Fortran order) to match the
        reshape applied to the projected coordinates.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            ``(rows * cols, 3)`` array of world coordinates.
        """
        xp = get_xp(self.use_gpu)
        columns = [grid.T.reshape(-1, 1)
                   for grid in (self.grid_x, self.grid_y, self.grid_z)]
        return xp.concatenate(columns, axis=1)

    def _get_pixels(self, image):
        """
        Sample the image at each grid cell's projected pixel.

        Parameters
        ----------
        image : numpy.ndarray or cupy.ndarray
            ``(H, W)`` or ``(H, W, C)`` image on the active backend.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            ``uint8`` array of shape ``(rows, cols, C)``; cells that
            fall outside the image are 0.
        """
        xp = get_xp(self.use_gpu)
        if image.ndim == 2:
            image = image[:, :, xp.newaxis]  # treat gray as one channel
        height, width = image.shape[:2]

        cols, rows = self.u_distorted, self.v_distorted
        # Index 0 marks invalid projections; the first row/column is
        # also excluded, preserving the original implementation
        outside = (cols <= 1) | (cols >= width) | (rows <= 1) | (
            rows >= height)

        # Clamp masked cells to a valid index, sample, then blank them
        safe_rows = xp.where(outside, 0, rows)
        safe_cols = xp.where(outside, 0, cols)
        sampled = image[safe_rows, safe_cols].astype(np.uint8)
        sampled[outside] = 0
        return sampled

    def _cirn_angles_to_rotation(self):
        """
        Build the world-to-camera rotation from CIRN angles.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            3x3 rotation matrix for the azimuth, tilt, and swing.
        """
        ca, sa = np.cos(self.azimuth), np.sin(self.azimuth)
        ct, st = np.cos(self.tilt), np.sin(self.tilt)
        cs, ss = np.cos(self.swing), np.sin(self.swing)

        rotation = np.array([
            [-ca * cs - sa * ct * ss, cs * sa - ss * ct * ca, -ss * st],
            [-ss * ca + cs * ct * sa, ss * sa + cs * ct * ca, cs * st],
            [st * sa, st * ca, -ct],
        ])
        return get_xp(self.use_gpu).asarray(rotation)

    def _build_intrinsic_matrix(self):
        """
        Build the camera intrinsic matrix K.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            3x3 matrix with negated focal lengths on the diagonal and
            the principal point in the last column, as in CIRN.
        """
        c0_u, c0_v, fx, fy = self.intrinsics[2:6]
        intrinsic = np.array([
            [-fx, 0.0, c0_u],
            [0.0, -fy, c0_v],
            [0.0, 0.0, 1.0],
        ])
        return get_xp(self.use_gpu).asarray(intrinsic)

    def _projection_matrix(self):
        """
        Compose the 3x4 world-to-image projection matrix P = K R [I|-C].

        Returns
        -------
        projection : numpy.ndarray or cupy.ndarray
            3x4 projection normalized so ``projection[2, 3] == 1``.
        rotation : numpy.ndarray or cupy.ndarray
            3x3 world-to-camera rotation.
        translation : numpy.ndarray or cupy.ndarray
            3x4 ``[I | -C]`` matrix with C the camera position.
        """
        xp = get_xp(self.use_gpu)
        intrinsic = self._build_intrinsic_matrix()
        rotation = self._cirn_angles_to_rotation()

        camera_xyz = xp.asarray(-self.extrinsics[:3]).reshape(-1, 1)
        translation = xp.concatenate([xp.eye(3), camera_xyz], axis=1)

        projection = xp.dot(intrinsic, xp.dot(rotation, translation))
        projection /= projection[2, 3]
        return projection, rotation, translation

    def _distort_uv(self, uv):
        """
        Apply lens distortion to undistorted pixel coordinates.

        Parameters
        ----------
        uv : numpy.ndarray or cupy.ndarray
            ``(2, N)`` undistorted U (row 0) and V (row 1) coordinates.

        Returns
        -------
        u_dist, v_dist : numpy.ndarray or cupy.ndarray
            ``(N,)`` distorted coordinates; points outside the image
            are set to 0.
        valid : numpy.ndarray or cupy.ndarray
            ``(N,)`` 1.0 where the tangential distortion is physically
            plausible, 0.0 otherwise.
        """
        xp = get_xp(self.use_gpu)
        n_cols, n_rows, c0_u, c0_v, fx, fy, d1, d2, d3, t1, t2 = (
            self.intrinsics)

        # Normalized image-plane coordinates
        x = (uv[0, :] - c0_u) / fx
        y = (uv[1, :] - c0_v) / fy

        # Radial and tangential distortion
        r2 = x**2 + y**2
        radial = 1 + d1 * r2 + d2 * r2**2 + d3 * r2**3
        dx = 2 * t1 * x * y + t2 * (r2 + 2 * x**2)
        dy = t1 * (r2 + 2 * y**2) + 2 * t2 * x * y

        u_dist = (x * radial + dx) * fx + c0_u
        v_dist = (y * radial + dy) * fy + c0_v

        # Zero out points that land outside the image
        outside = (u_dist < 0) | (u_dist > n_cols) | (v_dist < 0) | (
            v_dist > n_rows)
        u_dist[outside] = 0
        v_dist[outside] = 0

        # Largest tangential distortion possible inside the image
        # occurs at a corner; anything larger is a model artifact
        corner_u = xp.array([0, 0, n_cols, n_cols])
        corner_v = xp.array([0, n_rows, n_rows, 0])
        xm = (corner_u - c0_u) / fx
        ym = (corner_v - c0_v) / fy
        r2m = xm**2 + ym**2
        dxm = 2 * t1 * xm * ym + t2 * (r2m + 2 * xm**2)
        dym = t1 * (r2m + 2 * ym**2) + 2 * t2 * xm * ym

        valid = xp.ones_like(u_dist)
        valid[xp.abs(dy) > xp.max(xp.abs(dym))] = 0.0
        valid[xp.abs(dx) > xp.max(xp.abs(dxm))] = 0.0
        return u_dist, v_dist, valid

    def _xyz_to_distorted_uv(self):
        """
        Project every grid cell to distorted pixel coordinates.

        Returns
        -------
        u_dist, v_dist : numpy.ndarray or cupy.ndarray
            Float coordinates shaped like the grid; 0 where the cell
            is behind the camera, outside the image, or implausibly
            distorted.
        """
        xp = get_xp(self.use_gpu)
        projection, rotation, translation = self._projection_matrix()

        xyz = self._reshape_grids()
        xyz_homogeneous = xp.vstack((xyz.T, xp.ones(xyz.shape[0])))

        # Project to the image plane and perspective-divide
        uv_homogeneous = xp.dot(projection, xyz_homogeneous)
        uv = uv_homogeneous[:2, :] / uv_homogeneous[2, :]

        u_dist, v_dist, valid = self._distort_uv(uv)

        # Points with non-positive camera depth are behind the camera
        xyz_camera = xp.dot(xp.dot(rotation, translation), xyz_homogeneous)
        valid[xyz_camera[2, :] <= 0.0] = 0.0

        shape = self.grid_shape
        valid = valid.reshape(shape, order="F")
        return (u_dist.reshape(shape, order="F") * valid,
                v_dist.reshape(shape, order="F") * valid)

    def _compute_distorted_uv(self):
        """
        Compute the integer pixel indices sampled for each grid cell.

        Returns
        -------
        u_distorted, v_distorted : numpy.ndarray or cupy.ndarray
            Rounded integer column and row indices shaped like the grid;
            0 (invalid) where the grid elevation is NaN.
        """
        xp = get_xp(self.use_gpu)
        u_dist, v_dist = self._xyz_to_distorted_uv()
        # Cells without LiDAR coverage project to NaN; casting NaN to
        # int is undefined, so mark them invalid explicitly
        return tuple(xp.round(xp.nan_to_num(coord, nan=0.0)).astype(int)
                     for coord in (u_dist, v_dist))
