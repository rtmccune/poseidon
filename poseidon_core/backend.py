"""Select NumPy (CPU) or CuPy (GPU) array backends at runtime.

CuPy and cuCIM are optional. When they are missing, or no usable CUDA
device is present, POSEIDON runs every stage on the CPU with NumPy,
SciPy, and scikit-image, which provide the same operations.
"""

import functools
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
except ImportError:  # CPU-only environment
    cp = None


@functools.lru_cache(maxsize=1)
def gpu_available():
    """
    Report whether CuPy can run kernels on a CUDA device.

    A test kernel is executed because an installed CuPy build may lack
    kernels for the local GPU architecture.

    Returns
    -------
    bool
        True if CuPy is importable and a kernel ran successfully.
    """
    if cp is None:
        return False
    try:
        return bool((cp.arange(3) + 1).sum() == 6)
    except Exception as err:  # noqa: BLE001 - any CUDA failure means no
        logger.warning("CuPy is installed but unusable: %s", err)
        return False


def resolve_use_gpu(use_gpu):
    """
    Turn a user GPU preference into a definite backend choice.

    Parameters
    ----------
    use_gpu : bool or None
        True requires the GPU, False forces the CPU, and None uses the
        GPU only when one is available.

    Returns
    -------
    bool
        True to use CuPy, False to use NumPy.

    Raises
    ------
    RuntimeError
        If `use_gpu` is True but no usable GPU is available.
    """
    if use_gpu is None:
        return gpu_available()
    if use_gpu and not gpu_available():
        raise RuntimeError("use_gpu=True but CuPy cannot run on this "
                           "machine; pass use_gpu=False to use the CPU.")
    return bool(use_gpu)


def get_xp(use_gpu):
    """
    Return the array module for the chosen backend.

    Parameters
    ----------
    use_gpu : bool
        True for CuPy, False for NumPy.

    Returns
    -------
    module
        ``cupy`` or ``numpy``.
    """
    return cp if use_gpu else np


def to_numpy(array):
    """
    Copy an array to host memory as a NumPy array.

    Parameters
    ----------
    array : numpy.ndarray or cupy.ndarray
        Input array on either device.

    Returns
    -------
    numpy.ndarray
        Host copy (or the input itself if already NumPy).
    """
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)
