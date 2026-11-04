"""Read and write Zarr groups stored in single-file ZIP archives.

POSEIDON keeps each event's rectified images, labels, and depth maps
in one ``.zip`` Zarr store per product to avoid creating hundreds of
thousands of small files (inodes) on HPC file systems.

ZIP archives cannot delete or replace members in place, so writers
here always build a fresh archive under a temporary name and move it
over the target only after every array has been written. Re-running a
pipeline stage therefore replaces stale outputs, and an interrupted
run never leaves a half-written archive behind.
"""

import os
import warnings
from contextlib import contextmanager
from pathlib import Path

import zarr


@contextmanager
def open_zip_group(path):
    """
    Open a ZIP Zarr store read-only and close it on exit.

    Parameters
    ----------
    path : str or os.PathLike
        Path to an existing ``.zip`` Zarr store.

    Yields
    ------
    zarr.Group
        Root group of the store.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr ZIP store not found: {path}")
    store = zarr.storage.ZipStore(path, mode="r")
    try:
        yield zarr.open_group(store=store, mode="r")
    finally:
        store.close()


@contextmanager
def write_zip_group(path):
    """
    Create a new ZIP Zarr store, replacing any existing file atomically.

    Arrays are written to ``<path>.tmp``; on a clean exit the temporary
    archive replaces `path`. If the body raises, the temporary file is
    deleted and any previous archive at `path` is left untouched.

    Parameters
    ----------
    path : str or os.PathLike
        Destination ``.zip`` path. Parent directories are created.

    Yields
    ------
    zarr.Group
        Empty, writable root group.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)

    store = zarr.storage.ZipStore(tmp_path, mode="w")
    try:
        yield zarr.open_group(store=store, mode="w")
    except BaseException:
        store.close()
        tmp_path.unlink(missing_ok=True)
        raise
    store.close()
    os.replace(tmp_path, path)


def write_array(group, name, data):
    """
    Write one array to a group as a single, unsplit chunk.

    Parameters
    ----------
    group : zarr.Group
        Writable group, e.g. from :func:`write_zip_group`.
    name : str
        Array name within the group.
    data : numpy.ndarray
        Array to store. CuPy arrays must be moved to the host first.

    Returns
    -------
    zarr.Array
        The created array.
    """
    with warnings.catch_warnings():
        # Fixed-width strings (timestamp arrays) work but are not yet in
        # the Zarr v3 spec; the warning is noise for POSEIDON's own stores
        warnings.filterwarnings("ignore", message=".*does not have a Zarr V3")
        return group.create_array(name, data=data, chunks=data.shape)


def array_names(group, suffix=None):
    """
    List array names in a group, optionally filtered by suffix.

    Parameters
    ----------
    group : zarr.Group
        Group to list.
    suffix : str or sequence of str, optional
        Keep only names ending with this suffix (or any of these).

    Returns
    -------
    list of str
        Sorted array names.
    """
    names = sorted(group.array_keys())
    if suffix is None:
        return names
    suffixes = (suffix,) if isinstance(suffix, str) else tuple(suffix)
    return [name for name in names if name.endswith(suffixes)]


def read_zip_arrays(path, names=None):
    """
    Load arrays from a ZIP Zarr store into memory.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the ``.zip`` Zarr store.
    names : sequence of str, optional
        Arrays to load. Default loads every array in the store.

    Returns
    -------
    dict of str to numpy.ndarray
        Mapping of array name to its data.
    """
    with open_zip_group(path) as group:
        names = array_names(group) if names is None else names
        return {name: group[name][:] for name in names}
