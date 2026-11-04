"""Tests for naming, zarr_io, site_config, constants, and backend."""

import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from poseidon_core import backend
from poseidon_utils import constants, naming, zarr_io
from poseidon_utils.logging_utils import progress_interval, should_report
from poseidon_utils.site_config import (
    GridExtent,
    load_site_config,
    site_from_dict,
)

# ----- naming ----------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "CAM_CB_03_20231002131826.jpg",
    "CAM_CB_03_20231002131826_predseg_labels.png",
    "/some/dir/CAM_CB_03_20231002131826_rectified",
])
def test_names_parse_camera_and_utc_timestamp(name):
    """Camera, timestamp, and aware UTC time parse from SDFP names."""
    assert naming.extract_camera_name(name) == "CAM_CB_03"
    assert naming.extract_timestamp(name) == "20231002131826"
    assert naming.parse_timestamp(name) == datetime(
        2023, 10, 2, 13, 18, 26, tzinfo=timezone.utc)


def test_names_without_parts_return_none():
    """Missing or impossible parts give None rather than raising."""
    assert naming.extract_camera_name("photo_20231002131826.jpg") is None
    assert naming.parse_timestamp("CAM_CB_03.jpg") is None
    assert naming.parse_timestamp("CAM_CB_03_20231302131826.jpg") is None


def test_sensor_and_camera_ids_round_trip():
    """Sensor IDs and camera names convert in both directions."""
    assert naming.camera_from_sensor_id("DE_01") == "CAM_DE_01"
    assert naming.sensor_id_from_camera("CAM_DE_01") == "DE_01"


def test_feet_to_meters_is_exact_and_est_has_no_dst():
    """The conversion is exact; EST stays UTC-5 in summer."""
    assert constants.FEET_TO_METERS == 0.3048
    july = datetime(2024, 7, 1, 12, tzinfo=timezone.utc)
    assert july.astimezone(constants.EST).hour == 7
    assert july.astimezone(constants.EASTERN_TZ).hour == 8


# ----- logging helpers -------------------------------------------------------


def test_progress_reporting_hits_first_every_interval_and_last():
    """Progress logs on the first item, each interval, and the last."""
    total = 25
    interval = progress_interval(total)
    reported = [i for i in range(total) if should_report(i, total, interval)]
    assert interval == 2
    assert reported[0] == 0 and reported[-1] == total - 1
    assert progress_interval(3) == 1


# ----- zarr_io ---------------------------------------------------------------


def test_zip_store_round_trip(tmp_path):
    """Arrays written to a ZIP store read back unchanged."""
    path = tmp_path / "store.zip"
    data = {"a_depth_map_95_perc": np.arange(12.0).reshape(3, 4),
            "b_wse_map_95_perc": np.ones((2, 2), dtype=np.float32)}
    with zarr_io.write_zip_group(path) as group:
        for name, array in data.items():
            zarr_io.write_array(group, name, array)

    loaded = zarr_io.read_zip_arrays(path)
    assert loaded.keys() == data.keys()
    for name, array in data.items():
        np.testing.assert_array_equal(loaded[name], array)
        assert loaded[name].dtype == array.dtype
    with zarr_io.open_zip_group(path) as group:
        assert zarr_io.array_names(group, suffix="depth_map_95_perc") == [
            "a_depth_map_95_perc"]


def test_rewriting_a_zip_store_replaces_its_contents(tmp_path):
    """A second write replaces stale arrays (re-running a stage)."""
    path = tmp_path / "store.zip"
    for name, value in (("old", 1.0), ("new", 2.0)):
        with zarr_io.write_zip_group(path) as group:
            zarr_io.write_array(group, name, np.full((2, 2), value))

    loaded = zarr_io.read_zip_arrays(path)
    assert list(loaded) == ["new"]
    assert not (tmp_path / "store.zip.tmp").exists()


def test_failed_write_keeps_the_previous_store(tmp_path):
    """An error mid-write leaves the old archive and no temp file."""
    path = tmp_path / "store.zip"
    with zarr_io.write_zip_group(path) as group:
        zarr_io.write_array(group, "good", np.zeros(3))

    with pytest.raises(RuntimeError):
        with zarr_io.write_zip_group(path) as group:
            zarr_io.write_array(group, "partial", np.ones(3))
            raise RuntimeError("simulated crash")

    assert list(zarr_io.read_zip_arrays(path)) == ["good"]
    assert not (tmp_path / "store.zip.tmp").exists()


def test_opening_a_missing_store_raises(tmp_path):
    """Reading a nonexistent store raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        zarr_io.read_zip_arrays(tmp_path / "missing.zip")


# ----- site_config -----------------------------------------------------------


@pytest.mark.parametrize("site", ["carolina_beach", "down_east"])
def test_shipped_site_configs_are_valid(site):
    """Both site YAML files load with correctly shaped parameters."""
    config = load_site_config(site)
    assert config.name == site
    assert config.intrinsics.shape == (11,)
    assert config.extrinsics.shape == (6,)
    assert config.virtual_sensors.shape[1] == 2
    assert config.camera_name.startswith("CAM_")
    assert config.lidar_units in ("meters", "feet")

    # Virtual sensors must fall inside the configured grid
    grid = config.grid
    n_cols = round((grid.max_x - grid.min_x) / grid.resolution)
    n_rows = round((grid.max_y - grid.min_y) / grid.resolution)
    cols, rows = config.virtual_sensors.T
    assert (cols < n_cols).all() and (rows < n_rows).all()


def _site_dict(**overrides):
    """
    Return a minimal valid site mapping with optional overrides.

    Parameters
    ----------
    **overrides
        Keys to replace in the mapping.

    Returns
    -------
    dict
        Site mapping accepted by ``site_from_dict``.
    """
    data = {
        "name": "test_site", "sensor_id": "TS_01", "crs": "EPSG:32119",
        "grid": {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 5,
                 "resolution": 0.5},
        "intrinsics": list(range(11)), "extrinsics": list(range(6)),
    }
    data.update(overrides)
    return data


def test_site_paths_resolve_against_the_repo_root(tmp_path):
    """Relative paths resolve under a root; absolute paths are kept."""
    site = site_from_dict(_site_dict(lidar_file="data/x.laz",
                                     basemap=str(tmp_path / "b.tif")))
    assert site.resolve(site.lidar_file, root=tmp_path) == (
        tmp_path / "data" / "x.laz")
    assert site.resolve(site.basemap) == tmp_path / "b.tif"
    assert site.resolve(None) is None


@pytest.mark.parametrize("overrides, error", [
    ({"intrinsics": [1, 2, 3]}, "intrinsics"),
    ({"extrinsics": [1, 2]}, "extrinsics"),
    ({"lidar_units": "yards"}, "lidar_units"),
    ({"grid": {"min_x": 5, "max_x": 1, "min_y": 0, "max_y": 1,
               "resolution": 1}}, "minima"),
])
def test_invalid_site_values_are_rejected(overrides, error):
    """Malformed configs fail loudly with a helpful message."""
    with pytest.raises(ValueError, match=error):
        site_from_dict(_site_dict(**overrides))


def test_unknown_site_lists_available_sites():
    """Asking for a missing site names the sites that exist."""
    with pytest.raises(FileNotFoundError, match="carolina_beach"):
        load_site_config("atlantis")


def test_grid_extent_rejects_nonpositive_resolution():
    """A zero grid spacing is invalid."""
    with pytest.raises(ValueError, match="resolution"):
        GridExtent(0, 1, 0, 1, 0)


# ----- backend ---------------------------------------------------------------


def test_cpu_backend_uses_numpy():
    """Forcing the CPU selects NumPy and passes arrays through."""
    assert backend.resolve_use_gpu(False) is False
    assert backend.get_xp(False) is np
    array = np.arange(3)
    assert backend.to_numpy(array) is array


def test_requiring_a_missing_gpu_raises(monkeypatch):
    """use_gpu=True without a GPU raises; None falls back to the CPU."""
    monkeypatch.setattr(backend, "gpu_available", lambda: False)
    with pytest.raises(RuntimeError, match="use_gpu=False"):
        backend.resolve_use_gpu(True)
    assert backend.resolve_use_gpu(None) is False


def test_processing_works_without_gpu_or_plotting_libraries():
    """Core stages import and run with CuPy, cuCIM, rioxarray absent."""
    code = "\n".join([
        "import sys",
        "for name in ('cupy', 'cupyx', 'cucim', 'rioxarray'):",
        "    sys.modules[name] = None  # make 'import name' fail",
        "import numpy as np",
        "from poseidon_core import (DepthMapProcessor, GridGenerator,",
        "                           ImageRectifier, RoadwayAnalyzer)",
        "labels = np.zeros((20, 20), np.uint8); labels[5:15, 5:15] = 1",
        "processor = DepthMapProcessor(np.ones((20, 20)), plot_edges=False)",
        "maps = processor.process_single_depth_map(labels, 'x')",
        "print(processor.use_gpu, np.nanmax(maps['x_wse_map_mean']))",
    ])
    result = subprocess.run([sys.executable, "-c", code], check=True,
                            capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.stdout.split() == ["False", "1.0"]


def test_library_loggers_have_no_handlers():
    """Library modules leave log handling to the entry point."""
    import poseidon_core.image_rectifier  # noqa: F401
    import poseidon_utils.file_organizer  # noqa: F401

    for name in ("poseidon_core.image_rectifier",
                 "poseidon_utils.file_organizer"):
        assert logging.getLogger(name).handlers == []
