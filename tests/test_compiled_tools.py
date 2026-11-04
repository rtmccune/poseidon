"""Tests for the compiled C++ label and overlay generators.

Skipped unless the executables have been built into
``poseidon_utils/bin`` (see the README's C++ build instructions) or
the folder named by the ``POSEIDON_BIN`` environment variable.
"""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import REPO_ROOT

BIN = Path(os.environ.get("POSEIDON_BIN",
                          REPO_ROOT / "poseidon_utils" / "bin"))

#: Segmentation colors (hex RGB) and the class IDs they encode.
PALETTE = {"#3366CC": 1, "#DC3912": 2, "#FF9900": 3}


def bgr(hex_color):
    """
    Convert ``#RRGGBB`` to an OpenCV BGR tuple.

    Parameters
    ----------
    hex_color : str
        Color such as ``"#3366CC"``.

    Returns
    -------
    tuple of int
        ``(blue, green, red)``.
    """
    red, green, blue = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return blue, green, red


def tool(name):
    """
    Return a built executable's path or skip the test.

    Parameters
    ----------
    name : str
        Executable name.

    Returns
    -------
    pathlib.Path
        Executable path.
    """
    path = BIN / name
    if not path.exists():
        pytest.skip(f"{name} not built (see README: C++ Executables)")
    # Exit status 127 means the loader failed (e.g. OpenCV was upgraded)
    if subprocess.run([path], capture_output=True).returncode == 127:
        pytest.skip(f"{name} cannot start; rebuild it against the current "
                    f"OpenCV")
    return path


@pytest.fixture
def prediction(tmp_path):
    """
    Write a 30 x 30 prediction image with three color bands.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's per-test temporary directory.

    Returns
    -------
    pathlib.Path
        Folder holding ``scene_predseg.png``.
    """
    folder = tmp_path / "preds"
    folder.mkdir()
    image = np.zeros((30, 30, 3), np.uint8)  # black = background
    for i, color in enumerate(PALETTE):
        image[:, 10 * i:10 * i + 8] = bgr(color)
    cv2.imwrite(str(folder / "scene_predseg.png"), image)
    return folder


def test_label_generator_maps_colors_to_class_ids(prediction, tmp_path):
    """Each palette color becomes its class ID; others become 0."""
    (tmp_path / "list.txt").write_text("scene_predseg.png\n")
    subprocess.run([tool("pred_label_generator"), prediction,
                    tmp_path / "list.txt", tmp_path / "labels"], check=True,
                   capture_output=True)
    labels = cv2.imread(str(tmp_path / "labels" / "scene_predseg_labels.png"),
                        cv2.IMREAD_GRAYSCALE)
    for i, class_id in enumerate(PALETTE.values()):
        assert (labels[:, 10 * i:10 * i + 8] == class_id).all()
        assert (labels[:, 10 * i + 8:10 * i + 10] == 0).all()


def test_overlay_generator_blends_recolored_predictions(prediction,
                                                        tmp_path):
    """Water keeps its color and is alpha-blended over the image."""
    images = tmp_path / "images"
    images.mkdir()
    background = np.full((30, 30, 3), (200, 100, 50), np.uint8)
    cv2.imwrite(str(images / "scene.png"), background)
    (tmp_path / "list.txt").write_text("scene.png\n")

    subprocess.run([tool("overlay_generator"), images, prediction,
                    tmp_path / "overlays", tmp_path / "list.txt", "0.25"],
                   check=True, capture_output=True)
    overlay = cv2.imread(str(tmp_path / "overlays" /
                             "segmap_overlay_scene.png"))
    expected = 0.25 * np.array(bgr("#3366CC")) + 0.75 * background[0, 0]
    np.testing.assert_allclose(overlay[0, 0], expected, atol=1)
