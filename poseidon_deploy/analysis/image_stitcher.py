"""Stitch original images beside their segmentation overlays.

For each image in ``orig_dir`` with an overlay named
``segmap_overlay_<image name>`` in ``overlay_dir`` (as written by the
``overlay_generator`` executable), saves
``comparison_<image name>`` in ``output_dir`` with the original on the
left and the overlay on the right.

Example
-------
``python image_stitcher.py images/daylight images/overlays
images/stitched_comparisons --workers 8``
"""

import argparse
import concurrent.futures
import logging
from pathlib import Path

import cv2
import numpy as np

from poseidon_utils.logging_utils import (
    progress_interval,
    setup_logging,
    should_report,
)

logger = logging.getLogger(__name__)

#: Image extensions considered for stitching.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

#: Prefix of overlay filenames written by overlay_generator.
OVERLAY_PREFIX = "segmap_overlay_"


def stitch_image_pair(orig_path, overlay_path, output_path):
    """
    Save an original image and its overlay side by side.

    The overlay is resized to the original's size if they differ.

    Parameters
    ----------
    orig_path, overlay_path : str or os.PathLike
        Input images.
    output_path : str or os.PathLike
        Output image path.

    Returns
    -------
    str or None
        An error message, or None on success.
    """
    original = cv2.imread(str(orig_path))
    overlay = cv2.imread(str(overlay_path))
    if original is None:
        return f"Could not read original image: {orig_path}"
    if overlay is None:
        return f"Could not read overlay image: {overlay_path}"

    if original.shape != overlay.shape:
        overlay = cv2.resize(overlay, (original.shape[1], original.shape[0]))
    if not cv2.imwrite(str(output_path), np.hstack((original, overlay))):
        return f"Failed to save stitched image: {output_path}"
    return None


def find_pairs(orig_dir, overlay_dir, output_dir):
    """
    Match original images to their overlays.

    Parameters
    ----------
    orig_dir, overlay_dir, output_dir : pathlib.Path
        Input and output folders.

    Returns
    -------
    list of tuple of pathlib.Path
        ``(original, overlay, output)`` paths for every match.
    """
    pairs = []
    for orig in sorted(orig_dir.iterdir()):
        if not (orig.is_file() and orig.suffix.lower() in IMAGE_EXTENSIONS):
            continue
        overlay = overlay_dir / f"{OVERLAY_PREFIX}{orig.name}"
        if overlay.exists():
            output = output_dir / f"comparison_{orig.name}"
            pairs.append((orig, overlay, output))
    return pairs


def parse_args(argv=None):
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. Default reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Stitch original images and overlays side by side.")
    parser.add_argument("orig_dir", type=Path, help="Original images.")
    parser.add_argument("overlay_dir", type=Path,
                        help="Overlays from overlay_generator.")
    parser.add_argument("output_dir", type=Path,
                        help="Folder for stitched comparisons.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes (default: all CPUs).")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Stitch every matched image pair in parallel.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 1 if an input folder is missing, else 0.
    """
    setup_logging()
    args = parse_args(argv)
    for folder in (args.orig_dir, args.overlay_dir):
        if not folder.is_dir():
            logger.error("Input folder not found: %s", folder)
            return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.orig_dir, args.overlay_dir, args.output_dir)
    if not pairs:
        logger.warning("No pairs found: overlays must be named '%s<original "
                       "name>' in %s.", OVERLAY_PREFIX, args.overlay_dir)
        return 0

    logger.info("Stitching %d pairs.", len(pairs))
    interval = progress_interval(len(pairs))
    failures = 0
    with concurrent.futures.ProcessPoolExecutor(args.workers) as executor:
        futures = [executor.submit(stitch_image_pair, *pair)
                   for pair in pairs]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            error = future.result()
            if error:
                failures += 1
                logger.error(error)
            if should_report(i, len(pairs), interval):
                logger.info("  %d/%d pairs done", i + 1, len(pairs))
    logger.info("Stitching complete: %d succeeded, %d failed.",
                len(pairs) - failures, failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
