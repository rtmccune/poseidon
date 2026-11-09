"""List the images whose segmentations have not been scored yet.

Compares the image names in a folder with a segmentation review CSV
(``Filename`` and ``Segmentation_Score`` columns, as written by the
image reviewer). Review filenames are the stitched comparison names,
``comparison_<image name>``; the prefix is removed before matching.
Writes the unreviewed image names, one per line, and logs counts for
each folder.

Compare against the daylight folder: overlays and comparisons are made
only for daylight images, so night images in ``all_events_during_FOV``
never reach the reviewer.

Example
-------
From the repository root, with the default folders for a site::

    python poseidon_deploy/analysis/list_unreviewed_images.py \\
        --site down_east --labels data/segmentation_labels.csv

writes ``data/down_east/images_to_review.txt`` from
``data/down_east/images/daylight_all_events`` and also reports counts
for ``all_events_during_FOV``.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from poseidon_utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

#: Repository root (this file is in poseidon_deploy/analysis/).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Image extensions considered.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

#: Prefix the stitcher adds to comparison filenames.
COMPARISON_PREFIX = "comparison_"


def reviewed_names(labels_path, prefix=COMPARISON_PREFIX):
    """
    Read the original image names already scored in a review CSV.

    Parameters
    ----------
    labels_path : str or os.PathLike
        CSV with a ``Filename`` column.
    prefix : str, optional
        Prefix removed from each filename. Default ``"comparison_"``.

    Returns
    -------
    set of str
        Image names without the prefix.
    """
    labels = pd.read_csv(labels_path, dtype={"Filename": str})
    if "Filename" not in labels.columns:
        raise ValueError(f"{labels_path} has no 'Filename' column")
    # strip() also removes stray carriage returns from mixed line endings
    names = labels["Filename"].dropna().str.strip()
    return set(names.str.removeprefix(prefix))


def image_names(folder):
    """
    List the image filenames directly inside a folder.

    Parameters
    ----------
    folder : str or os.PathLike
        Image folder.

    Returns
    -------
    set of str
        Image filenames.
    """
    return {p.name for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}


def unreviewed_images(image_dir, reviewed):
    """
    Return the sorted names in ``image_dir`` that are not in ``reviewed``.

    Parameters
    ----------
    image_dir : str or os.PathLike
        Image folder.
    reviewed : set of str
        Image names already reviewed.

    Returns
    -------
    list of str
        Unreviewed image names, sorted (chronological for SDFP names).
    """
    return sorted(image_names(image_dir) - reviewed)


def report(folder, reviewed):
    """
    Log how many of a folder's images have been reviewed.

    Parameters
    ----------
    folder : pathlib.Path
        Image folder.
    reviewed : set of str
        Image names already reviewed.
    """
    names = image_names(folder)
    logger.info("%s: %d images, %d reviewed, %d to review, "
                "%d reviewed but not in this folder",
                folder, len(names), len(names & reviewed),
                len(names - reviewed), len(reviewed - names))


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
        Parsed arguments, with site defaults filled in.
    """
    parser = argparse.ArgumentParser(
        description="List images not yet scored in a segmentation "
                    "review CSV.")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Review CSV with a Filename column.")
    parser.add_argument("--site",
                        help="Site name; sets the default folders under "
                             "data/<site>/.")
    parser.add_argument("--image_dir", type=Path,
                        help="Images to check (default: "
                             "data/<site>/images/daylight_all_events).")
    parser.add_argument("--output", type=Path,
                        help="Text file of unreviewed names (default: "
                             "data/<site>/images_to_review.txt).")
    parser.add_argument("--compare_dir", type=Path, action="append",
                        help="Extra folder to report counts for; may be "
                             "repeated (default with --site: "
                             "images/all_events_during_FOV).")
    parser.add_argument("--prefix", default=COMPARISON_PREFIX,
                        help="Prefix removed from review filenames "
                             "(default: %(default)s).")
    args = parser.parse_args(argv)

    if args.site:
        site_data = REPO_ROOT / "data" / args.site
        args.image_dir = args.image_dir or (
            site_data / "images" / "daylight_all_events")
        args.output = args.output or site_data / "images_to_review.txt"
        if args.compare_dir is None:
            args.compare_dir = [site_data / "images"
                                / "all_events_during_FOV"]
    if args.image_dir is None or args.output is None:
        parser.error("give --site, or both --image_dir and --output")
    args.compare_dir = args.compare_dir or []
    return args


def main(argv=None):
    """
    Write the unreviewed image list.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 1 if an input is missing, else 0.
    """
    setup_logging()
    args = parse_args(argv)
    if not args.labels.is_file():
        logger.error("Review CSV not found: %s", args.labels)
        return 1
    if not args.image_dir.is_dir():
        logger.error("Image folder not found: %s", args.image_dir)
        return 1

    reviewed = reviewed_names(args.labels, args.prefix)
    logger.info("%s: %d reviewed images", args.labels, len(reviewed))
    for folder in args.compare_dir:
        if folder.is_dir():
            report(folder, reviewed)
        else:
            logger.warning("Skipping missing folder: %s", folder)
    report(args.image_dir, reviewed)

    todo = unreviewed_images(args.image_dir, reviewed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{name}\n" for name in todo))
    logger.info("Wrote %d unreviewed image names to %s", len(todo),
                args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
