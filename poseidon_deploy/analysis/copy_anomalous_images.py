"""Copy the stitched comparison images of flagged anomalies.

For every ``<roi>_anomaly_report.csv`` written by
``find_stat_anomalies.py``, the stitched original/overlay images whose
filenames contain each anomaly's timestamp are copied into
``<dest_dir>/<roi>/`` for review.

Example
-------
``python copy_anomalous_images.py --reports_dir . --images_dir
images/stitched_comparisons --dest_dir images/anomaly_comparisons``
"""

import argparse
import glob
import logging
import os
import shutil

import pandas as pd

from poseidon_utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

#: Filename suffix of the anomaly reports.
REPORT_SUFFIX = "_anomaly_report.csv"


def copy_report_images(report_path, images_dir, dest_dir):
    """
    Copy the images matching every timestamp in one anomaly report.

    Parameters
    ----------
    report_path : str
        Anomaly report CSV with a ``Time`` column.
    images_dir : str
        Folder of stitched ``.jpg`` comparison images.
    dest_dir : str
        Parent output folder; images go to ``<dest_dir>/<roi>/``.

    Returns
    -------
    copied : int
        Images copied.
    missing : int
        Report timestamps with no matching image.
    """
    roi = os.path.basename(report_path).removesuffix(REPORT_SUFFIX)
    roi_dir = os.path.join(dest_dir, roi)
    os.makedirs(roi_dir, exist_ok=True)

    copied = missing = 0
    for time in pd.read_csv(report_path)["Time"]:
        stamp = pd.to_datetime(time).strftime("%Y%m%d%H%M%S")
        matches = glob.glob(os.path.join(images_dir, f"*{stamp}*.jpg"))
        if not matches:
            logger.warning("No image for %s (%s)", time, roi)
            missing += 1
        for match in matches:
            target = os.path.join(roi_dir, os.path.basename(match))
            if not os.path.exists(target):
                shutil.copy2(match, target)
                copied += 1
    logger.info("%s: copied %d images to %s", roi, copied, roi_dir)
    return copied, missing


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
        description="Copy stitched images for flagged anomalies.")
    parser.add_argument("--reports_dir", default=".",
                        help="Folder of *_anomaly_report.csv (default: .).")
    parser.add_argument("--images_dir",
                        default=os.path.join("images",
                                             "stitched_comparisons"),
                        help="Folder of stitched comparison images.")
    parser.add_argument("--dest_dir",
                        default=os.path.join("images", "anomaly_comparisons"),
                        help="Output parent folder.")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Copy anomaly images for every report found.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 1 if no reports were found, else 0.
    """
    setup_logging()
    args = parse_args(argv)
    reports = sorted(glob.glob(os.path.join(args.reports_dir,
                                            f"*{REPORT_SUFFIX}")))
    if not reports:
        logger.error("No anomaly reports in %s; run find_stat_anomalies.py "
                     "first.", args.reports_dir)
        return 1

    totals = [copy_report_images(report, args.images_dir, args.dest_dir)
              for report in reports]
    logger.info("Done: %d images copied, %d timestamps without images.",
                sum(t[0] for t in totals), sum(t[1] for t in totals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
