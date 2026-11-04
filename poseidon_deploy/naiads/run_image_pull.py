"""Stage 00: copy archived camera images taken during flood events.

Reads the abbreviated flood event CSV and copies every archive image
within each event window (plus an optional buffer) into one folder.

Example
-------
``python run_image_pull.py --drive /archive --csv abbr_flood_events.csv
--dest images/all_events --buffer 3``
"""

from poseidon_utils.image_handler import ImageHandler

import cli_common


def main(argv=None):
    """
    Pull flood event images from the SDFP archive.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    parser = cli_common.make_parser(__doc__.splitlines()[0])
    parser.add_argument("--drive", required=True,
                        help="Root of the SDFP image archive.")
    parser.add_argument("--csv", required=True,
                        help="Abbreviated flood event CSV.")
    parser.add_argument("--dest", required=True,
                        help="Folder receiving the images.")
    parser.add_argument("--buffer", type=float, default=0,
                        help="Hours added before and after each event.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Copy threads.")
    args = cli_common.start(parser, argv)

    ImageHandler(args.drive).pull_flood_event_images(
        destination_folder=args.dest, flood_event_csv_path=args.csv,
        time_buffer_hours=args.buffer, max_workers=args.workers)


if __name__ == "__main__":
    main()
