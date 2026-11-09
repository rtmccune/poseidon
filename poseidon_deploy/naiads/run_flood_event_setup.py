"""Stage 05: build one folder per flood event with its data and images.

Steps:

1. Optionally (``--hour_filter``) keep events starting within an
   Eastern-time hour window, writing ``--filtered_abbr_csv``.
2. Write each event's sensor records to ``<event>/<event>.csv``.
   Events with no records (confirmed only from images) get no CSV and
   are processed from their images alone.
3. Copy images and labels into ``<event>/<image_subfolder>`` and
   ``<event>/<label_subfolder>``.
4. Delete event folders that received no images.

Example
-------
``python run_flood_event_setup.py --abbr_events_csv abbr.csv
--full_sensor_csv flood_events.csv --image_dir images/daylight
--label_dir images/labels --output_dir flood_events``
"""

import logging
import os

from poseidon_utils import file_organizer

import cli_common

logger = logging.getLogger("poseidon.naiads")


def main(argv=None):
    """
    Organize flood event folders.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    parser = cli_common.make_parser(__doc__.splitlines()[0])
    parser.add_argument("--abbr_events_csv", required=True,
                        help="Abbreviated flood event CSV.")
    parser.add_argument("--full_sensor_csv", default=None,
                        help="Detailed sensor records CSV. Events without "
                             "records, or every event if omitted or "
                             "missing, are processed from images only.")
    parser.add_argument("--image_dir", required=True,
                        help="Folder of images to sort into events.")
    parser.add_argument("--label_dir", required=True,
                        help="Folder of label images to sort into events.")
    parser.add_argument("--output_dir", required=True,
                        help="Parent folder for the event folders.")
    parser.add_argument("--image_subfolder", default="orig_images",
                        help="Image subfolder name inside each event.")
    parser.add_argument("--label_subfolder", default="labels",
                        help="Label subfolder name inside each event.")
    parser.add_argument("--padding_hours", type=float, default=3,
                        help="Hours added before and after each event.")
    parser.add_argument("--hour_filter", action="store_true",
                        help="Drop events starting outside "
                             "--start_hour..--end_hour (Eastern time).")
    parser.add_argument("--filtered_abbr_csv", default=None,
                        help="Output of --hour_filter; default is "
                             "<abbr_events_csv stem>_filtered.csv.")
    parser.add_argument("--start_hour", type=int, default=6,
                        help="First Eastern hour kept (inclusive).")
    parser.add_argument("--end_hour", type=int, default=19,
                        help="First Eastern hour dropped (exclusive).")
    args = cli_common.start(parser, argv)

    events_csv = args.abbr_events_csv
    if args.hour_filter:
        stem, ext = os.path.splitext(args.abbr_events_csv)
        events_csv = args.filtered_abbr_csv or f"{stem}_filtered{ext}"
        file_organizer.filter_abbr_flood_csv_by_eastern_time(
            args.abbr_events_csv, events_csv, args.start_hour, args.end_hour)

    sensor_csv = args.full_sensor_csv
    if sensor_csv and not os.path.exists(sensor_csv):
        logger.warning("Sensor records CSV not found: %s; treating every "
                       "event as image-only.", sensor_csv)
        sensor_csv = None
    file_organizer.create_flood_csvs_and_subfolders(
        events_csv, sensor_csv, args.output_dir, args.padding_hours)
    for source, subfolder in ((args.image_dir, args.image_subfolder),
                              (args.label_dir, args.label_subfolder)):
        file_organizer.organize_images_into_flood_events(
            source, events_csv, args.output_dir, subfolder,
            args.padding_hours)
    file_organizer.prune_empty_event_folders(args.output_dir,
                                             args.image_subfolder)


if __name__ == "__main__":
    main()
