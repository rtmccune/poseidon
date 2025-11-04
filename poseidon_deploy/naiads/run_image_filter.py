import argparse
import poseidon_utils


def main():
    parser = argparse.ArgumentParser(description="Filter images.")

    parser.add_argument(
        "--drive",
        type=str,
        required=True,
        help="Root file path to the image archive drive.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Folder containing original images.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Destination folder for filtered images.",
    )
    parser.add_argument(
        "--start", type=int, default=6, help="Start time (Eastern)."
    )
    parser.add_argument(
        "--end", type=int, default=19, help="End time (Eastern)."
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Maximum worker threads."
    )

    args = parser.parse_args()

    handler = poseidon_utils.ImageHandler(sdfp_image_drive=args.drive)

    handler.copy_images_using_hour_window(
        image_dir=args.image_dir,
        destination_folder=args.dest,
        start_hour_east=args.start,
        end_hour_east=args.end,
        max_workers=args.workers,
    )
    print("Image filtering complete.")


if __name__ == "__main__":
    main()
