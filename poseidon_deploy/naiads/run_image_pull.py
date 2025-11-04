import argparse
import poseidon_utils


def main():
    parser = argparse.ArgumentParser(description="Pull flood event images.")

    parser.add_argument(
        "--drive",
        type=str,
        required=True,
        help="Root file path to the image archive drive.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Destination folder for pulled images.",
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to the flood event CSV."
    )
    parser.add_argument(
        "--buffer", type=float, default=0, help="Time buffer in hours."
    )

    args = parser.parse_args()

    handler = poseidon_utils.ImageHandler(sdfp_image_drive=args.drive)

    handler.pull_flood_event_images(
        destination_folder=args.dest,
        flood_event_csv_path=args.csv,
        time_buffer_hours=args.buffer,
    )
    print("Image pull complete.")


if __name__ == "__main__":
    main()
