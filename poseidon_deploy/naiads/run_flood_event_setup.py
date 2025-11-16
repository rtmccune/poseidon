import argparse
import sys
from datetime import datetime

import poseidon_utils.file_organizer as file_organizer


def main():
    parser = argparse.ArgumentParser(
        description="Run the full flood event data organization pipeline."
    )

    # --- Path Arguments ---
    parser.add_argument(
        "--abbr_events_csv",
        type=str,
        required=True,
        help="Path to the initial abbreviated flood events CSV.",
    )
    parser.add_argument(
        "--filtered_abbr_csv",
        type=str,
        required=True,
        help="Path to save the intermediate, time-filtered abbreviated events CSV.",
    )
    parser.add_argument(
        "--full_sensor_csv",
        type=str,
        required=True,
        help="Path to the full sensor data CSV (all readings).",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Folder containing all original images to be sorted.",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Folder containing all labels to be sorted.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Main output directory to create event subfolders in.",
    )

    # --- Configuration Arguments ---
    parser.add_argument(
        "--image_subfolder",
        type=str,
        default="orig_images",
        help="Name of the subfolder inside each event folder to store images (default: 'orig_images').",
    )
    parser.add_argument(
        "--label_subfolder",
        type=str,
        default="labels",
        help="Name of the subfolder inside each event folder to store labels (default: 'labels').",
    )
    parser.add_argument(
        "--start_hour",
        type=int,
        default=6,
        help="Start hour (Eastern Time, inclusive) for filtering (default: 6).",
    )
    parser.add_argument(
        "--end_hour",
        type=int,
        default=19,
        help="End hour (Eastern Time, exclusive) for filtering (default: 19).",
    )
    parser.add_argument(
        "--padding_hours",
        type=int,
        default=3,
        help="Hours of padding to add before/after events (default: 3).",
    )

    args = parser.parse_args()

    # --- Pipeline Step 1: Filter Events by Time ---
    file_organizer.filter_abbr_flood_csv_by_eastern_time(
        input_path=args.abbr_events_csv,
        output_path=args.filtered_abbr_csv,
        min_hour=args.start_hour,
        max_hour=args.end_hour,
    )

    # --- Pipeline Step 2: Create Folders & Sensor CSVs ---
    file_organizer.create_flood_csvs_and_subfolders(
        abbr_events_path=args.filtered_abbr_csv,  # Use output of Step 1
        full_events_path=args.full_sensor_csv,
        output_parent_dir=args.output_dir,
        padding_hours=args.padding_hours,
    )

    # --- Pipeline Step 3: Organize Images into Folders ---
    file_organizer.organize_images_into_flood_events(
        image_folder=args.image_dir,
        csv_file=args.filtered_abbr_csv,  # Use filtered CSV from Step 1
        destination_folder=args.output_dir,  # Use same output dir as Step 2
        subfolder_name=args.image_subfolder,
        padding_hours=args.padding_hours,
    )
    
    # --- Pipeline Step 4: Organize Labels into Folders ---
    file_organizer.organize_images_into_flood_events(
        image_folder=args.label_dir,
        csv_file=args.filtered_abbr_csv,  # Use filtered CSV from Step 1
        destination_folder=args.output_dir,  # Use same output dir as Step 2
        subfolder_name=args.label_subfolder,
        padding_hours=args.padding_hours,
    )

    # --- Pipeline Step 5: Prune Empty Event Folders ---
    # NEW STEP: Clean up folders that have no images
    file_organizer.prune_empty_event_folders(
        output_parent_dir=args.output_dir,
        image_subfolder_name=args.image_subfolder,
    )

if __name__ == "__main__":
    main()
