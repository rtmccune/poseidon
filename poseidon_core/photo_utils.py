import os
import re
import pytz
import cv2
import shutil
import random
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageColor


def list_files_in_date_directory(base_dir, date, camera_id):
    """List all files in a directory for a specific date."""
    # Set directory to match provided date
    directory = os.path.join(
        base_dir, f"{date.year} Archive", camera_id, date.strftime("%Y-%m-%d")
    )

    # Initialize empty file list
    file_list = []

    # If folder for provided day exists in research storage
    if os.path.exists(directory):
        # walk through directory folder
        for root, _, files in os.walk(directory):
            # and append each file to the file list
            for file in files:
                file_list.append(os.path.join(root, file))
    return file_list


def filter_files(file_list, start_time, end_time):
    """Filter files within the specified time range and camera ID."""
    # Initialize empty list to store filtered files
    filtered_files = []

    # For each file in provided list
    for file in file_list:
        try:
            # strip timestamp from file name
            file_name = os.path.basename(file)
            file_timestamp_str = file_name.split("_")[3].split(".")[0]
            file_timestamp = datetime.strptime(file_timestamp_str, "%Y%m%d%H%M%S")

            # and localize to time aware UTC
            utc_timezone = pytz.utc
            file_timestamp = utc_timezone.localize(file_timestamp)

            # then add to filtered file list if it is between the provided start and end time
            if start_time <= file_timestamp <= end_time:
                filtered_files.append(file)
        except (IndexError, ValueError):
            continue
    return filtered_files


def copy_file(file_path, destination_folder):
    """Copy a file to the destination folder."""
    if os.path.exists(file_path):
        shutil.copy(file_path, destination_folder)
        # print(f"Successfully copied: {file_path}")


def pull_files(df, base_dir, destination_folder):
    # Create destination folder if it does not already exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    tasks = []

    # Progress bar for processing rows
    with tqdm(total=df.shape[0], desc="Processing rows") as pbar:
        # For each row in the dataframe
        for _, row in df.iterrows():
            # set the start time, end time, and camera ID
            start_time = row["start_time_UTC"]
            end_time = row["end_time_UTC"]
            camera_id = row["camera_ID"]

            # set the current time to begin iterating through folders
            current_date = start_time.date()
            #  set the end date (possible that the flood event occurred over more than one date)
            end_date = end_time.date()

            # While the current iteration of the date is earlier than the end date
            while current_date <= end_date:
                # list the files in the current date iteration's corresponding folder
                file_list = list_files_in_date_directory(
                    base_dir, current_date, camera_id
                )

                # filter the file list to between the start and end time
                # (using end time rather than end date accounts for multiday events)
                filtered_files = filter_files(file_list, start_time, end_time)

                # add the file paths and destination folder to the task list to be copied
                for file_path in filtered_files:
                    tasks.append((file_path, destination_folder))

                # continue to the next day
                current_date += timedelta(days=1)

            pbar.update(1)

    # Progress bar for copying files
    with tqdm(total=len(tasks), desc="Copying files") as pbar:
        # copy the files in the task list using the copy function
        for file_path, destination_folder in tasks:
            copy_file(file_path, destination_folder)
            pbar.update(1)


def list_files_in_image_directory(image_dir):
    """List all non-hidden files in image directory."""

    # Initialize empty file list
    file_list = []

    # If folder for provided day exists in research storage
    if os.path.exists(image_dir):
        # walk through directory folder
        for root, _, files in os.walk(image_dir):
            # and append each non-hidden file to the file list
            for file in files:
                # Check if the file is not hidden (does not start with a dot)
                if not file.startswith("."):
                    file_list.append(os.path.join(root, file))
    return file_list


def extract_timestamp(filename):
    """Extracts UTC timestamp from filenames."""
    # Regular expression pattern to match the UTC timestamp (format: YYYYMMDDHHMMSS)
    pattern = r"\d{14}"

    match = re.search(pattern, filename)
    return match.group(0) if match else None


def filter_images_by_daylight(file_list):
    # Extract timestamps
    timestamps = [extract_timestamp(filename) for filename in file_list]

    # Convert timestamps to datetime objects
    utc_times = [datetime.strptime(ts, "%Y%m%d%H%M%S") for ts in timestamps]

    # Define the time zone for Eastern Time
    eastern = pytz.timezone("US/Eastern")

    filtered_dataframe = pd.DataFrame(
        {
            "filename": [
                file_list[i]
                for i, utc_time in enumerate(utc_times)
                if pytz.utc.localize(utc_time).astimezone(eastern).hour in range(6, 18)
            ]
        }
    )

    return filtered_dataframe


def copy_daylight_images(img_directory):

    file_list = list_files_in_image_directory(img_directory)

    filtered_dataframe = filter_images_by_daylight(file_list)

    destination_folder = os.path.join(img_directory, "daylight_images")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Progress bar for copying files
    with tqdm(total=len(filtered_dataframe), desc="Copying files") as pbar:
        # copy the files in the task list using the copy function
        for index, row in filtered_dataframe.iterrows():
            copy_file(row["filename"], destination_folder)
            pbar.update(1)


def extract_sensor_name(filename):
    """Extracts sensor ID from filenames."""
    # Regular expression pattern to match the sensor name
    pattern = r"CAM_[A-Z]{2}_[0-9]{2}"

    match = re.search(pattern, filename)
    return match.group(0) if match else None


def gen_image_and_label_dataframes(img_dir, labels_dir):

    image_list = list_files_in_image_directory(img_dir)
    labeled_image_list = list_files_in_image_directory(labels_dir)

    # Extract timestamps
    image_timestamps = [extract_timestamp(filename) for filename in image_list]
    label_timestamps = [extract_timestamp(filename) for filename in labeled_image_list]

    # Extract sensor IDs
    image_sensor_ids = [extract_sensor_name(filename) for filename in image_list]
    label_sensor_ids = [
        extract_sensor_name(filename) for filename in labeled_image_list
    ]

    image_df = pd.DataFrame(
        {
            "sensor": [
                image_sensor_ids[i] for i, sensor_id in enumerate(image_sensor_ids)
            ],
            "timestamp": [
                image_timestamps[i] for i, timestamp in enumerate(image_timestamps)
            ],
        }
    )

    label_df = pd.DataFrame(
        {
            "sensor": [
                label_sensor_ids[i] for i, sensor_id in enumerate(label_sensor_ids)
            ],
            "timestamp": [
                label_timestamps[i] for i, timestamp in enumerate(label_timestamps)
            ],
        }
    )

    return image_df, label_df


def gen_unlabeled_images_folder(img_dir, labels_dir):

    image_df, label_df = gen_image_and_label_dataframes(img_dir, labels_dir)

    # Perform a left join with an indicator to mark rows that match
    merged_df = image_df.merge(
        label_df, on=["sensor", "timestamp"], how="left", indicator=True
    )

    # Filter out rows that have a match in label_df
    filtered_dataframe = merged_df[merged_df["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )

    image_list = list_files_in_image_directory(img_dir)

    destination_folder = os.path.join(img_dir, "filtered_images")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Progress bar for copying files
    with tqdm(total=len(filtered_dataframe), desc="Copying files") as pbar:
        # copy the files in the file list using the copy function
        for index, row in filtered_dataframe.iterrows():
            copy_file(image_list[index], destination_folder)
            pbar.update(1)


def remove_labeled_images(img_dir, labels_dir):

    image_df, label_df = gen_image_and_label_dataframes(img_dir, labels_dir)

    # Perform a left join with an indicator to mark rows that match
    merged_df = image_df.merge(
        label_df, on=["sensor", "timestamp"], how="left", indicator=True
    )

    # Filter out rows that have a match in label_df
    filtered_dataframe = merged_df[merged_df["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )

    image_list = list_files_in_image_directory(img_dir)

    # Progress bar for removing files
    with tqdm(total=len(filtered_dataframe), desc="Copying files") as pbar:
        # remove the files in the file list using the copy function
        for index, row in filtered_dataframe.iterrows():
            os.remove(image_list[index])
            pbar.update(1)


def gen_date_list(start_year, end_year):
    all_dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    date = datetime(year, month, day)
                    all_dates.append(date)
                except ValueError:
                    continue

    return all_dates


def get_random_images(base_dir, camera_id, num_images, start_year, end_year):
    all_dates = gen_date_list(start_year, end_year)

    random.shuffle(all_dates)
    selected_images = []

    while len(selected_images) < num_images and all_dates:
        date = all_dates.pop()
        date_dir = os.path.join(
            base_dir, f"{date.year} Archive", camera_id, date.strftime("%Y-%m-%d")
        )

        if os.path.exists(date_dir):
            # Get all image files from the directory
            images = list_files_in_image_directory(date_dir)

            # Filter out the daytime images
            filtered_images = filter_images_by_daylight(images)["filename"].tolist()
            random.shuffle(filtered_images)

            # Random percentage (1% to 10%) of filtered images from this date's folder
            percentage = random.uniform(0.01, 0.05)
            num_to_select = max(1, int(num_images * percentage))

            selected_images.extend(filtered_images[:num_to_select])
            selected_images = selected_images[
                :num_images
            ]  # Ensure we don't exceed the desired number

    if len(selected_images) < num_images:
        print(
            f"Warning: Only found {len(selected_images)} images instead of the requested {num_images} images."
        )

    return selected_images


def create_test_image_set(
    base_dir, destination_folder, camera_id, num_images, start_year=2022, end_year=2023
):

    random_images = get_random_images(
        base_dir, camera_id, num_images, start_year, end_year
    )

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Progress bar for copying files
    with tqdm(total=len(random_images), desc="Copying files") as pbar:
        # copy the files in the file list using the copy function
        for img in random_images:
            copy_file(img, destination_folder)
            pbar.update(1)


def orig_rgb_to_gray_labels(rgb_image, color_map):
    """
    Converts an RGB image to a grayscale array representing labels based on the given color map.
    :param rgb_image: numpy array
        RGB image array.
    :param color_map: dict
        Color map where RGB colors are mapped to labels.
    :return: numpy array
        Grayscale array representing labels.
    """

    # Initialize labels array with zeros
    labels_image = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

    # Create masks for each color in the color map
    masks = [(rgb_image == np.array(color)).all(axis=2) for color in color_map.values()]

    # Combine masks to find the label for each pixel
    for label, mask in enumerate(masks):
        labels_image[mask] = label

    return labels_image


def create_labels_from_preds(preds_folder, labels_destination, color_map=None):
    # Check if the directory exists
    os.makedirs(labels_destination, exist_ok=True)

    # Set default color map corresponding to plotly qualitative G10
    if color_map is None:
        color_map = {
            "#3366CC": "#3366CC",
            "#DC3912": "#DC3912",
            "#FF9900": "#FF9900",
            "#109618": "#1BB392",
            "#990099": "#5B1982",
            "#0099C6": "#C1C4C9",
            "#DD4477": "#FA9BDA",
            "#66AA00": "#A2DDF2",
            "#B82E2E": "#047511",
            "#316395": "#755304",
        }

    # Precompute RGB values from keys in color map.
    orig_rgb = [ImageColor.getrgb(hex_color_in) for hex_color_in in color_map.keys()]

    # Create new dictionary mapping of integers to RGB values.
    orig_rgb_dict = {0: (0, 0, 0)}
    for i, rgb in enumerate(orig_rgb, start=1):
        orig_rgb_dict[i] = rgb

    preds_list = os.listdir(preds_folder)

    # Process each prediction image in parallel
    def process_image(preds):
        preds_path = os.path.join(preds_folder, preds)
        preds_image = cv2.imread(preds_path)

        # Convert image to RGB
        preds_image_rgb = cv2.cvtColor(preds_image, cv2.COLOR_BGR2RGB)

        # Convert the RGB image into integer labels
        gray_img = orig_rgb_to_gray_labels(preds_image_rgb, orig_rgb_dict)

        file_root, file_ext = os.path.splitext(preds)

        # Create the new filename with the label appended
        label_image_name = f"{file_root}_labels.png"
        label_image_path = os.path.join(labels_destination, label_image_name)

        cv2.imwrite(label_image_path, gray_img)

    with tqdm(total=len(preds_list), desc="Generating labels from predictions") as pbar:
        with ThreadPoolExecutor() as executor:
            for _ in executor.map(process_image, preds_list):
                pbar.update(1)

    return None


def quantify_water_on_roadway(labels_folder, roadway_mask_path, csv_path=None):
    roadway_mask = cv2.imread(roadway_mask_path)
    roadway_mask = cv2.cvtColor(roadway_mask, cv2.COLOR_BGR2GRAY)
    mask = np.array(roadway_mask) > 0

    roadway_pixels = np.sum(mask == 1)

    # Initialize a list to store the results
    results = []

    # Process each image in the folder
    with tqdm(
        total=len(os.listdir(labels_folder)), desc="Quantifying water on the roadway"
    ) as pbar:
        for image_name in os.listdir(labels_folder):
            if image_name.endswith(".png"):
                image_path = os.path.join(labels_folder, image_name)

                # Load the grayscale image
                grayscale_image = Image.open(image_path).convert("L")
                image_array = np.array(grayscale_image)

                # Apply the mask using element-wise multiplication
                masked_image = np.where(mask, image_array, 0)

                # Count the number of pixels with the integer value 1 within the masked area
                count = np.sum(masked_image == 1)

                percentage = round((count / roadway_pixels) * 100, 2)

                # Append the results to the list
                results.append(
                    {
                        "ImageName": image_name,
                        "Pixel Count": count,
                        "Percent of Roadway": percentage,
                    }
                )

                pbar.update(1)

    # Convert the list of results to a DataFrame
    water_quantities_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    if csv_path is None:
        csv_path = "water_on_roadway.csv"

    water_quantities_df.to_csv(csv_path, index=False)

    return print(f"Results saved to {csv_path}")


def plot_images_side_by_side(
    images_folder, overlays_folder, output_folder, csv_file, dpi=250
):
    """
    This function plots original images and overlay images from their respective folders side by side,
    and adds pixel count and percentage from a CSV file to the plot.

    :param images_folder: str
        Path to the folder containing the original images.
    :param overlays_folder: str
        Path to the folder containing the segmentation overlays.
    :param output_folder: str
        Path to place the side-by-side images.
    :param csv_file: str
        Path to the CSV file containing ImageName, Pixel Count, and Percent of Roadway.
    :param dpi: int
        The dpi for side-by-side plots. Default value of 250.
    :return: None
        There is no return from this function.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Extract the base name (first 23 characters) of ImageName for matching
    df["BaseName"] = df["ImageName"].str[:23]

    # Sort DataFrame by the base name
    df.sort_values("BaseName", inplace=True)

    # Get a list of files in each folder
    images = [
        os.path.join(images_folder, filename)
        for filename in os.listdir(images_folder)
        if not filename.startswith(".")
        and os.path.isfile(os.path.join(images_folder, filename))
    ]
    overlays = [
        os.path.join(overlays_folder, filename)
        for filename in os.listdir(overlays_folder)
        if not filename.startswith(".")
        and os.path.isfile(os.path.join(overlays_folder, filename))
    ]

    # Sort the files so that they match.
    images.sort()
    overlays.sort()

    # Ensure both folders have the same number of images
    if len(images) != len(overlays):
        print("Error: The two folders must contain the same number of images.")
        return

    with tqdm(total=len(images), desc="Generating side-by-sides") as pbar:
        for image, overlay in zip(images, overlays):
            # Extract the base name (without folder path) for comparison
            base_name = os.path.basename(image)[:23]

            # Find the corresponding row in the CSV using the BaseName column
            row = df[df["BaseName"] == base_name].iloc[0]
            pixel_count = row["Pixel Count"]
            percent_roadway = row["Percent of Roadway"]

            # Open images from both folders
            img1 = Image.open(image)
            img2 = Image.open(overlay)

            # Create a new figure
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Plot images side by side
            axes[0].imshow(img1)
            axes[0].axis("off")
            axes[0].set_title("Original Image")

            axes[1].imshow(img2)
            axes[1].axis("off")
            axes[1].set_title("Segmentation Overlay")

            # Add text with pixel count and percentage
            fig.suptitle(
                f"Total Number of Pixels Classified as Water: {pixel_count}, Percent of Pixels in Roadway Classified as Water: {percent_roadway:.2f}%",
                fontsize=12,
            )

            # Save the figure to the output folder
            output_file = os.path.join(
                output_folder, f"side_by_side_{os.path.basename(image)}"
            )
            plt.savefig(output_file, bbox_inches="tight", dpi=dpi)
            plt.close(fig)

            # Update the progress bar
            pbar.update(1)

    return print(f"Side by side images saved successfully.")
