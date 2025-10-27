import os
import re
import cv2
import shutil
import itertools
import cupy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import ImageColor
from concurrent.futures import ThreadPoolExecutor


def extract_camera_name(filename):
    """Extracts sensor ID from filenames."""
    pattern = r"CAM_[A-Z]{2}_[0-9]{2}"
    match = re.search(pattern, filename)
    return match.group(0) if match else None


def extract_timestamp(filename):
    """Extracts UTC timestamp from filenames."""
    pattern = r"\d{14}"
    match = re.search(pattern, filename)
    return match.group(0) if match else None


def organize_images_by_flood_events(
    image_folder, csv_file, destination_folder, subfolder_name
):
    """
    Organizes images into folders based on flood event time ranges and camera IDs from a CSV file.

    Parameters:
    ----------
    image_folder : str
        Path to the directory containing the original images.

    csv_file : str
        Path to the CSV file that defines flood event time ranges and associated sensor IDs.
        The CSV must contain the following columns:
        - 'sensor_ID' : The ID of the sensor (without 'CAM_' prefix).
        - 'start_time_UTC' : The start time of the flood event in UTC.
        - 'end_time_UTC' : The end time of the flood event in UTC.

    destination_folder : str
        Path where the organized folders should be created.

    subfolder_name : str
        Name of the subfolder inside each event folder to store the corresponding images (e.g., "orig_images").

    Behavior:
    --------
    For each row in the CSV, the function:
    - Creates a folder named `{sensor_ID}_{start_time}_{end_time}`.
    - Moves all images from `image_folder` whose extracted timestamps fall within the flood event's time range
      and whose extracted camera name matches the `sensor_ID` (with 'CAM_' prefix) into the specified subfolder.

    Notes:
    -----
    This function relies on two helper functions:
    - `extract_timestamp(filename)` : Extracts and returns a UTC timestamp from the filename.
    - `extract_camera_name(filename)` : Extracts and returns the camera ID (with 'CAM_' prefix) from the filename.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert the start and end time columns to datetime
    df["start_time_UTC"] = pd.to_datetime(
        df["start_time_UTC"], utc=True
    ) - pd.Timedelta(hours=3)
    df["end_time_UTC"] = pd.to_datetime(
        df["end_time_UTC"], utc=True
    ) + pd.Timedelta(hours=3)

    # Format datetime to string for folder naming
    df["start_time_str"] = df["start_time_UTC"].dt.strftime("%Y%m%d%H%M%S")
    df["end_time_str"] = df["end_time_UTC"].dt.strftime("%Y%m%d%H%M%S")

    df["camera_ID"] = "CAM_" + df["sensor_ID"]

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        camera_id = row["camera_ID"]
        sensor_id = row["sensor_ID"]
        start_time_str = row["start_time_str"]
        end_time_str = row["end_time_str"]

        start_time = row["start_time_UTC"]
        end_time = row["end_time_UTC"]

        # Create a folder name based on sensor ID and time range
        folder_name = f"{sensor_id}_{start_time_str}_{end_time_str}"
        folder_path = os.path.join(destination_folder, folder_name)

        # Create the main folder and the orig_images subfolder
        os.makedirs(os.path.join(folder_path, subfolder_name), exist_ok=True)

        # Iterate through the image files in the folder
        for filename in os.listdir(image_folder):
            if filename.endswith(
                (".png", ".jpg", ".jpeg")
            ):  # Add any other image formats if needed
                # Extract timestamp from the filename
                timestamp = pd.to_datetime(
                    extract_timestamp(filename), format="%Y%m%d%H%M%S", utc=True
                )
                camera_name = extract_camera_name(filename)

                if (
                    timestamp
                    and start_time <= timestamp <= end_time
                    and camera_name == camera_id
                ):
                    # Move the file to the corresponding folder
                    shutil.copy(
                        os.path.join(image_folder, filename),
                        os.path.join(folder_path, subfolder_name, filename),
                    )


def orig_rgb_to_gray_labels(rgb_image, color_map, use_gpu):
    """
    Converts an RGB image to a grayscale array representing labels based on the given color map.
    :param rgb_image: numpy array
        RGB image array.
    :param color_map: dict
        Color map where RGB colors are mapped to labels.
    :return: numpy array
        Grayscale array representing labels.
    """
    if use_gpu:  # If GPU processing
        # Convert the input RGB image to a CuPy array
        rgb_image = cp.array(rgb_image)

        # Initialize labels array with zeros
        labels_image = cp.zeros(rgb_image.shape[:2], dtype=np.uint8)

        # Create masks for each color in the color map
        masks = [
            (rgb_image == cp.array(color)).all(axis=2)
            for color in color_map.values()
        ]

        # Combine masks to find the label for each pixel
        for label, mask in enumerate(masks):
            labels_image[mask] = label

    else:  # Else, CPU processing

        # Initialize labels array with zeros
        labels_image = np.zeros(rgb_image.shape[:2], dtype=np.uint8)

        # Create masks for each color in the color map
        masks = [
            (rgb_image == np.array(color)).all(axis=2)
            for color in color_map.values()
        ]

        # Combine masks to find the label for each pixel
        for label, mask in enumerate(masks):
            labels_image[mask] = label

    return labels_image


def process_image(
    predictions_dir, labels_destination, orig_rgb_dict, prediction_list, use_gpu
):
    """
    Processes a single predicted segmentation image by converting it from RGB color classes
    to a grayscale integer label image based on a provided color map.

    Parameters:
    ----------
    predictions_dir : str
        Path to the directory containing the input predicted segmentation image.

    labels_destination : str
        Path to the directory where the output grayscale label image will be saved (CPU mode only).

    orig_rgb_dict : dict
        Dictionary mapping integer class labels to RGB tuples. Used to translate RGB pixels into class labels.

    prediction_list : str
        Filename of the prediction image to be processed.

    use_gpu : bool
        If True, processes the image using GPU acceleration via CuPy and returns the result.
        If False, processes on CPU and writes the grayscale image to disk.

    Returns:
    -------
    gray_img : np.ndarray or cp.ndarray
        The grayscale label image where each pixel value corresponds to a class label.
        - If `use_gpu` is True, returns a CuPy array.
        - If `use_gpu` is False, writes the image to disk and returns nothing.
    """
    preds_path = os.path.join(predictions_dir, prediction_list)
    preds_image = cv2.imread(preds_path)

    # Convert image to RGB
    preds_image_rgb = cv2.cvtColor(preds_image, cv2.COLOR_BGR2RGB)

    # Convert the RGB image into integer labels
    gray_img = orig_rgb_to_gray_labels(preds_image_rgb, orig_rgb_dict, use_gpu)

    if use_gpu:
        return gray_img

    else:
        file_root, _ = os.path.splitext(prediction_list)

        # Create the new filename with the label appended
        label_image_name = f"{file_root}_labels.png"
        label_image_path = os.path.join(labels_destination, label_image_name)

        cv2.imwrite(label_image_path, gray_img)


def create_labels_from_predsegs(
    predictions_dir,
    labels_destination,
    color_map=None,
    use_gpu=False,
    batch_size=20,
):
    """
    Converts predicted segmentation images into integer-labeled grayscale label maps and saves them to disk.

    Parameters:
    ----------
    predictions_dir : str
        Path to the directory containing predicted segmentation images (color-based classification).

    labels_destination : str
        Path to the directory where grayscale label images will be saved. Directory will be created if it doesn't exist.

    color_map : dict, optional
        A dictionary mapping original RGB hex color codes (as strings) to output hex values.
        If None, a default color map similar to Plotly's qualitative G10 scheme will be used.

    use_gpu : bool, default=False
        If True, processes images using CuPy on the GPU for faster performance (recommended for large datasets).
        If False, processes images on the CPU using multithreading.

    batch_size : int, default=20
        Number of images to process in a single batch when using GPU mode.

    Returns:
    -------
    None
        This function saves grayscale label images to the specified directory and does not return a value.

    Notes:
    -----
    - Predicted segmentation images are assumed to use specific RGB colors to denote classes.
    - Each unique color in the color map is assigned an integer label starting from 1 (0 is reserved for background).
    - The output grayscale label images have the same spatial resolution as the input predictions.
    - When using the GPU, image batches are loaded into memory and processed with CuPy for speed.
    """
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
    orig_rgb = [
        ImageColor.getrgb(hex_color_in) for hex_color_in in color_map.keys()
    ]

    # Create new dictionary mapping of integers to RGB values.
    orig_rgb_dict = {0: (0, 0, 0)}
    for i, rgb in enumerate(orig_rgb, start=1):
        orig_rgb_dict[i] = rgb

    prediction_list = os.listdir(predictions_dir)

    # Process each prediction image in parallel
    with tqdm(
        total=len(prediction_list), desc="Generating labels from predictions"
    ) as pbar:
        if use_gpu:
            # for prediction in prediction_list:
            #     process_image(predictions_dir, labels_destination, orig_rgb_dict, prediction, use_gpu=use_gpu)
            #     pbar.update(1)
            for i in range(0, len(prediction_list), batch_size):
                # Process a batch of images
                batch_predictions = prediction_list[i : i + batch_size]
                batch_height, batch_width = cv2.imread(
                    os.path.join(predictions_dir, batch_predictions[0])
                ).shape[:2]

                # Allocate a large CuPy array for the entire batch
                batch_labels = cp.zeros(
                    (len(batch_predictions), batch_height, batch_width),
                    dtype=cp.uint8,
                )

                for j, prediction in enumerate(batch_predictions):
                    gray_img = process_image(
                        predictions_dir,
                        labels_destination,
                        orig_rgb_dict,
                        prediction,
                        use_gpu,
                    )
                    batch_labels[j] = (
                        gray_img  # Directly store in the batch_labels array
                    )

                # Write the batch of images to disk
                for j, prediction in enumerate(batch_predictions):
                    file_root, _ = os.path.splitext(prediction)
                    label_image_name = f"{file_root}_labels.png"
                    label_image_path = os.path.join(
                        labels_destination, label_image_name
                    )
                    cv2.imwrite(
                        label_image_path, batch_labels[j].get()
                    )  # Offload to CPU once for each image

                pbar.update(
                    len(batch_predictions)
                )  # Update progress bar by the number of processed images

        else:
            # Use ThreadPoolExecutor for CPU processing
            with ThreadPoolExecutor() as executor:
                for _ in executor.map(
                    process_image,
                    itertools.repeat(predictions_dir),
                    itertools.repeat(labels_destination),
                    itertools.repeat(orig_rgb_dict),
                    prediction_list,
                    itertools.repeat(use_gpu),
                ):
                    pbar.update(1)

    return None
