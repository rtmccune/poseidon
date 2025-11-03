# python script: prepare_job_lists.py
import os
import argparse
from pathlib import Path
import numpy as np

def prepare_job_lists(image_folder: str, num_jobs: int, output_dir: str = None):
    """
    Scans a directory of images and splits the file list into a specified
    number of text files for use in an HPC job array.
    """
    # --- CHANGE 1: Use the resolved path object consistently ---
    # Resolve the input path immediately to get a full, absolute path.
    image_path = Path(image_folder).resolve()
    
    # Check if the source directory exists early on
    if not image_path.is_dir():
        print(f"Error: The specified image folder does not exist: '{image_path}'")
        return

    if output_dir is None:
        # This logic now works because the argparse default is None.
        # We create 'job_file_lists' in the parent directory of the source images.
        final_output_path = image_path.parent / 'job_file_lists'
        print(f"Output directory not specified. Defaulting to: {final_output_path}")
    else:
        # If an output directory is specified, resolve it to an absolute path too.
        final_output_path = Path(output_dir).resolve()
        print(f"Using specified output directory: {final_output_path}")
        
    # Create the final output directory if it doesn't exist.
    final_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get sorted list of files using modern pathlib
    # This is cleaner than os.listdir + os.path.join
    files = sorted([f.name for f in image_path.iterdir() if f.is_file() and not f.name.startswith('.')])
    
    if not files:
        print(f"Warning: No image files found in '{image_path}'.")
        return
        
    total_files = len(files)

    # Split into chunks for number of jobs
    file_chunks = np.array_split(files, num_jobs)

    for i, chunk in enumerate(file_chunks):
        # LSF job array indices start at 1, so we'll name our files 1, 2, 3...
        
        # --- CHANGE 2: Use the CORRECT output path variable ---
        # Use `final_output_path` here instead of the original `output_dir`.
        # Also using pathlib's `/` operator for joining paths.
        list_filename = final_output_path / f"file_list_{i+1}.txt"
        
        with open(list_filename, 'w') as f:
            for filename in chunk:
                f.write(filename + '\n')

    # --- CHANGE 3: Report the CORRECT output path ---
    print(f"\nSuccess! Created {len(file_chunks)} file lists in '{final_output_path}' for a total of {total_files} files.")

# This block allows the script to be called from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare file lists for an HPC job array by splitting a directory's contents."
    )

    parser.add_argument(
        "image_folder", 
        type=str, 
        help="The full path to the directory containing the images."
    )
    parser.add_argument(
        "num_jobs", 
        type=int, 
        help="The number of job array tasks (i.e., the number of file lists to create)."
    )
    
    # --- CHANGE 4 (CRITICAL): The argparse default must be None ---
    # This allows our `if output_dir is None:` check to work correctly.
    parser.add_argument(
        "-o", "--output-dir", 
        type=str, 
        default=None, #<-- THIS IS THE KEY FIX
        help="The directory to save file lists. (Default: creates 'job_file_lists' next to the image folder)"
    )

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    prepare_job_lists(args.image_folder, args.num_jobs, args.output_dir)