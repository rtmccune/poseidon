import argparse
import sys

# Assuming the function is in 'poseidon_utils/hpc_tools.py'
import poseidon_utils.file_organizer as file_organizer


def main():
    parser = argparse.ArgumentParser(
        description="Scans a directory and splits file lists for an HPC job array."
    )

    # --- Path Arguments ---
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory containing files to be processed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the 'file_list_*.txt' files. "
        "(Default: 'job_file_lists' in the parent of --image_dir)",
    )

    # --- Configuration Arguments ---
    parser.add_argument(
        "--num_jobs",
        type=int,
        required=True,
        help="The total number of job array tasks (i.e., number of file lists to create).",
    )

    args = parser.parse_args()

    # --- Generate Job Lists ---
    print("Starting job list preparation...")

    # Call the function from your utility module
    file_organizer.prepare_job_lists(
        image_folder=args.image_dir,
        num_jobs=args.num_jobs,
        output_dir=args.output_dir,
    )

    print("Job list preparation finished.")


if __name__ == "__main__":
    main()
