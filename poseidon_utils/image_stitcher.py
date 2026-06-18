import cv2
import numpy as np
import argparse
from pathlib import Path
import concurrent.futures
import sys

def stitch_image_pair(orig_path, overlay_path, output_path):
    """Reads two images, stitches them side-by-side, and saves the result."""
    # Read images
    orig_img = cv2.imread(str(orig_path))
    overlay_img = cv2.imread(str(overlay_path))

    if orig_img is None:
        return f"Error: Could not read {orig_path}"
    if overlay_img is None:
        return f"Error: Could not read {overlay_path}"

    # Ensure dimensions match before stacking. 
    # (Your C++ script enforces this, but this is a safe fallback).
    if orig_img.shape != overlay_img.shape:
        overlay_img = cv2.resize(overlay_img, (orig_img.shape[1], orig_img.shape[0]))

    # Perform horizontal stacking (side-by-side)
    # To stack vertically (top-and-bottom), change hstack to vstack
    stitched_img = np.hstack((orig_img, overlay_img))

    # Save the result
    success = cv2.imwrite(str(output_path), stitched_img)
    if not success:
        return f"Error: Failed to save to {output_path}"
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Stitch original images and overlays side-by-side.")
    parser.add_argument("orig_dir", help="Directory containing original images")
    parser.add_argument("overlay_dir", help="Directory containing the C++ generated overlays")
    parser.add_argument("output_dir", help="Directory to save the stitched comparisons")
    parser.add_argument("--workers", type=int, default=None, 
                        help="Number of CPU cores to use. Defaults to all available.")
    
    args = parser.parse_args()

    orig_dir = Path(args.orig_dir)
    overlay_dir = Path(args.overlay_dir)
    output_dir = Path(args.output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather tasks
    tasks = []
    # Assuming standard image formats. Expand this tuple if you use others like .tif
    valid_extensions = ('.png', '.jpg', '.jpeg') 

    for orig_path in orig_dir.iterdir():
        if orig_path.is_file() and orig_path.suffix.lower() in valid_extensions:
            # Match the naming convention from your C++ script
            overlay_filename = f"segmap_overlay_{orig_path.name}"
            overlay_path = overlay_dir / overlay_filename
            output_path = output_dir / f"comparison_{orig_path.name}"

            if overlay_path.exists():
                tasks.append((orig_path, overlay_path, output_path))
            else:
                print(f"Warning: No matching overlay found for {orig_path.name}")

    if not tasks:
        print("No matching image pairs found. Exiting.")
        sys.exit(0)

    print(f"Found {len(tasks)} image pairs. Starting parallel stitching...")

    # Process in parallel
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks to the process pool
        futures = {
            executor.submit(stitch_image_pair, orig, over, out): orig.name 
            for orig, over, out in tasks
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                error_msg = future.result()
                if error_msg:
                    print(error_msg)
                else:
                    success_count += 1
                    # Print progress on the same line (similar to your C++ logic)
                    progress = (success_count / len(tasks)) * 100
                    print(f"\rProgress: [{progress:.2f}%] {success_count}/{len(tasks)}", end="")
            except Exception as e:
                print(f"\nCritical error processing {filename}: {e}")

    print("\n\nStitching complete!")

if __name__ == "__main__":
    main()