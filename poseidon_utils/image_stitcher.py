import cv2
import numpy as np
import argparse
from pathlib import Path
import concurrent.futures
import sys

def stitch_image_pair(orig_path, overlay_path, output_path):
    """Reads two images, stitches them side-by-side, and saves the result."""
    orig_img = cv2.imread(str(orig_path))
    overlay_img = cv2.imread(str(overlay_path))

    if orig_img is None:
        return f"Error: Could not read original image: {orig_path}"
    if overlay_img is None:
        return f"Error: Could not read overlay image: {overlay_path}"

    if orig_img.shape != overlay_img.shape:
        overlay_img = cv2.resize(overlay_img, (orig_img.shape[1], orig_img.shape[0]))

    stitched_img = np.hstack((orig_img, overlay_img))

    success = cv2.imwrite(str(output_path), stitched_img)
    if not success:
        return f"Error: Failed to save stitched image to {output_path}"
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Stitch original images and overlays side-by-side.")
    parser.add_argument("orig_dir", help="Directory containing original images")
    parser.add_argument("overlay_dir", help="Directory containing the C++ generated overlays")
    parser.add_argument("output_dir", help="Directory to save the stitched comparisons")
    parser.add_argument("--workers", type=int, default=None, 
                        help="Number of CPU cores to use.")
    
    args = parser.parse_args()

    orig_dir = Path(args.orig_dir).resolve()
    overlay_dir = Path(args.overlay_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # --- DIAGNOSTIC LOGGING ---
    print("="*60)
    print("RUN TIME DIAGNOSTICS")
    print("="*60)
    print(f"Originals Directory: {orig_dir}")
    print(f"  - Exists? {orig_dir.exists()}")
    if orig_dir.exists():
        total_orig_files = len(list(orig_dir.iterdir()))
        print(f"  - Total files found inside: {total_orig_files}")

    print(f"Overlays Directory:  {overlay_dir}")
    print(f"  - Exists? {overlay_dir.exists()}")
    if overlay_dir.exists():
        total_overlay_files = len(list(overlay_dir.iterdir()))
        print(f"  - Total files found inside: {total_overlay_files}")
    print("="*60)

    if not orig_dir.exists() or not overlay_dir.exists():
        print("Critical Error: One or both input directories do not exist. Exiting.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    # Added common variations to ensure nothing is missed
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'} 

    for orig_path in orig_dir.iterdir():
        if orig_path.is_file() and orig_path.suffix.lower() in valid_extensions:
            # Match naming convention: segmap_overlay_CAM_DE_...
            overlay_filename = f"segmap_overlay_{orig_path.name}"
            overlay_path = overlay_dir / overlay_filename
            output_path = output_dir / f"comparison_{orig_path.name}"

            if overlay_path.exists():
                tasks.append((orig_path, overlay_path, output_path))
            else:
                # Truncated log statement to avoid overwhelming the .out file
                pass 

    print(f"Matching pairs identified for processing: {len(tasks)}")

    if not tasks:
        print("\n[Failure Analysis]")
        print("No matching image pairs found. Possible reasons:")
        print("1. The files inside the Originals directory do not match the expected extensions.")
        print("2. The Overlays folder contains files, but their names do not exactly match 'segmap_overlay_<original_name>'.")
        if orig_dir.exists() and total_orig_files > 0:
            sample_file = next(orig_dir.iterdir())
            print(f"Sample source filename found: '{sample_file.name}'")
            print(f"Expected overlay filename to be: 'segmap_overlay_{sample_file.name}'")
        sys.exit(0)

    print(f"Starting parallel stitching across {args.workers or 'all available'} workers...")

    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(stitch_image_pair, orig, over, out): orig.name 
            for orig, over, out in tasks
        }

        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                error_msg = future.result()
                if error_msg:
                    print(f"\n{error_msg}")
                else:
                    success_count += 1
                    progress = (success_count / len(tasks)) * 100
                    print(f"\rProgress: [{progress:.2f}%] {success_count}/{len(tasks)}", end="", flush=True)
            except Exception as e:
                print(f"\nCritical error processing {filename}: {e}")

    print("\n\nStitching complete!")

if __name__ == "__main__":
    main()