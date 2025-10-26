import cupy as cp
import numpy as np
import timeit
import sys


class PondProcessor:
    """
    A mock class to hold the methods and the base grid for testing.
    """

    def __init__(self, grid_shape):
        # Create a base grid, full of NaNs
        self.elev_grid_cp = cp.full(grid_shape, cp.nan, dtype=cp.float32)
        print(f"Initialized mock processor with grid shape {grid_shape}.")

    # --- Version 1: Original Method ---
    def combine_depth_maps_original(self, pond_depths):
        maps_to_combine = list(pond_depths.values())

        if not maps_to_combine:
            return cp.full_like(self.elev_grid_cp, cp.nan)

        combined_map = maps_to_combine[0]

        for i in range(1, len(maps_to_combine)):
            current_map = maps_to_combine[i]
            combined_map = cp.fmax(combined_map, current_map)

        return combined_map

    # --- Version 2: Efficient Method ---
    def combine_depth_maps_efficient(self, pond_depths):
        maps_to_combine = list(pond_depths.values())

        if not maps_to_combine:
            return cp.full_like(self.elev_grid_cp, cp.nan)

        # A small optimization: if there's only one map, no need to stack
        if len(maps_to_combine) == 1:
            return maps_to_combine[0]

        # Stack all maps into a single (N, H, W) array
        try:
            stacked_maps = cp.stack(maps_to_combine, axis=0)
        except cp.cuda.memory.OutOfMemoryError:
            print("\n--- ERROR: Out of GPU memory during cp.stack()! ---")
            print("The 'efficient' method failed. This can happen if you have")
            print("too many ponds or the grid is too large for your VRAM.")
            print("Try reducing NUM_PONDS or GRID_SHAPE.")
            sys.exit(1)

        # Perform a single, optimized reduction along the 'N' axis
        combined_map = cp.nanmax(stacked_maps, axis=0)

        return combined_map


def generate_test_data(grid_shape, num_ponds):
    """
    Generates a dictionary of test pond maps.
    Each map is a CuPy array full of NaNs, with a random
    rectangular "pond" of float values inserted.
    """
    print(
        f"Generating test data: {num_ponds} ponds, grid shape {grid_shape}..."
    )
    pond_depths = {}
    H, W = grid_shape

    # Use numpy for random setup on CPU first, then transfer
    for i in range(num_ponds):
        # Create a base map of NaNs
        base_np = np.full(grid_shape, np.nan, dtype=np.float32)

        # Define a random rectangle for the pond
        x_start = np.random.randint(0, W // 2)
        y_start = np.random.randint(0, H // 2)
        x_size = np.random.randint(W // 4, W // 2)
        y_size = np.random.randint(H // 4, H // 2)

        # Ensure it doesn't go out of bounds
        x_end = min(x_start + x_size, W)
        y_end = min(y_start + y_size, H)

        # Get the actual final shape
        final_y_size = y_end - y_start
        final_x_size = x_end - x_start

        # Create random depth data for that rectangle
        pond_data = (
            np.random.rand(final_y_size, final_x_size).astype(np.float32) * 10.0
        )

        # Place the pond data into the base NaN map
        base_np[y_start:y_end, x_start:x_end] = pond_data

        # Transfer to GPU and store in the dictionary
        pond_depths[i] = cp.asarray(base_np)

    print("Test data generated and transferred to GPU.")
    return pond_depths


# --- Main script logic ---
if __name__ == "__main__":

    # --- Parameters to Tweak ---
    GRID_SHAPE = (2048, 2048)  # Height and Width of the maps
    NUM_PONDS = 100  # Number of maps to combine.
    # Higher numbers show a bigger difference.
    TIMING_NUMBER = 10  # Runs per test loop (lower for slow functions)
    TIMING_REPEAT = 10  # Number of times to repeat the test (to get min/avg)
    # --- End Parameters ---

    # --- 1. Setup ---
    processor = PondProcessor(GRID_SHAPE)
    test_data = generate_test_data(GRID_SHAPE, NUM_PONDS)

    print("\n--- Verifying Correctness ---")

    # Run both functions once to check output
    try:
        result_orig = processor.combine_depth_maps_original(test_data)
        result_eff = processor.combine_depth_maps_efficient(test_data)

        # Ensure the results are identical (handles NaNs correctly)
        if cp.allclose(result_orig, result_eff, equal_nan=True):
            print("Verification successful: Outputs match. 👍")
        else:
            print("VERIFICATION FAILED: Outputs do not match! 👎")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during verification: {e}")
        sys.exit(1)

    # --- 2. Benchmark ---
    print(f"\n--- Starting Benchmark ---")
    print(f"Grid: {GRID_SHAPE}, Ponds: {NUM_PONDS}")
    print(f"Runs per test: {TIMING_NUMBER}, Repeats: {TIMING_REPEAT}")

    # Setup code for timeit.
    # CRITICAL: We define a sync() function to force Python
    # to wait for the GPU to finish. Without this, you only time
    # the kernel *launch*, not its *execution*.
    setup_code = """
import cupy as cp
def sync():
    cp.cuda.Stream.null.synchronize()
    """

    # --- Time Original Method ---
    print("\nTiming original (fmax loop)...")
    t_original = timeit.repeat(
        stmt="processor.combine_depth_maps_original(test_data); sync()",
        setup=setup_code,
        globals={"processor": processor, "test_data": test_data},
        number=TIMING_NUMBER,
        repeat=TIMING_REPEAT,
    )

    # --- Time Efficient Method ---
    print("Timing efficient (stack + nanmax)...")
    t_efficient = timeit.repeat(
        stmt="processor.combine_depth_maps_efficient(test_data); sync()",
        setup=setup_code,
        globals={"processor": processor, "test_data": test_data},
        number=TIMING_NUMBER,
        repeat=TIMING_REPEAT,
    )

    # --- 3. Report Results ---
    print("\n--- 📊 Results ---")

    def report_stats(name, times_list):
        # Calculate time per single run
        times_per_run = [t / TIMING_NUMBER for t in times_list]
        best = min(times_per_run)
        avg = sum(times_per_run) / len(times_per_run)

        print(f"[{name}]")
        print(f"  Best time per run: {best * 1000:.4f} ms")
        print(f"  Avg. time per run: {avg * 1000:.4f} ms")
        return best

    best_orig = report_stats("Original (fmax loop)", t_original)
    best_eff = report_stats("Efficient (stack + nanmax)", t_efficient)

    print("\n--- 🚀 Summary ---")
    if best_eff < best_orig:
        speedup = best_orig / best_eff
        print(f"The efficient method is {speedup:.2f}x faster.")
    else:
        speedup = best_eff / best_orig
        print(
            f"The original method is {speedup:.2f}x faster (this is unexpected)."
        )
