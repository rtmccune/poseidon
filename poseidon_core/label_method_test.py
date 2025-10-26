import cupy as cp
from cucim.skimage.morphology import binary_closing
from cucim.skimage.measure import label
from cucim.skimage import filters as cucim_filters
import timeit

# --- Configuration ---
ARRAY_SHAPE = (4096, 4096)  # Test on a large 4k x 4k image
N_RUNS = 10  # Number of times to run each function
MIN_POND_SIZE = 55  # The min_size threshold


# --- Function 1: Original Method ---
def _label_ponds_original(gpu_label_array):
    """Original method from the user."""
    labels_squeezed = gpu_label_array.squeeze()
    water_mask = labels_squeezed == 1
    masked_labels = water_mask.astype(cp.uint8)

    closed_data = binary_closing(
        masked_labels, footprint=cp.ones((3, 3), dtype=cp.uint8)
    )

    # --- FIX WAS HERE (added return_num=True) ---
    labeled_data, num_features = label(closed_data, return_num=True)

    min_size = MIN_POND_SIZE
    unique, counts = cp.unique(labeled_data, return_counts=True)
    small_ponds = unique[counts < min_size]

    mask_small_ponds = cp.isin(labeled_data, small_ponds)
    labeled_data[mask_small_ponds] = 0

    # --- AND FIX WAS HERE (added return_num=True) ---
    labeled_data, num_features = label(labeled_data, return_num=True)

    return labeled_data


# --- Function 2: Optimized Method ---
def _label_ponds_optimized(gpu_label_array):
    """Optimized method using a lookup table (LUT)."""
    labels_squeezed = gpu_label_array.squeeze()
    water_mask = labels_squeezed == 1
    masked_labels = water_mask.astype(cp.uint8)

    closed_data = binary_closing(
        masked_labels, footprint=cp.ones((3, 3), dtype=cp.uint8)
    )

    # --- AND FIX WAS HERE (added return_num=True) ---
    labeled_data, num_features = label(closed_data, return_num=True)

    if num_features == 0:
        return labeled_data

    # Use bincount (faster than unique)
    counts = cp.bincount(labeled_data.ravel(), minlength=num_features + 1)

    min_size = MIN_POND_SIZE
    all_labels = cp.arange(num_features + 1, dtype=cp.int32)
    large_ponds = all_labels[(counts >= min_size) & (all_labels != 0)]

    # Create the Lookup Table (LUT)
    lut = cp.zeros(num_features + 1, dtype=cp.int32)
    new_labels = cp.arange(1, len(large_ponds) + 1, dtype=cp.int32)
    lut[large_ponds] = new_labels

    # Apply the LUT (fast, single operation)
    relabeled_data = lut[labeled_data]

    return relabeled_data


# --- Test Harness ---


def generate_test_data(shape):
    """Creates realistic, blob-like test data on the GPU."""
    print(f"Generating test data of shape {shape}...")
    # Create smooth, blob-like noise
    noise = cucim_filters.gaussian(cp.random.rand(*shape), sigma=5)
    # Threshold to create binary "ponds"
    binary_mask = noise > 0.65
    # Create the final test array in the expected format (1, H, W)
    test_data = binary_mask.astype(cp.uint8)[cp.newaxis, :, :]
    print("Test data generated and moved to GPU.")
    return test_data


def main():
    # 1. Create the test data
    test_data_gpu = generate_test_data(ARRAY_SHAPE)

    # 2. Define wrapper functions for timeit
    #    We MUST include synchronization to get accurate GPU timings
    def time_original():
        _label_ponds_original(test_data_gpu)
        cp.cuda.Device().synchronize()  # Wait for GPU to finish

    def time_optimized():
        _label_ponds_optimized(test_data_gpu)
        cp.cuda.Device().synchronize()  # Wait for GPU to finish

    # 3. Warm-up
    #    Run once to compile CUDA kernels, etc.
    print("Warming up GPU kernels...")
    time_original()
    time_optimized()
    print("Warm-up complete.")

    # 4. Run benchmark
    print(f"Running benchmarks ({N_RUNS} runs each)...")

    # Time the original function
    total_time_original = timeit.timeit(time_original, number=N_RUNS)
    avg_time_original = (total_time_original / N_RUNS) * 1000  # in ms

    # Time the optimized function
    total_time_optimized = timeit.timeit(time_optimized, number=N_RUNS)
    avg_time_optimized = (total_time_optimized / N_RUNS) * 1000  # in ms

    # 5. Print results
    print("\n--- Benchmark Results ---")
    print(f"Array Shape:   {ARRAY_SHAPE}")
    print(f"Number of Runs: {N_RUNS}\n")

    print(f"Original Method:")
    print(f"  Total time:   {total_time_original:.4f} s")
    print(f"  Average time: {avg_time_original:.2f} ms per run")

    print(f"\nOptimized Method (LUT):")
    print(f"  Total time:   {total_time_optimized:.4f} s")
    print(f"  Average time: {avg_time_optimized:.2f} ms per run")

    print("\n--- Conclusion ---")
    speedup = avg_time_original / avg_time_optimized
    print(f"The optimized method is {speedup:.2f}x faster.")


if __name__ == "__main__":
    main()
