import os
import timeit
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# --- Class Definition ---
# Contains both the original and efficient methods for testing
class PondPlotter:
    def __init__(self, plot_dir):
        # This path is just a placeholder; we've disabled directory creation
        self.pond_edge_elev_plot_dir = plot_dir

    def plot_pond_edge_elevations(
        self,
        labeled_data,
        contour_values_per_pond,
        file_name,
    ):
        """
        [Original Inefficient Method]
        Plots/saves histograms of edge elevations for individual ponds.
        """
        all_pond_dir = os.path.join(self.pond_edge_elev_plot_dir, "all_ponds")
        ind_pond_dir = os.path.join(self.pond_edge_elev_plot_dir, "ind_ponds")
        # MODIFIED: Commented out file I/O to isolate computation time
        # os.makedirs(all_pond_dir, exist_ok=True)
        # os.makedirs(ind_pond_dir, exist_ok=True)

        edge_elevs = []  # initialize pond depth dictionary

        # array of unique pond labels
        unique_pond_ids = cp.unique(labeled_data)

        bins = np.linspace(0, 2, 51)

        for pond_id in unique_pond_ids:
            if pond_id == 0:  # Skip background
                continue

            if pond_id.item() not in contour_values_per_pond:
                print(
                    f"Warning: No contours found for pond_id {pond_id.item()}. "
                    f"Skipping plotting for this pond."
                )
                continue

            # INEFFICIENCY 1: CPU -> GPU
            pond_edge_elevs = cp.array(contour_values_per_pond[pond_id.item()])

            edge_elevs.append(pond_edge_elevs.get())

            # INEFFICIENCY 2: GPU -> CPU (implicit, 3 times)
            mean_val = np.mean(pond_edge_elevs)
            median_val = np.median(pond_edge_elevs)
            percentile_95 = np.percentile(pond_edge_elevs, 95)

            # Plot histogram
            plt.figure(figsize=(8, 6))
            # INEFFICIENCY 3: GPU -> CPU (explicit)
            plt.hist(
                pond_edge_elevs.get(),
                bins=bins,
                color="lightblue",
                edgecolor="black",
                alpha=0.7,
            )

            # Overlay vertical lines
            plt.axvline(
                mean_val.get(), # Minor GPU -> CPU
                color="red",
                linestyle="dashed",
                linewidth=2,
                label=f"Mean: {mean_val:.2f}",
            )
            plt.axvline(
                median_val.get(), # Minor GPU -> CPU
                color="green",
                linestyle="solid",
                linewidth=2,
                label=f"Median: {median_val:.2f}",
            )
            plt.axvline(
                percentile_95.get(), # Minor GPU -> CPU
                color="purple",
                linestyle="dashdot",
                linewidth=2,
                label=f"95th %ile: {percentile_95:.2f}",
            )

            plt.xlim(0, 2)
            plt.xlabel("Elevation")
            plt.ylabel("Frequency")
            plt.title(f"Elevation Histogram for Pond {pond_id}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)

            # MODIFIED: Commented out file I/O
            # plt.savefig(f"{ind_pond_dir}/{file_name}_Pond_{pond_id}")
            plt.close() # Still close to prevent memory leak

        if not edge_elevs:
            return

        all_edge_elevs = np.concatenate(edge_elevs)
        plt.figure(figsize=(8, 6))
        plt.hist(
            all_edge_elevs,
            bins=bins,
            color="lightcoral",
            edgecolor="black",
            alpha=0.7,
        )
        mean_all = np.mean(all_edge_elevs)
        median_all = np.median(all_edge_elevs)
        percentile_95_all = np.percentile(all_edge_elevs, 95)
        plt.axvline(
            mean_all,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean_all:.2f}",
        )
        plt.axvline(
            median_all,
            color="green",
            linestyle="solid",
            linewidth=2,
            label=f"Median: {median_all:.2f}",
        )
        plt.axvline(
            percentile_95_all,
            color="purple",
            linestyle="dashdot",
            linewidth=2,
            label=f"95th %ile: {percentile_95_all:.2f}",
        )
        plt.xlim(0, 2)
        plt.xlabel("Elevation")
        plt.ylabel("Frequency")
        plt.title("Elevation Histogram - All Ponds Combined")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # MODIFIED: Commented out file I/O
        # plt.savefig(f"{all_pond_dir}/{file_name}_All_Ponds_Histogram")
        plt.close() # Still close to prevent memory leak

        return None


    def plot_pond_edge_elevations_efficient(
        self,
        labeled_data,
        contour_values_per_pond,
        file_name,
    ):
        """
        [Optimized Efficient Method]
        Plots/saves histograms of edge elevations for individual ponds.
        """
        all_pond_dir = os.path.join(self.pond_edge_elev_plot_dir, "all_ponds")
        ind_pond_dir = os.path.join(self.pond_edge_elev_plot_dir, "ind_ponds")
        # MODIFIED: Commented out file I/O to isolate computation time
        # os.makedirs(all_pond_dir, exist_ok=True)
        # os.makedirs(ind_pond_dir, exist_ok=True)

        edge_elevs = []  # List will hold NumPy arrays

        # EFFICIENT: Get unique IDs from GPU, transfer to CPU *once*.
        unique_pond_ids = cp.unique(labeled_data).get()

        bins = np.linspace(0, 2, 51)

        for pond_id in unique_pond_ids:
            if pond_id == 0:  # Skip background
                continue

            pond_id_key = pond_id.item()

            if pond_id_key not in contour_values_per_pond:
                print(
                    f"Warning: No contours found for pond_id {pond_id_key}. "
                    f"Skipping plotting for this pond."
                )
                continue

            # EFFICIENT: Data stays on CPU (NumPy)
            pond_edge_elevs_np = contour_values_per_pond[pond_id_key]

            edge_elevs.append(pond_edge_elevs_np)

            # EFFICIENT: All stats computed on CPU
            mean_val = np.mean(pond_edge_elevs_np)
            median_val = np.median(pond_edge_elevs_np)
            percentile_95 = np.percentile(pond_edge_elevs_np, 95)

            # Plot histogram
            plt.figure(figsize=(8, 6))
            # EFFICIENT: Plotting from CPU (no .get())
            plt.hist(
                pond_edge_elevs_np,
                bins=bins,
                color="lightblue",
                edgecolor="black",
                alpha=0.7,
            )

            # Overlay vertical lines
            plt.axvline(
                mean_val,  # No .get()
                color="red",
                linestyle="dashed",
                linewidth=2,
                label=f"Mean: {mean_val:.2f}",
            )
            plt.axvline(
                median_val, # No .get()
                color="green",
                linestyle="solid",
                linewidth=2,
                label=f"Median: {median_val:.2f}",
            )
            plt.axvline(
                percentile_95, # No .get()
                color="purple",
                linestyle="dashdot",
                linewidth=2,
                label=f"95th %ile: {percentile_95:.2f}",
            )

            plt.xlim(0, 2)
            plt.xlabel("Elevation")
            plt.ylabel("Frequency")
            plt.title(f"Elevation Histogram for Pond {pond_id}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)

            # MODIFIED: Commented out file I/O
            # plt.savefig(f"{ind_pond_dir}/{file_name}_Pond_{pond_id}")
            plt.close() # Still close to prevent memory leak

        if not edge_elevs:
            return

        # This part was already efficient
        all_edge_elevs = np.concatenate(edge_elevs)
        plt.figure(figsize=(8, 6))
        plt.hist(
            all_edge_elevs,
            bins=bins,
            color="lightcoral",
            edgecolor="black",
            alpha=0.7,
        )
        mean_all = np.mean(all_edge_elevs)
        median_all = np.median(all_edge_elevs)
        percentile_95_all = np.percentile(all_edge_elevs, 95)
        plt.axvline(
            mean_all,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean_all:.2f}",
        )
        plt.axvline(
            median_all,
            color="green",
            linestyle="solid",
            linewidth=2,
            label=f"Median: {median_all:.2f}",
        )
        plt.axvline(
            percentile_95_all,
            color="purple",
            linestyle="dashdot",
            linewidth=2,
            label=f"95th %ile: {percentile_95_all:.2f}",
        )
        plt.xlim(0, 2)
        plt.xlabel("Elevation")
        plt.ylabel("Frequency")
        plt.title("Elevation Histogram - All Ponds Combined")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # MODIFIED: Commented out file I/O
        # plt.savefig(f"{all_pond_dir}/{file_name}_All_Ponds_Histogram")
        plt.close() # Still close to prevent memory leak

        return None


# --- Main execution block for timing ---
if __name__ == "__main__":
    
    # --- Test Parameters ---
    NUM_PONDS = 100         # Number of ponds to simulate
    IMG_SIZE = (2000, 2000) # Size of the labeled data array
    N_RUNS = 5              # Number of times to run each function for timeit
    
    print(f"--- Preparing Mock Data ---")
    print(f"Simulating {NUM_PONDS} ponds in a {IMG_SIZE} image...")
    
    # 1. Labeled data (on GPU)
    #    Simulate an image with IDs from 0 (background) to NUM_PONDS
    labeled_data_cp = cp.random.randint(0, NUM_PONDS + 1, size=IMG_SIZE, dtype=cp.int32)
    
    # 2. Contour values (on CPU)
    #    This dict maps pond_id (int) -> elevation data (np.ndarray)
    contour_values_per_pond_np = {}
    for i in range(1, NUM_PONDS + 1):
        # Simulate each pond having between 50 and 500 contour points
        num_values = np.random.randint(50, 500)
        # Simulate elevations between 0.0 and 2.0
        contour_values_per_pond_np[i] = np.random.rand(num_values).astype(np.float32) * 2
        
    # 3. File name
    file_name = "timing_test_run"
    
    # 4. Create class instance
    plotter = PondPlotter(plot_dir="/tmp/dummy_plot_dir") # Dummy path
    
    print("Mock data generated. Starting benchmark...")
    print(f"Each function will be run {N_RUNS} times.\n")

    # --- Time the Original (Inefficient) Method ---
    print("--- Timing Original Method ---")
    
    # We use a lambda function to pass arguments to timeit
    t_original = timeit.timeit(
        lambda: plotter.plot_pond_edge_elevations(
            labeled_data_cp, contour_values_per_pond_np, file_name
        ),
        number=N_RUNS
    )
    
    avg_original = t_original / N_RUNS
    print(f"Total time ({N_RUNS} runs): {t_original:.4f} seconds")
    print(f"Average time per run: {avg_original:.4f} seconds\n")


    # --- Time the New (Efficient) Method ---
    print("--- Timing Efficient Method ---")
    
    t_efficient = timeit.timeit(
        lambda: plotter.plot_pond_edge_elevations_efficient(
            labeled_data_cp, contour_values_per_pond_np, file_name
        ),
        number=N_RUNS
    )
    
    avg_efficient = t_efficient / N_RUNS
    print(f"Total time ({N_RUNS} runs): {t_efficient:.4f} seconds")
    print(f"Average time per run: {avg_efficient:.4f} seconds\n")

    # --- Final Results ---
    print("--- Results ---")
    print(f"Average Original:   {avg_original:.4f} s")
    print(f"Average Efficient:  {avg_efficient:.4f} s")
    if avg_efficient > 0:
        speedup = avg_original / avg_efficient
        print(f"\n✅ Efficient version was {speedup:.2f} times faster.")
    else:
        print("\nEfficient version was too fast to measure speedup.")