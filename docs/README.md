<p align="center">
  <img src="logo.png" alt="POSEIDON banner" width="500">
</p>

<h1 align="center">POSEIDON</h1>

<p align="center">
  <b>Photogrammetric Observation and Segmentation for Estimation of Inundation Depth ON-demand</b>
</p>

---

**POSEIDON** is a Python-based scientific tool for automated flood-level
extraction from imagery. It integrates image segmentation and
georectification to detect water surfaces and estimate flood depths across
real-world terrain.

## How it works

1. **Flood events** – `poseidon_utils.EventTracker` downloads Sunny Day
   Flooding Project (SDFP) water levels and lists flood events.
2. **Images** – `poseidon_utils.ImageHandler` pulls event images from the
   SDFP archive and keeps those taken in daylight.
3. **Segmentation** – a Segmentation Gym model labels water in each image.
4. **Grid** – `poseidon_core.GridGenerator` interpolates ground LiDAR onto
   a regular elevation grid for the site.
5. **Rectification** – `poseidon_core.ImageRectifier` projects images and
   labels onto that grid using the camera calibration.
6. **Depths** – `poseidon_core.DepthMapProcessor` finds each pond, takes its
   water surface elevation (WSE) from the ground elevation along its
   waterline, and computes depth = WSE − ground.
7. **Products** – `DepthMapPlotter` maps depths and virtual sensor time
   series; `RoadwayAnalyzer` summarizes flooding on a roadway.

## Repository layout

```
poseidon_core/          Processing pipeline (grid, rectify, depths, plots, roadway)
poseidon_utils/         Shared utilities and data access
  constants.py            Unit conversions and time zones
  naming.py               Camera/timestamp parsing of SDFP filenames
  logging_utils.py        setup_logging() for scripts and notebooks
  zarr_io.py              Atomic ZIP Zarr store helpers
  site_config.py          Per-site YAML configuration loader
  event_tracker.py        SDFP API client and flood event CSVs (shared with flood_event_QC)
  image_handler.py        SDFP image archive access and filtering (shared with flood_event_QC)
  utils.py                log() helper used by the two shared modules
  file_organizer.py       Per-event folder organization
  src/                    C++ label and overlay generators
poseidon_deploy/
  configs/sites/          One YAML per site: camera, grid, CRS, sensors, ROI
  naiads/                 Command-line drivers for each pipeline stage
  hpc_submission_scripts/ LSF job scripts that run the drivers
  analysis/               Cross-event analysis and QC plots
  environment/            Conda environment and lock files
  calibration/            Intrinsic camera calibration notebook
  segmentation/           Segmentation Gym submodule, models, templates
notebooks/              Worked examples of the Python API
tests/                  pytest suite
```

## Setup

### Clone the repository
```bash
git clone --recurse-submodules https://github.com/rtmccune/poseidon.git
cd poseidon
```

### Environment
With an NVIDIA GPU (Linux):
```bash
conda env create --file poseidon_deploy/environment/poseidon-env.yml
```
Without a GPU (any OS; every stage runs on the CPU):
```bash
conda env create --file poseidon_deploy/environment/poseidon-env-no-gpu.yml
```
On a linux-64 HPC system, the explicit spec installs the exact package
builds of the lock file without solving:
```bash
conda create --prefix /path/to/conda/env \
    --file poseidon_deploy/environment/explicit-spec.txt
```
To refresh the lock after editing `poseidon-env.yml` (requires
`conda-lock`):
```bash
cd poseidon_deploy/environment
conda-lock lock -f poseidon-env.yml -p linux-64 \
    --lockfile poseidon-linux-64-lock.yml
python parse_lock.py poseidon-linux-64-lock.yml explicit-spec.txt
```
Then install POSEIDON into the environment:
```bash
conda activate poseidon
python -m pip install -e .
```
Use `python -m pip`, not a bare `pip`: if the environment has no `pip`
of its own, `pip` resolves to the system one (on the NC State HPC, a
Python 3.9 user install), which fails with
`requires a different Python: 3.9 not in '>=3.11'`.

GPU acceleration (CuPy/cuCIM) is optional. Classes take `use_gpu=None` to
use a GPU when one is available, `False` to force the CPU, or `True` to
require a GPU.

### C++ executables
The label and overlay generators need OpenCV (included in the conda
environment). From an activated environment:
```bash
cd poseidon_utils/src
cmake -S . -B build
cmake --build build
```
The executables are written to `poseidon_utils/bin/`. Rebuild them after
upgrading OpenCV.

### Segmentation Gym container
From the repository root:
```bash
mkdir -p poseidon_deploy/segmentation/container   # not in a fresh clone
apptainer pull poseidon_deploy/segmentation/container/seg_gym.sif \
    oras://ghcr.io/rtmccune/segmentation-gym-tf:latest
```
The image is about 7 GB. On a cluster, point `APPTAINER_CACHEDIR` and
`APPTAINER_TMPDIR` at project storage first so the download does not
fill your home quota.

## Site configuration

Each monitoring site is described once in
`poseidon_deploy/configs/sites/<site>.yaml`: camera intrinsics and
extrinsics, rectification grid extent and resolution, coordinate
reference system, LiDAR units, virtual sensor pixels, camera
latitude/longitude, and default data paths (LiDAR, basemap, roadway ROI).
Drivers take `--site <name>` instead of these values on every command
line. Add a site by copying an existing YAML; the calibration notebook
prints intrinsics in the YAML format.

```python
import poseidon_utils
site = poseidon_utils.load_site_config("carolina_beach")
```

## Running the pipeline on the HPC

1. Copy `poseidon_deploy/hpc_paths.template.env` to
   `poseidon_deploy/hpc_paths.env` and set `POSEIDON_ENV`, `IMAGE_DRIVE`,
   and `SITE`. Data are read from and written to `data/$SITE/`.
2. Submit each stage from `poseidon_deploy/hpc_submission_scripts/`.

The HPC is moving from LSF to Slurm. Scripts with `#SBATCH` headers are
submitted with `sbatch`, e.g. `sbatch 07_make_rectify.sh`:

* Stage 02 and the segmentation `99_*` scripts use the GPU partitions.
* Stages 07, 08, 10 and 11 run on the CPU, on idle partner nodes
  (`--account=kanarde_cpu --partition=compute_partners --qos=short`, a
  2-hour limit per task). They are limited by file reads and writes, so a
  GPU does not speed them up.

The other stages still have `#BSUB` headers for the LSF `ccee` queue and
are submitted with `bsub < 05_flood_event_setup.sh`. `_common.sh` works
under both. List the accounts and QOS you can use with
`sacctmgr show assoc user=$USER format=Account%20,Partition%20,QOS%60`.

Slurm GPU requests must name a GPU type; the scripts ask for one A100.
Override it, or any other header, on the command line:
`sbatch --gres=gpu:l40s:1 02_segment_lists.sh`. Jobs under two hours can
also use idle partner GPUs with
`sbatch --partition=gpu_partners --qos=short_gpu ...`. Check free GPUs
with `si --gpus` and your jobs with `squeue -u $USER`.

| Stage | Script | Purpose |
|---|---|---|
| 00 | `00_pull_and_save_images.sh` | Copy archive images within each flood event |
| 01 | `01_filter_images.sh` | Keep images taken between sunrise and sunset |
| – | `99_create_job_lists.sh` | Split image folders into job-array file lists |
| 02 | `02_segment_lists.sh` | Segment images (job array) |
| 03 | `03_make_labels.sh` | Convert predictions to label images (job array) |
| 04 | `04_make_overlays.sh` | Draw QC overlays (job array) |
| 05 | `05_flood_event_setup.sh` | Create one folder per event with data and images |
| 06 | `06_prep_rect_grid.sh` | Build the site's elevation grid (once per site) |
| 07 | `07_make_rectify.sh` | Rectify images and labels (one task per event) |
| 08 | `08_calc_depths.sh` | Compute WSE and depth maps (one task per event) |
| 09 | `09_make_plots.sh` | Plot maps and virtual sensor series (per event) |
| 10 | `10_calc_roadway.sh` | Roadway accessibility time series (per event) |
| 11 | `11_summarize_road_stats.sh` | Summarize roadway statistics over events |

Per-event stages are job arrays: set the array size to the number of
event folders (`#SBATCH --array=1-N` or `#BSUB -J name[1-N]`). Rerun one
event with, for example,
`TARGET_EVENT=<event folder> sbatch --array=1 07_make_rectify.sh`
(LSF: `TARGET_EVENT=<event folder> bsub -J one < 09_make_plots.sh`).
Re-running a stage replaces its earlier outputs.

The remaining `99_*` scripts run QC and analysis (overlay comparisons,
accessibility plots, anomaly detection) and segmentation model training.

### Roadway ROIs

Stage 10 summarizes depths on the site's `roi_json`, a LabelMe shape
drawn on a rectified (plan view) image. A plan-view ROI assumes the
rectified road lines up with the plan view. To avoid that assumption,
draw a polygon labeled `roadway` on an original camera image (the camera
view is assumed fixed) and set `camera_roi_json` in the site config.
Stage 10 then also rasterizes that polygon and rectifies it with the
camera model, using the same grid-to-pixel lookup as the labels, so its
ROI is exactly the grid cells that sample road pixels. Each ROI writes
its own `<roi>_*_time_series.csv` and `<roi>_roi_mask.png`.

Stage 11 summarizes both ROIs and runs
`poseidon_deploy/analysis/compare_roadway_rois.py`, which writes to
`data/<site>/roi_comparison/`:
- per-image and per-event differences;
- agreement on the impassable flag;
- bias, MAE, RMSE and correlation of coverage and depth;
- the overlap (IoU) of the two footprints, with a figure.

### Roadway coverage and depth plots

`99_road_access_plots.sh` runs `poseidon_deploy/analysis/access_plotter.py`
after stage 10 and writes everything to `data/<site>/plots/`. Every run
also writes `plot_summary.md` (and `plot_summary.csv`) there. It lists
the run settings and, for each figure, how many images and events it used
and which ones were left out.

**Reading a filename.** Outputs are named `<roi>_<metric>_<plot>_<kind>`,
for example `shell_hill_rd_camera_polygon_MeanDepth_1c_binned_depth_good_only.jpg`:

- `<roi>` is the ROI (the LabelMe JSON's name). `shell_hill_rd_polygon` is
  drawn on the rectified plan view; `shell_hill_rd_camera_polygon` is
  drawn on the camera image and then rectified.
- `<metric>` is how one image's depths over the ROI become one number.
  Only cells labelled as water count. A water cell whose ground sits
  above the pond's water surface has depth 0.
  - `MaxDepth`: the deepest water cell on the ROI.
  - `MeanDepth`: the average depth of the ROI's water cells.
  - `MedianDepth`: the middle depth of the ROI's water cells; less
    affected than the mean by a few deep cells.

  Depths come from the `95_perc` depth maps: each pond's water surface
  elevation is the 95th percentile of the ground elevation along its
  waterline, and depth is that elevation minus the ground.
- **Roadway coverage** is the percentage of the ROI's cells with depth
  above 0.
- `<plot>` is the plot number below. A `b` uses every image with water;
  a `c` uses only images scored good in `segmentation_labels.csv`.
  Images that are unreviewed or scored poor count as "other".

**The plots.** Plots 1–1c have one point per image; plots 2–5 have one
point per event.

| Plot | File ending | Shows | Images and events used |
|---|---|---|---|
| 1 | `_1_evolution_plot.jpg` | Coverage against depth, one dot per image; good segmentations in blue, others in grey | Every image with water on the ROI |
| 1b, 1c binned | `_1b_binned_depth.jpg`, `_1c_binned_depth_good_only.jpg` | **The prediction plot.** See below | 1b: all images with coverage > 0. 1c: good images only |
| 1b, 1c table | `_1b_depth_by_coverage.csv`, `_1c_depth_by_coverage.csv` | The numbers behind the binned plot: the depth lookup table | Same as the plot |
| 1b, 1c logistic | `_1b_reversed_fit.jpg`, `_1c_reversed_fit_good_only.jpg` | A smooth logistic curve of depth given coverage, its median, an 80% band and a 95% confidence band | As the binned plot; images with coverage 0 left out |
| 1b, 1c fit files | `_fit_summary.csv`, `_band_calibration.csv` | Fit parameters with confidence intervals, R², RMSE and MAE; and, by coverage range, the share of images inside the band and below the median, plus the fit's mean error | Same as the fit |
| 2 | `_2_summary_scatter.png` | Each event's peak coverage against its peak depth | One point per event, from its good images (all images without labels); events with none are left out |
| 3 | `_3_depth_cdf.png` | Share of events whose peak depth is at or below each depth | Same events as plot 2 |
| 4 | `_4_depth_pdf.png` | How event peak depths are distributed: a density histogram and a smooth density curve | Same events as plot 2 |
| 5 | `_5_depth_exceedance.png` | Share of events whose peak depth reaches at least each depth | Same events as plot 2 |
| 6 | `_6_sensor_1to1.jpg` | Image depth against the in-situ sensor's depth, with a 1:1 line | Good images with a sensor reading within 5 minutes; image-only events drop out |
| 7 | `<roi>_<Max/Mean/Median>WSE_7_sensor_1to1.jpg` | The same for water surface elevation | As plot 6 |
| 8 | `<roi>_SensorDepth_8_binned_depth_good_only.jpg`, `_8_depth_by_coverage.csv`, `_8_reversed_fit_good_only.jpg` (+ fit files) | The 1c plots, but with the in-situ sensor's depth for each image instead of the image-derived depth. Coverage still comes from the image. Made once per ROI, since it doesn't depend on the depth metric | Good images with coverage > 0 and a sensor reading within 5 minutes; image-only events drop out |

Plot 8 compares with 1c: if the sensor's depths by coverage bin match the image-derived ones, the image depths are calibrated against an independent measurement. Bear in mind that the sensor measures at one point while the image metric covers the whole ROI.

Because each plot keeps different images, the counts differ from plot to
plot; `plot_summary.md` gives them.

**The binned plot (1b, 1c).** Images are grouped into 10% coverage bins
below 99%. Images at 99% or more (a fully covered road) go in a separate
panel on the right, because once the road is covered, depth can keep
rising while coverage stays at 100%. For each bin:
- the shaded box spans the 10th to 90th percentile of depth, where 80% of
  the images fall;
- the blue circle is the median depth and the orange diamond the mean
  depth, each with a 95% confidence interval from resampling whole events;
- a hollow marker means fewer than 10 images or 3 events, so treat that
  bin with caution.

To predict depth from an observed coverage, find the bin in
`_1c_depth_by_coverage.csv`. Use the median as the typical depth and the
10th–90th percentiles as the expected range. In the full-coverage bin,
coverage no longer pins down depth: read its 10th percentile as roughly
the least depth to expect. Change the binning with `--coverage_bin_width`
and `--full_coverage`.

The logistic plots are a smooth summary, not a prediction tool. Check
their `_band_calibration.csv` first: the share of images below the median
should stay near 0.5, and the share inside the band near 0.8, in every
coverage range. Colors come from `poseidon_deploy/analysis/plot_style.py`,
a colour-blind-checked selection from the feathers palettes.

### Events without sensor data

Stage 05 needs `flood_events.csv`, the sensor readings inside each event.
When the event sheet comes from QC or was edited by hand, rebuild it from
the sheet, keeping the sheet's event numbers:
```bash
python poseidon_deploy/naiads/run_pull_event_series.py \
    --abbr_events_csv data/down_east/abbr_flood_events.csv
```
This needs SDFP API credentials (`SDFP_API_USERNAME` and
`SDFP_API_PASSWORD`, or `mysecrets.py`). Events with no readings, such as
floods confirmed only from images during a sensor outage, are listed and
left out. Stage 05 still builds their folders from the images, and every
later stage runs on them; plots show the virtual sensors without an
observed series.

### Reviewing segmentations

`99_image_overlay_comparison.sh` stitches each daylight image beside its
overlay into `images/stitched_comparisons/comparison_<image name>`. To list
the images not yet scored in a review CSV (`Filename`,
`Segmentation_Score`), run from the repository root:
```bash
python poseidon_deploy/analysis/list_unreviewed_images.py \
    --site down_east --labels data/segmentation_labels.csv
```
This writes `data/down_east/images_to_review.txt` and logs reviewed and
remaining counts for the daylight and `all_events_during_FOV` folders.

## Example notebooks

* `notebooks/flood_event_pull.ipynb` downloads SDFP water levels for the
  Carolina Beach and Down East sensors and builds each site's flood event,
  outage, and plot outputs (needs SDFP API credentials).
* `notebooks/poseidon_workflow.ipynb` runs grid generation, rectification,
  depth estimation, and roadway analysis on one event with the Python API.

`event_tracker.py`, `image_handler.py`, and `utils.py` in `poseidon_utils`
are kept identical to the copies in the flood_event_QC repository; change
them in both places.

## Tests

From the repository root:
```bash
python -m pytest
```
GPU tests run only when CuPy can use a local GPU (`python -m pytest -m gpu`
selects them). Sun-position tests need `astral`, and the C++ tests need the
built executables (or `POSEIDON_BIN=/path/to/bin`).

## Code style

Code follows PEP 8 (79-character lines) with numpy-style docstrings,
checked by `ruff` using the configuration in `pyproject.toml`:
```bash
ruff check .
```
