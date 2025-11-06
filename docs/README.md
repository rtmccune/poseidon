<p align="center">
  <img src="logo.png" alt="POSEIDON banner" width="500">
</p>

<h1 align="center">POSEIDON</h1>

<p align="center">
  <b>Photogrammetric Observation and Segmentation for Estimation of Inundation Depth ON-demand</b>
</p>

---

**POSEIDON** is a Python-based scientific tool for automated flood-level extraction from imagery.  
It integrates image segmentation and georectification to detect water surfaces and estimate flood depths across real-world terrain.


# Setup
## Clone the Repository
```bash
git clone --recurse-submodules https://github.com/rtmccune/poseidon.git
```

## Environment
To create a conda environment for this workflow navigate to your local copy of the 'poseidon' repository and run the following command:
```bash
conda env create --file poseidon_deploy/environment.yml
```

To create a conda environment on a linux-64 based system (such as an HPC system) for this workflow navigate to the 'poseidon' repository and run the following command:
```bash
conda env create --prefix /path/to/conda/env --file poseidon_deploy/poseidon-linux-64-lock.yml
```

## C++ Executables
These executables rely on the opencv library for processing. First, ensure that you have opencv installed on your system.

To install opencv:
  On Ubuntu/Debian-based systems run:
  ```bash
  sudo apt install libopencv-dev
  ```
  
  On Fedora/RHEL-based systems run:
  ```bash
  sudo dnf install opencv-devel
  ```

  On macOS:
  ```bash
  brew install opencv
  ```

After the working 'poseidon' environment and OpenCV have been installed on your system, compile the executables using cmake:

1. Activate poseidon conda environment.
  ```bash
  conda activate poseidon
  ```

2. Create a build directory.
  ```bash
  cd poseidon_utils/src
  mkdir build
  cd build
  ```
3. Configure the build for your system.
  ```bash
  cmake ..
  ```
4. Build the exectuables.
  ```bash
  make
  ```
The compiled executables will now exist in the 'bin' directory for use. You can test that these compiled executables are working by running the test_compiled_funcs.sh script from the docs/tests directory. To run these tests from the 'bin' directory simply:
```bash
../../docs/tests/test_compiled_funcs.sh
```
## Segmentation Gym Container
From the repository root directory run:
```bash
apptainer pull poseidon_deploy/segmentation/container/seg_gym.sif oras://ghcr.io/rtmccune/segmentation-gym-tf:latest
```

## Python Tests
To test the posedion library, run the following command from the 'poseidon' directory.
  ```bash
  python -m pytest -v
  ```
