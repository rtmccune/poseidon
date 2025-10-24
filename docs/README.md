<p align="center">
  <img src="docs/logo.png" alt="POSEIDON banner" width="500">
</p>

<h1 align="center">POSEIDON</h1>

<p align="center">
  <b>Photogrammetric Observation and Segmentation for Estimation of Inundation Depth ON-demand</b>
</p>

---

**POSEIDON** is a Python-based scientific tool for automated flood-level extraction from imagery.  
It integrates image segmentation and georectification to detect water surfaces and estimate flood depths across real-world terrain.


# Setup
## Environment
To create a conda environment for this workflow navigate to your local copy of the 'poseidon' repository and run the following command:
```bash
conda env create --file poseidon_deploy/environment.yml
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

1. Create a build directory in the desired location:
  ```bash
  mkdir build
  cd build
  ```
2. Configure the build for your system.
  ```bash
  cmake /path/to/poseidon/poseidon_utils/c_helpers
  ```
3. Build the exectuables.
  ```bash
  make
  ```
The compiled executables will now exist in the 'build' directory for use. You can test that these compiled executables are working by running the test_compiled_funcs.sh script from the docs/tests directory. To run these tests from the 'build' directory simply:
```bash
/home/directory/poseidon/docs/tests/test_compiled_funcs.sh
```
## Tests
To test the core poseidon functions, navigates to 'poseidon/docs/tests' and run the following command. Be sure to change the path to your local path to poseidon.
  ```bash
  PYTHONPATH=/path/to/poseidon pytest -v
  ```
