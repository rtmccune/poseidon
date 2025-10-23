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


# Utility Scripts
---
## C++ Executables
These executables rely on the opencv library for processing. First, ensure that you have opencv installed on your system.

To install opencv:
  On Ubuntu/Debian-based systems run:
  '''bash
  sudo apt install libopencv-dev
  '''
  
  On Fedora/RHEL-based systems run:
  '''bash
  sudo dnf install opencv-devel
  '''

  On macOS:
  '''bash
  brew install opencv
  '''

To compile executables using cmake:

1. Create a build directory in the desired location:
  '''bash
  mkdir build
  cd build
  '''
2. Configure the build for your system.
  '''bash
  cmake /path/to/poseidon/utils
  '''
3. Build the exectuables.
  '''bash
  make
  '''

