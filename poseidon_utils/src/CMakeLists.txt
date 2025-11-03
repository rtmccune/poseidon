cmake_minimum_required(VERSION 3.16)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin")

project(POSEIDON VERSION 1.0 LANGUAGES CXX)

# Find the CONDA_PREFIX environment variable.
# This points to your active conda env (e.g., /home/user/miniconda3/envs/poseidon)
set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")

if(NOT EXISTS "${CONDA_LIB_DIR}")
    message(WARNING "CONDA_PREFIX/lib not found. RPATH may not be set correctly. "
                    "Make sure you run cmake from an active conda environment.")
else()
    # Set the RPATH for all compiled executables.
    # This tells the linker to look in your conda env's lib folder *first*
    # when searching for .so files at runtime.
    set(CMAKE_INSTALL_RPATH "${CONDA_LIB_DIR}")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    
    message(STATUS "Setting RPATH to: ${CONDA_LIB_DIR}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find threads library for overlay_generator
find_package(Threads REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS})

# ------------------------------
# pred_label_generator executable
# ------------------------------
add_executable(pred_label_generator pred_label_generator.cpp)
target_link_libraries(pred_label_generator ${OpenCV_LIBS})

# ------------------------------
# overlay_generator executable
# ------------------------------
add_executable(overlay_generator overlay_generator.cpp)
target_link_libraries(overlay_generator ${OpenCV_LIBS} Threads::Threads)

# Optional: Add optimization flags for overlay_generator release build
set_target_properties(overlay_generator PROPERTIES
    COMPILE_FLAGS_RELEASE "-O3 -DNDEBUG"
)
