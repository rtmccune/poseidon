cmake_minimum_required(VERSION 3.16)

project(POSEIDON VERSION 1.0 LANGUAGES CXX)

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
