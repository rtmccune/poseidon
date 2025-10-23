#!/bin/bash

# Exit immediately if any command fails
set -e

cat << 'EOF'
                       ███████████████                                                
                   ████████       █████████                                           
               ██████                    █████                                        
             ████             █             ████                                      
           ████        █     ███     █         ███                                    
          ███          ███  █████   ██           ███                                  
        ███            ███   ███   ███            ███                                 
       ███             ██    ███    ██             ███                                
      ███              ██    ███    ██              ███                               
     ███              ███    ███    ███              ███                              
     ██               ███    ███     ███              ██                              
    ███              ████    ███    ████               ██                             
    ██                █████  ███  █████                ██                             
    ██                 ███████████████                 ██                            
    ██                     ███████                     ██                            
    ██                       ████                      ██                            
    ██                       ███████                   ██                            
    ██                       ███████                   ██                             
    ██                       ██████                    ██                             
     ██                    ███████                    ███                             
     ███                    ███████                  ███                              
      ███                    ██████                 ███                               
       ███                   █████                 ███                                
        ███       ███████   █████    ██████       ███                                 
         ████ █████ █████████████ ███ ██████████ ███                                  
           ████ ██  ██  █████████████  ███ ██ ████                                    
             ████ ████  █████████████   ████████                                      
                ██████████████████████████████                                        
                   ████████████████████████                                           
                       ███████████████                                                                                                                                                                                                                                                                                                         
██████╗  ██████╗ ███████╗███████╗██╗██████╗  ██████╗ ███╗   ██╗
██╔══██╗██╔═══██╗██╔════╝██╔════╝██║██╔══██╗██╔═══██╗████╗  ██║
██████╔╝██║   ██║███████╗█████╗  ██║██║  ██║██║   ██║██╔██╗ ██║
██╔═══╝ ██║   ██║╚════██║██╔══╝  ██║██║  ██║██║   ██║██║╚██╗██║
██║     ╚██████╔╝███████║███████╗██║██████╔╝╚██████╔╝██║ ╚████║
╚═╝      ╚═════╝ ╚══════╝╚══════╝╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝
EOF

# --- 1. Check for Python and OpenCV ---
echo "Checking for Python 3 and OpenCV library..."
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo "Error: 'python3' or 'python' command not found."
        echo "Please install Python 3 to run these tests."
        exit 1
    fi
fi

# Try to import cv2 (OpenCV)
$PYTHON_CMD -c "import cv2"
if [ $? -ne 0 ]; then
    echo "Error: Python OpenCV module ('cv2') not found."
    echo "Please install it, e.g., 'pip install opencv-python-headless'"
    exit 1
fi
echo "Python and OpenCV found ($PYTHON_CMD)."

# --- 2. Setup Test Environment ---
echo "Creating testing sandbox..."
SANDBOX_DIR="testing_sandbox"
rm -rf $SANDBOX_DIR
mkdir $SANDBOX_DIR
cd $SANDBOX_DIR

# Create input/output directories
mkdir image_folder
mkdir segs_folder
mkdir overlay_dest
mkdir labels_dest

# --- 3. Create Test Images with Python ---
echo "Generating test images with Python/OpenCV..."

# Create a small python script to generate our images
cat << EOF > generate_images.py
import cv2
import numpy as np
import os

print("  -> Creating image_folder/test_img.png")
# Create a 100x100 blue background image
# OpenCV uses BGR, so blue is (255, 0, 0)
bg_img = np.zeros((100, 100, 3), dtype=np.uint8)
bg_img[:] = (255, 0, 0) # BGR Blue
cv2.imwrite("image_folder/test_img.png", bg_img)

print("  -> Creating segs_folder/test_img_predseg.png")
# Create a 100x100 segmentation mask
# Color is "#3366CC" (R=51, G=102, B=204)
# In BGR: (204, 102, 51)
seg_img = np.zeros((100, 100, 3), dtype=np.uint8)
seg_img[:] = (204, 102, 51) # BGR for #3366CC
cv2.imwrite("segs_folder/test_img_predseg.png", seg_img)

print("Test images created successfully.")
EOF

# Run the python script
$PYTHON_CMD generate_images.py

# --- 4. Test 1: create_overlay ---
echo "---"
echo "Running Test 1: create_overlay"

# 1. Create file list
echo "test_img.png" > file_list.txt

# 2. Run the executable (it's in the parent dir, so ../)
echo "Running ./create_overlay..."
../overlay_generator image_folder segs_folder overlay_dest file_list.txt 0.5

# 3. Check for the expected output file
OVERLAY_OUTPUT="overlay_dest/segmap_overlay_test_img.png"
if [ -f "$OVERLAY_OUTPUT" ]; then
    echo "SUCCESS: Test 1 passed. Output file '$OVERLAY_OUTPUT' was created."
else
    echo "FAILURE: Test 1 failed. Output file '$OVERLAY_OUTPUT' was NOT created."
    exit 1
fi

# --- 5. Test 2: create_labels ---
echo "---"
echo "Running Test 2: create_labels"

# 1. Create file list
echo "test_img_predseg.png" > label_file_list.txt

# 2. Run the executable
echo "Running ./create_labels..."
../pred_label_generator segs_folder label_file_list.txt labels_dest

# 3. Check for the expected output file
LABEL_OUTPUT="labels_dest/test_img_predseg_labels.png"
if [ -f "$LABEL_OUTPUT" ]; then
    echo "SUCCESS: Test 2 passed. Output file '$LABEL_OUTPUT' was created."
else
    echo "FAILURE: Test 2 failed. Output file '$LABEL_OUTPUT' was NOT created."
    exit 1
fi

# --- 6. Cleanup ---
echo "---"
echo "All tests passed successfully!"
echo "Cleaning up sandbox directory..."
cd ..
rm -rf $SANDBOX_DIR

exit 0
