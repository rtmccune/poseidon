#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <fstream> // Required for reading the file list

#include <opencv2/opencv.hpp>

// Create a namespace alias for std::filesystem for brevity
namespace fs = std::filesystem;

/**
 * @brief A custom comparator struct for using cv::Vec3b as a key in std::map.
 *
 * std::map requires a strict weak ordering (operator<) to sort its keys.
 * cv::Vec3b does not provide this by default, so we define one.
 * This performs a lexicographical comparison on the B, G, and R channels.
 */
struct Vec3bCompare {
    bool operator()(const cv::Vec3b& a, const cv::Vec3b& b) const {
        if (a[0] != b[0]) return a[0] < b[0]; // Compare Blue
        if (a[1] != b[1]) return a[1] < b[1]; // Compare Green
        return a[2] < b[2]; // Compare Red
    }
};

/**
 * @brief Converts a 6-digit hexadecimal color string to a cv::Vec3b BGR color.
 *
 * Handles hex strings with or without a leading '#'.
 *
 * @param hex A string representing the color in "#RRGGBB" or "RRGGBB" format.
 * @return cv::Vec3b The color in OpenCV's default BGR format.
 */
cv::Vec3b hexToBgr(const std::string& hex) {
    // Remove '#' prefix if it exists
    std::string hexColor = (hex[0] == '#') ? hex.substr(1) : hex;
    int r, g, b;
    std::stringstream ss;
    
    // Parse Red
    ss << std::hex << hexColor.substr(0, 2);
    ss >> r;
    ss.clear(); // Clear the stream state
    
    // Parse Green
    ss << std::hex << hexColor.substr(2, 2);
    ss >> g;
    ss.clear();
    
    // Parse Blue
    ss << std::hex << hexColor.substr(4, 2);
    ss >> b;
    
    // Return in BGR order, as expected by OpenCV
    return cv::Vec3b(b, g, r);
}

/**
 * @brief Converts a BGR color image to a single-channel grayscale label image.
 *
 * Iterates through each pixel of the BGR image and uses the reverseColorMap
 * to find the corresponding 8-bit integer label. Unmapped colors
 * are mapped to 0 (background).
 *
 * @param bgrImage The input 3-channel (BGR) color segmentation image.
 * @param reverseColorMap A map from cv::Vec3b BGR colors to uint8_t labels.
 * @return cv::Mat A single-channel (CV_8UC1) image of integer labels.
 */
cv::Mat convertBgrToGrayscaleLabels(const cv::Mat& bgrImage, const std::map<cv::Vec3b, uint8_t, Vec3bCompare>& reverseColorMap) {
    // Create a new single-channel image initialized to all zeros
    cv::Mat labelsImage = cv::Mat::zeros(bgrImage.rows, bgrImage.cols, CV_8UC1);

    for (int r = 0; r < bgrImage.rows; ++r) {
        for (int c = 0; c < bgrImage.cols; ++c) {
            const cv::Vec3b& pixelColor = bgrImage.at<cv::Vec3b>(r, c);
            
            // Find the BGR color in the map
            auto it = reverseColorMap.find(pixelColor);
            
            // If the color is found, set the label
            if (it != reverseColorMap.end()) {
                labelsImage.at<uint8_t>(r, c) = it->second;
            }
            // If not found, it remains 0 (background)
        }
    }
    return labelsImage;
}

/**
 * @brief Processes a single color segmentation image.
 *
 * This function is executed by a worker thread. It reads a BGR segmentation
 * image, converts it to a grayscale label image using the provided map,
 * and saves the result to the destination folder.
 *
 * @param predPath Path to the input color segmentation image.
 * @param labelsDestination Folder to save the output label image.
 * @param reverseColorMap Map to convert BGR colors to integer labels.
 */
void processImage(const fs::path& predPath, const fs::path& labelsDestination, const std::map<cv::Vec3b, uint8_t, Vec3bCompare>& reverseColorMap) {
    // Static mutex to protect std::cerr from interleaved output from multiple threads
    static std::mutex cerr_mutex;
    
    try {
        // Construct the output filename, e.g., "image101_predseg_labels.png"
        std::string fileStem = predPath.stem().string();
        std::string newFilename = fileStem + "_labels.png";
        fs::path labelImagePath = labelsDestination / newFilename;

        // Skip if the file already exists
        if (fs::exists(labelImagePath)) {
            return;
        }

        // Read the color segmentation image
        cv::Mat predImage = cv::imread(predPath.string(), cv::IMREAD_COLOR);
        if (predImage.empty()) {
            std::lock_guard<std::mutex> lock(cerr_mutex);
            std::cerr << "\nWarning: Could not read image: " << predPath << std::endl;
            return;
        }

        // Perform the conversion
        cv::Mat grayImg = convertBgrToGrayscaleLabels(predImage, reverseColorMap);
        
        // Save the new grayscale label image
        cv::imwrite(labelImagePath.string(), grayImg);

    } catch (const std::exception& e) {
        // Catch any exceptions (e.g., from OpenCV or filesystem)
        std::lock_guard<std::mutex> lock(cerr_mutex);
        std::cerr << "\nError processing " << predPath << ": " << e.what() << std::endl;
    }
}

/**
 * @brief Main entry point for the label generation application.
 *
 * This program reads a list of filenames from a text file, constructs
 * full paths to those images, and converts them from 3-channel BGR
 * segmentation maps to 1-channel grayscale label maps in parallel.
 *
 * Arguments:
 * argv[1]: <image_base_dir> - The base folder containing the BGR segmentation images.
 * argv[2]: <path_to_file_list.txt> - Path to a .txt file where each line
 * is a filename (e.g., "image101_predseg.png") located in the base dir.
 * argv[3]: <labels_destination_folder> - The folder to save the generated label images.
 */
int main(int argc, char* argv[]) {
    // --- 1. Argument Parsing and Validation ---
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <image_base_dir> <path_to_file_list.txt> <labels_destination_folder>" << std::endl;
        return 1;
    }

    fs::path image_base_dir(argv[1]);
    fs::path file_list_path(argv[2]);
    fs::path labels_destination(argv[3]);

    // Check that inputs are valid
    if (!fs::is_directory(image_base_dir)) {
        std::cerr << "Error: Image base directory not found or is not a directory: " << image_base_dir << std::endl;
        return 1;
    }
    if (!fs::exists(file_list_path) || !fs::is_regular_file(file_list_path)) {
        std::cerr << "Error: File list not found or is not a regular file: " << file_list_path << std::endl;
        return 1;
    }

    // Create the destination directory if it doesn't exist
    fs::create_directories(labels_destination);

    // --- 2. Setup Color-to-Label Map ---
    // This map defines the conversion from BGR colors to integer labels.
    std::vector<std::string> hex_colors = {
        "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099",
        "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395"
    };
    std::map<cv::Vec3b, uint8_t, Vec3bCompare> reverseColorMap;
    
    // Explicitly map black (0,0,0) to label 0
    reverseColorMap[cv::Vec3b(0, 0, 0)] = 0;
    
    // Map the hex colors to labels 1, 2, 3, ...
    uint8_t label_id = 1;
    for (const auto& hex : hex_colors) {
        reverseColorMap[hexToBgr(hex)] = label_id++;
    }

    // --- 3. Read File List and Construct Full Paths ---
    std::vector<fs::path> preds_list;
    std::ifstream file(file_list_path);
    std::string filename;
    
    if (file.is_open()) {
        while (std::getline(file, filename)) {
            if (!filename.empty()) {
                // Construct the full path: <image_base_dir> / <filename>
                preds_list.push_back(image_base_dir / filename);
            }
        }
        file.close();
    } else {
        std::cerr << "Error: Could not open file list: " << file_list_path << std::endl;
        return 1;
    }
    
    if (preds_list.empty()) {
        std::cout << "No image paths found in the input file." << std::endl;
        return 0;
    }

    // --- 4. Setup Thread Pool ---
    size_t num_files = preds_list.size();
    
    // Atomic counters for threads to safely share progress
    std::atomic<size_t> processed_count = 0; // Tracks how many are complete
    std::atomic<size_t> file_index = 0;      // Tracks the next file to process
    
    unsigned int num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0) num_threads = 4; // Fallback
    
    std::cout << "Starting label generation with " << num_threads << " threads for " << num_files << " images." << std::endl;

    std::vector<std::thread> threads;
    
    // Static mutex to protect the printf progress bar
    static std::mutex progress_mutex;

    // --- 5. Start Worker Threads ---
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            while (true) {
                // Atomically get the next file index and increment
                size_t index = file_index.fetch_add(1);
                
                // If the index is out of bounds, this thread's work is done
                if (index >= num_files) break;

                // Process the image
                processImage(preds_list[index], labels_destination, reverseColorMap);
                
                // Atomically increment the completed count
                size_t count = processed_count.fetch_add(1) + 1;
                
                // Update progress bar periodically
                if (count % 100 == 0 || count == num_files) {
                    // Lock the mutex to prevent garbled console output
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    // Use printf/fflush for a single-line updating progress bar
                    printf("\rProgress: %zu / %zu (%.2f%%)", count, num_files, (float)count / num_files * 100.0);
                    fflush(stdout);
                }
            }
        });
    }

    // --- 6. Join Threads ---
    // Wait for all worker threads to finish
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\nLabel generation complete. Files saved to: " << labels_destination << std::endl;

    return 0;
}