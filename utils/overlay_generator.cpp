#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <fstream>
#include <clocale> // For std::setlocale

// OpenCV Headers
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
 * @param hex A string representing the color in "#RRGGBB" format.
 * @return cv::Vec3b The color in OpenCV's default BGR format.
 */
cv::Vec3b hexToBgr(const std::string& hex) {
    int r, g, b;
    // Parse the hex string as R, G, B
    sscanf(hex.substr(1).c_str(), "%02x%02x%02x", &r, &g, &b);
    
    // Construct the cv::Vec3b in (B, G, R) order, as expected by OpenCV
    return cv::Vec3b(b, g, r);
}

/**
 * @brief Re-colors an image based on a provided color map.
 *
 * Iterates through each pixel of the input image. If the pixel's color
 * is a key in the colorMap, it's replaced with the map's corresponding value.
 * Otherwise, the original color is kept.
 *
 * @param inputImg The original segmentation image to be recolored.
 * @param colorMap A std::map mapping original colors (keys) to new colors (values).
 * @return cv::Mat A new image with the colors remapped.
 */
cv::Mat recolorImage(const cv::Mat& inputImg, const std::map<cv::Vec3b, cv::Vec3b, Vec3bCompare>& colorMap) {
    cv::Mat outputImg = cv::Mat::zeros(inputImg.size(), inputImg.type());
    
    for (int r = 0; r < inputImg.rows; ++r) {
        for (int c = 0; c < inputImg.cols; ++c) {
            const cv::Vec3b& originalColor = inputImg.at<cv::Vec3b>(r, c);
            
            // Find the original color in the map
            auto it = colorMap.find(originalColor);
            
            if (it != colorMap.end()) {
                // If found, replace it with the new color
                outputImg.at<cv::Vec3b>(r, c) = it->second;
            } else {
                // Otherwise, keep the original color
                outputImg.at<cv::Vec3b>(r, c) = originalColor;
            }
        }
    }
    return outputImg;
}

/**
 * @brief Processes a single pair of images (background and foreground).
 *
 * This function is designed to be executed by a worker thread. It reads
 * the background and segmentation images, recolors the segmentation,
 * overlays it onto the background, and saves the result to the destination path.
 * Includes error checking for file I/O and dimension mismatches.
 *
 * @param imagePath Path to the original background image.
 * @param segPath Path to the foreground segmentation image.
 * @param destPath Full path (including filename) to save the final overlay.
 * @param alpha The transparency level for the foreground overlay (0.0 to 1.0).
 * @param colorMap The color map to use for recoloring the segmentation.
 */
void processImagePair(
    const std::string& imagePath,
    const std::string& segPath,
    const std::string& destPath,
    double alpha,
    const std::map<cv::Vec3b, cv::Vec3b, Vec3bCompare>& colorMap)
{
    // A static mutex to protect std::cerr from interleaved/garbled
    // output from multiple threads.
    static std::mutex cerr_mutex;

    cv::Mat background = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::Mat foreground = cv::imread(segPath, cv::IMREAD_COLOR);

    // --- 1. Validate that images were read correctly ---
    if (background.empty()) {
        std::lock_guard<std::mutex> lock(cerr_mutex);
        std::cerr << "\n[Error] Failed to read background image: " << imagePath << std::endl;
        return;
    }
    if (foreground.empty()) {
        std::lock_guard<std::mutex> lock(cerr_mutex);
        std::cerr << "\n[Error] Failed to read foreground image: " << segPath << std::endl;
        return;
    }
    
    // --- 2. Recolor the segmentation mask ---
    cv::Mat recoloredForeground = recolorImage(foreground, colorMap);

    // --- 3. Validate that image dimensions match before overlaying ---
    if (background.size() != recoloredForeground.size()) {
        std::lock_guard<std::mutex> lock(cerr_mutex);
        std::cerr << "\n[Error] Image size mismatch. Background (" << background.cols << "x" << background.rows 
                  << ") vs Foreground (" << recoloredForeground.cols << "x" << recoloredForeground.rows 
                  << "). Files: " << imagePath << " and " << segPath << std::endl;
        return;
    }

    // --- 4. Blend the images ---
    cv::Mat overlay;
    double beta = 1.0 - alpha; // Transparency for the background
    double gamma = 0.0;        // Scalar added to each sum
    
    // Perform the weighted sum: overlay = (foreground * alpha) + (background * beta) + gamma
    cv::addWeighted(recoloredForeground, alpha, background, beta, gamma, overlay);

    // --- 5. Validate that the final overlay image is not empty ---
    if (overlay.empty()) {
        std::lock_guard<std::mutex> lock(cerr_mutex);
        std::cerr << "\n[Error] Generated overlay is empty for files: " << imagePath << " and " << segPath << std::endl;
        return;
    }

    // --- 6. Save the final image, checking for success ---
    bool success = cv::imwrite(destPath, overlay);
    if (!success) {
        std::lock_guard<std::mutex> lock(cerr_mutex);
        std::cerr << "\n[Error] FAILED TO WRITE image to disk at: " << destPath 
                  << ". Check folder permissions and path validity." << std::endl;
    }
}

/**
 * @brief Main entry point for the image overlay application.
 *
 * This program takes a list of image files, finds corresponding segmentation
 * masks, recolors the masks, and overlays them on the original images.
 * The processing is done in parallel using a thread pool.
 *
 * Arguments:
 * argv[1]: <image_folder> - Path to the folder containing original images.
 * argv[2]: <segs_folder> - Path to the folder containing segmentation images.
 * argv[3]: <destination_folder> - Path where overlay images will be saved.
 * argv[4]: <file_list_path> - Path to a .txt file listing the basenames
 * of the original images to process.
 * argv[5]: [alpha=0.5] - (Optional) Transparency of the overlay. Defaults to 0.5.
 */
int main(int argc, char* argv[]) {
    // Force the program to use the standard "C" locale for number formatting.
    // This ensures std::stod (and other functions) use '.' as the decimal
    // separator, regardless of the system's regional settings.
    std::setlocale(LC_ALL, "C");

    // --- 1. Argument Parsing and Validation ---
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <image_folder> <segs_folder> <destination_folder> <file_list_path> [alpha=0.5]" << std::endl;
        return 1;
    }

    fs::path imageFolder(argv[1]);
    fs::path segsFolder(argv[2]);
    fs::path destFolder(argv[3]);
    fs::path fileListPath(argv[4]);
    double alpha = (argc > 5) ? std::stod(argv[5]) : 0.5;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  - Image Folder: " << imageFolder << std::endl;
    std::cout << "  - Segs Folder:  " << segsFolder << std::endl;
    std::cout << "  - Dest Folder:  " << destFolder << std::endl;
    std::cout << "  - File List:    " << fileListPath << std::endl;
    std::cout << "  - Alpha:        " << alpha << std::endl;

    // --- 2. Setup Color Map ---
    // Define the mapping from original segmentation colors (keys) to
    // new desired colors (values). Colors are defined as BGR via hexToBgr.
    std::map<cv::Vec3b, cv::Vec3b, Vec3bCompare> colorMap;
    colorMap[hexToBgr("#3366CC")] = hexToBgr("#3366CC");
    colorMap[hexToBgr("#DC3912")] = hexToBgr("#DC3912");
    colorMap[hexToBgr("#FF9900")] = hexToBgr("#C1C4C9");
    colorMap[hexToBgr("#109618")] = hexToBgr("#A2DDF2");
    colorMap[hexToBgr("#990099")] = hexToBgr("#047511");
    colorMap[hexToBgr("#0099C6")] = hexToBgr("#C1C4C9");
    colorMap[hexToBgr("#DD4477")] = hexToBgr("#FA9BDA");
    colorMap[hexToBgr("#66AA00")] = hexToBgr("#A2DDF2");
    colorMap[hexToBgr("#B82E2E")] = hexToBgr("#047511");
    colorMap[hexToBgr("#316395")] = hexToBgr("#755304");

    try {
        // Ensure the destination directory exists.
        fs::create_directories(destFolder);

        // --- 3. Collect File Paths with Correct Naming Logic ---
        std::vector<fs::path> imageFiles, segFiles;
        std::ifstream fileListStream(fileListPath);
        if (!fileListStream) {
            std::cerr << "Error: Cannot open file list: " << fileListPath << std::endl;
            return 1;
        }

        std::string image_basename;
        // Read the file list line by line
        while (std::getline(fileListStream, image_basename)) {
            if (image_basename.empty()) continue;

            // Find the last '.' to separate the stem and extension
            size_t const pos = image_basename.find_last_of(".");
            // Get the filename stem (e.g., "image101" from "image101.png")
            std::string stem = (pos == std::string::npos) ? image_basename : image_basename.substr(0, pos);
            
            // Construct the corresponding prediction filename (e.g., "image101_predseg.png")
            std::string pred_basename = stem + "_predseg.png";

            fs::path currentImagePath = imageFolder / image_basename;
            fs::path currentSegPath = segsFolder / pred_basename;

            // Check if both files in the pair actually exist before adding
            if (!fs::exists(currentImagePath) || !fs::exists(currentSegPath)) {
                if (!fs::exists(currentImagePath)) std::cerr << "\nWarning: Skipping pair because image does not exist: " << currentImagePath << std::endl;
                if (!fs::exists(currentSegPath)) std::cerr << "\nWarning: Skipping pair because prediction does not exist: " << currentSegPath << std::endl;
                continue;
            }

            imageFiles.push_back(currentImagePath);
            segFiles.push_back(currentSegPath);
        }

        if (imageFiles.empty()) {
            std::cout << "No valid image pairs found from the file list." << std::endl;
            return 0;
        }

        size_t totalFiles = imageFiles.size();
        std::cout << "Found " << totalFiles << " image pairs to process from file list." << std::endl;

        // --- 4. Setup Thread Pool ---
        unsigned int num_threads = std::thread::hardware_concurrency();
        std::cout << "Using " << num_threads << " threads for processing." << std::endl;
        
        std::vector<std::thread> threads;
        // Atomic index to track which file is next to be processed
        std::atomic<size_t> file_idx(0);
        // Atomic counter for tracking progress
        std::atomic<int> progress(0);

        // Define the worker lambda function that threads will execute
        auto worker = [&]() {
            while (true) {
                // Atomically get the next file index and increment the shared counter
                size_t i = file_idx.fetch_add(1);
                
                // If the index is out of bounds, this thread is done
                if (i >= totalFiles) break;

                // Construct the full output path
                fs::path destPath = destFolder / ("segmap_overlay_" + imageFiles[i].filename().string());
                
                // Call the main processing function for this pair
                processImagePair(
                    imageFiles[i].string(),
                    segFiles[i].string(),
                    destPath.string(),
                    alpha,
                    colorMap
                );
                
                // Atomically increment the progress counter
                int current_progress = ++progress;
                float percentage = (static_cast<float>(current_progress) / totalFiles) * 100.0f;
                
                // A static mutex is required to protect std::cout
                // from being written to by multiple threads simultaneously.
                static std::mutex cout_mutex;
                std::lock_guard<std::mutex> lock(cout_mutex);
                
                // Use carriage return '\r' and std::flush to create a 
                // single, updating line for the progress bar.
                std::cout << "\rGenerating segmentation overlays... [" << std::fixed << std::setprecision(2) << percentage << "%] "
                          << current_progress << "/" << totalFiles << std::flush;
            }
        };

        // --- 5. Start and Join Threads ---
        
        // Launch the worker threads
        for (unsigned int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker);
        }

        // Wait for all worker threads to complete their tasks
        for (auto& t : threads) {
            t.join();
        }

        // Print a final newline to move past the progress bar line
        std::cout << std::endl << "Task complete. Overlays generated successfully for this job." << std::endl;

    } catch (const std::exception& e) {
        // Catch any standard exceptions (e.g., from std::stod or filesystem)
        std::cerr << "A critical error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}