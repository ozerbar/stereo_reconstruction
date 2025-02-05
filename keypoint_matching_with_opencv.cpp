#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

// Helper struct to store relevant data for each method
struct FeatureMethod {
    std::string name;
    cv::Ptr<cv::Feature2D> featureExtractor;
    int normType; // NORM_HAMMING or NORM_L2, etc.
};

int main() {
    // --------------------------------------------------------------------------
    // 1. Load Images
    // --------------------------------------------------------------------------
    std::string leftImagePath  = "../Datasets/artroom1/im0.png";
    std::string rightImagePath = "../Datasets/artroom1/im1.png";

    cv::Mat leftImage  = cv::imread(leftImagePath,  cv::IMREAD_GRAYSCALE);
    cv::Mat rightImage = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);

    if (leftImage.empty() || rightImage.empty()) {
        std::cerr << "Error: Could not load stereo image pair. Check paths!" << std::endl;
        return -1;
    }

    // --------------------------------------------------------------------------
    // 2. Define the methods (detector + descriptor) we want to compare
    // --------------------------------------------------------------------------
    // Note:
    // - ORB, BRISK, AKAZE, etc. produce binary descriptors -> use NORM_HAMMING
    // - SIFT, SURF produce float descriptors -> use NORM_L2
    // Make sure your OpenCV version includes the relevant features.
    std::vector<FeatureMethod> methods;

    // ORB
    methods.push_back({
        "ORB",
        cv::ORB::create(), // ORB Feature2D
        cv::NORM_HAMMING
    });

    // SIFT (available in default OpenCV >=4.4 or with opencv_contrib in earlier versions)
    methods.push_back({
        "SIFT",
        cv::SIFT::create(), // SIFT Feature2D
        cv::NORM_L2
    });

    // BRISK
    methods.push_back({
        "BRISK",
        cv::BRISK::create(),
        cv::NORM_HAMMING
    });

    // (Optional) AKAZE
    // methods.push_back({
    //     "AKAZE",
    //     cv::AKAZE::create(),
    //     cv::NORM_HAMMING
    // });

    // (Optional) SURF if you have xfeatures2d
    // methods.push_back({
    //     "SURF",
    //     cv::xfeatures2d::SURF::create(),
    //     cv::NORM_L2
    // });

    // --------------------------------------------------------------------------
    // 3. Compare methods
    // --------------------------------------------------------------------------
    const int numMatchesToDraw = 30;    // Adjust as needed
    bool doRatioTest = true;           // If you want to apply Lowe's ratio test
    double ratioThresh = 0.75;         // Typical ratio threshold

    double bestAvgDistance = DBL_MAX;  // Keep track of best average distance
    std::string bestMethodName;        // Store best method name

    for (auto &method : methods) {
        // -----------------------------------
        // 3.1 Detect and compute descriptors
        // -----------------------------------
        std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
        cv::Mat descriptorsLeft, descriptorsRight;

        method.featureExtractor->detectAndCompute(leftImage,  cv::noArray(), keypointsLeft,  descriptorsLeft);
        method.featureExtractor->detectAndCompute(rightImage, cv::noArray(), keypointsRight, descriptorsRight);

        if (descriptorsLeft.empty() || descriptorsRight.empty()) {
            std::cerr << "Warning: No descriptors found for method: " << method.name << std::endl;
            continue;
        }

        // -----------------------------------
        // 3.2 Match descriptors
        // -----------------------------------
        // For float descriptors (SIFT/SURF) -> NORM_L2
        // For binary descriptors (ORB/BRISK/AKAZE) -> NORM_HAMMING
        cv::Ptr<cv::DescriptorMatcher> matcher;

        // You can choose BFMatcher or FLANN-based here
        if (method.normType == cv::NORM_HAMMING || method.normType == cv::NORM_HAMMING2) {
            // For binary descriptors, BFMatcher with NORM_HAMMING is typical
            matcher = cv::BFMatcher::create(method.normType, false);
        } else {
            // For float descriptors (SIFT, SURF) you can choose BFMatcher with NORM_L2
            // or FLANN. Here BFMatcher for simplicity
            matcher = cv::BFMatcher::create(method.normType, false);
        }

        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descriptorsLeft, descriptorsRight, knnMatches, 2);

        // -----------------------------------
        // 3.3 (Optional) Ratio Test
        // -----------------------------------
        std::vector<cv::DMatch> goodMatches;
        if (doRatioTest) {
            for (size_t i = 0; i < knnMatches.size(); i++) {
                if (knnMatches[i].size() == 2) {
                    float dist1 = knnMatches[i][0].distance;
                    float dist2 = knnMatches[i][1].distance;
                    if (dist1 < ratioThresh * dist2) {
                        goodMatches.push_back(knnMatches[i][0]);
                    }
                }
            }
        } else {
            // If not using ratio test, simply take the best match of each KNN
            for (size_t i = 0; i < knnMatches.size(); i++) {
                if (!knnMatches[i].empty()) {
                    goodMatches.push_back(knnMatches[i][0]);
                }
            }
        }

        // -----------------------------------
        // 3.4 Sort matches by distance, keep top N
        // -----------------------------------
        std::sort(goodMatches.begin(), goodMatches.end(),
                  [](const cv::DMatch &a, const cv::DMatch &b) {
                      return a.distance < b.distance;
                  });
        if ((int)goodMatches.size() > numMatchesToDraw) {
            goodMatches.erase(goodMatches.begin() + numMatchesToDraw, goodMatches.end());
        }

        // -----------------------------------
        // 3.5 Compute average distance (for comparison)
        // -----------------------------------
        double sumDist = 0.0;
        for (auto &m : goodMatches) {
            sumDist += m.distance;
        }
        double avgDist = goodMatches.empty() ? 999999.0 : sumDist / goodMatches.size();

        // ----------------------------------------------------------------------
        // Print results
        // ----------------------------------------------------------------------
        std::cout << "Method: " << method.name 
                  << ", Keypoints Left: " << keypointsLeft.size() 
                  << ", Keypoints Right: " << keypointsRight.size()
                  << ", Good Matches: " << goodMatches.size() 
                  << ", Avg Dist: " << avgDist << std::endl;

        // Keep track of "best" method by lowest average distance
        if (avgDist < bestAvgDistance && !goodMatches.empty()) {
            bestAvgDistance = avgDist;
            bestMethodName = method.name;
        }

        // ----------------------------------------------------------------------
        // 4. Draw the top matches and save to file
        // ----------------------------------------------------------------------
        cv::Mat matchImage;
        cv::drawMatches(leftImage, keypointsLeft, 
                        rightImage, keypointsRight, 
                        goodMatches, matchImage, 
                        cv::Scalar(255, 0, 0), // color for keypoints
                        cv::Scalar(0, 255, 0)); // color for matches

        // (Optional) Draw thicker lines manually:
        // Shift right keypoint x-coordinates by leftImage.cols
        for (const auto &m : goodMatches) {
            cv::Point2f ptLeft  = keypointsLeft[m.queryIdx].pt;
            cv::Point2f ptRight = keypointsRight[m.trainIdx].pt;
            ptRight.x += (float)leftImage.cols;

            cv::line(matchImage, ptLeft, ptRight, cv::Scalar(0, 255, 0), 2);
        }

        // Output filename: method name, plus something like output_<method>.jpg
        std::string outputPath = "output_matches_" + method.name + ".jpg";
        if (cv::imwrite(outputPath, matchImage)) {
            std::cout << "Image with top " << numMatchesToDraw << " matches saved: " 
                      << outputPath << std::endl << std::endl;
        } else {
            std::cerr << "Could not save image: " << outputPath << std::endl << std::endl;
        }
    }

    // --------------------------------------------------------------------------
    // 5. Print final conclusion about which method was "best"
    // --------------------------------------------------------------------------
    std::cout << "Best method based on lowest average distance: " 
              << bestMethodName << " (avg dist = " << bestAvgDistance << ")" 
              << std::endl;

    return 0;
}
