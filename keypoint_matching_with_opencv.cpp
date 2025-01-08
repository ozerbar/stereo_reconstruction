#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

int main() {
    // Paths to stereo image pair
    std::string leftImagePath = "../Datasets/artroom1/im0.png";
    std::string rightImagePath = "../Datasets/artroom1/im1.png";

    // Load stereo images
    cv::Mat leftImage = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat rightImage = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);

    if (leftImage.empty() || rightImage.empty()) {
        std::cerr << "Error: Could not load stereo image pair. Check paths!" << std::endl;
        return -1;
    }

    // Detect keypoints and extract descriptors using ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;

    orb->detectAndCompute(leftImage, cv::noArray(), keypointsLeft, descriptorsLeft);
    orb->detectAndCompute(rightImage, cv::noArray(), keypointsRight, descriptorsRight);

    // Match descriptors using BFMatcher
    cv::BFMatcher matcher(cv::NORM_HAMMING, true); // Use Hamming distance for ORB
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsLeft, descriptorsRight, matches);

    // Sort matches by distance
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
        return a.distance < b.distance;
    });

    // Draw the top 10 matches
    const int numMatchesToDraw = 10;
    std::vector<cv::DMatch> topMatches(matches.begin(), matches.begin() + std::min((int)matches.size(), numMatchesToDraw));

    // Create a canvas for matches
    cv::Mat matchImage;
    cv::drawMatches(leftImage, keypointsLeft, rightImage, keypointsRight, topMatches, matchImage);

    // Manually draw thicker match lines
    for (const auto& match : topMatches) {
        cv::Point2f ptLeft = keypointsLeft[match.queryIdx].pt;
        cv::Point2f ptRight = keypointsRight[match.trainIdx].pt;
        ptRight.x += leftImage.cols; // Shift right image keypoints to the right on the canvas

        cv::line(matchImage, ptLeft, ptRight, cv::Scalar(0, 255, 0), 3); // Thicker green line
    }

    // Save the result to a file
    std::string outputPath = "output_with_opencv.jpg";
    if (cv::imwrite(outputPath, matchImage)) {
        std::cout << "Image with thicker match lines saved as " << outputPath << std::endl;
    } else {
        std::cerr << "Error: Could not save the image." << std::endl;
        return -1;
    }

    return 0;
}
