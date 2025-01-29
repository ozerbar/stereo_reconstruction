#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>

int main()
{
    // Load images
    cv::Mat leftImage = cv::imread("/workspace/Datasets/im0.png", cv::IMREAD_COLOR);  // query image
    cv::Mat rightImage = cv::imread("/workspace/Datasets/im1.png", cv::IMREAD_COLOR); // train image
    if (leftImage.empty() || rightImage.empty())
    {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }

    // Initialize ORB detector and variables for keypoints and descriptors
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;

    // Initialize ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);

    // Detect keypoints
    orb->detect(leftImage, keypointsLeft);
    orb->detect(rightImage, keypointsRight);

    // Compute descriptors
    orb->compute(leftImage, keypointsLeft, descriptorsLeft);
    orb->compute(rightImage, keypointsRight, descriptorsRight);

    // Match descriptors using Hamming Distance
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptorsLeft, descriptorsRight, matches);

    // Select max and min distance, and define the good match
    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < descriptorsLeft.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    // Filter good matches based on distance
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptorsLeft.rows; i++)
    {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    // Collect matched points
    std::vector<cv::Point2f> ptsLeft, ptsRight;
    for (unsigned int i = 0; i < good_matches.size(); ++i)
    {
        ptsLeft.push_back(keypointsLeft[good_matches[i].queryIdx].pt);
        ptsRight.push_back(keypointsRight[good_matches[i].trainIdx].pt);
    }

    if (ptsLeft.size() < 4) // Need at least 4 points to calculate homography
    {
        std::cerr << "Error: Not enough matches to compute Homography!" << std::endl;
        return -1;
    }

    // Compute Homography matrix from right image to left image
    cv::Mat H = cv::findHomography(ptsRight, ptsLeft, cv::RANSAC);
    std::cout << "Homography Matrix (Right to Left): \n"
              << H << std::endl;

    // Apply the homography matrix to rectify the right image (map right to left)
    cv::Mat rectifiedRight;
    cv::warpPerspective(rightImage, rectifiedRight, H, leftImage.size());

    // Save the rectified right image
    cv::imwrite("rectified_right_to_left.jpg", rectifiedRight);

    // Now compute disparity map using StereoBM (block matching)
    cv::Mat leftGray, rightGray;
    cv::cvtColor(leftImage, leftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rectifiedRight, rightGray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 9); // 16 disparity levels and block size 9
    cv::Mat disparity;
    stereo->compute(leftGray, rightGray, disparity); // Compute disparity

    // Normalize disparity to view it as an image
    cv::Mat disparityNormalized;
    double minVal, maxVal;
    cv::minMaxLoc(disparity, &minVal, &maxVal);
    disparity.convertTo(disparityNormalized, CV_8U, 255.0 / (maxVal - minVal));

    cv::imwrite("disparity_image.jpg", disparityNormalized);

    std::cout << "Rectified right image and disparity map saved successfully." << std::endl;

    return 0;
}
