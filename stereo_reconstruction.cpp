#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>


#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "includes/common.hpp"



// /* ==========  SGBM disparity function ========== */
// cv::Mat computeDisparitySGBM(const cv::Mat &leftGray, const cv::Mat &rightGray)
// {
//     cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
//         0,   // minDisparity
//         128,  // numDisparities
//         23   // blockSize
//     );
//     // More params can be set if needed



/* ==========  FastSGBM disparity function ========== */

// Add a new function to use the custom FastSGBM implementation
cv::Mat computeDisparityFastSGBM(cv::Mat &leftGray, cv::Mat &rightGray, int dispRange, int p1, int p2)
{
    FastSGBM sgbm(leftGray.rows, leftGray.cols, dispRange, p1, p2, true);
    cv::Mat disparity;
    sgbm.compute_disp(leftGray, rightGray, disparity);
    return disparity;
}



/* =============================
 * 2) Main: do both Uncalibrated and Calibrated for rectification and disparity
 * =============================*/


int main()
{
    // 1) Load images
    std::string leftImagePath  = "../Datasets/Shopvac-imperfect/im0.png";
    std::string rightImagePath = "../Datasets/Shopvac-imperfect/im1.png";

    cv::Mat leftImage  = cv::imread(leftImagePath,  cv::IMREAD_COLOR);
    cv::Mat rightImage = cv::imread(rightImagePath, cv::IMREAD_COLOR);
    if (leftImage.empty() || rightImage.empty()) {
        std::cerr << "[Error] Failed to load images.\n";
        return -1;
    }

    // 2) Define feature methods
    struct FeatureMethod {
        std::string name;
        cv::Ptr<cv::Feature2D> extractor;
        int normType;
    };
    std::vector<FeatureMethod> methods = {
        { "ORB",   cv::ORB::create(),   cv::NORM_HAMMING },
        { "SIFT",  cv::SIFT::create(),  cv::NORM_L2      },
        { "BRISK", cv::BRISK::create(), cv::NORM_HAMMING }
    };

    // 3) Suppose we have camera intrinsics for "calibrated" approach
    //    (Replace these with your actual intrinsics)
    cv::Mat K0 = (cv::Mat_<double>(3,3) << 7228.4,  0,      1112.085,
                                           0,       7228.4, 1010.431,
                                           0,       0,      1.0);
    cv::Mat K1 = (cv::Mat_<double>(3,3) << 7228.4,  0,      1628.613,
                                           0,       7228.4, 1010.431,
                                           0,       0,      1.0);
    cv::Mat D0 = cv::Mat::zeros(1,5, CV_64F);
    cv::Mat D1 = cv::Mat::zeros(1,5, CV_64F);

    // 4) Loop over each feature method
    for (auto &method : methods)
    {
        std::cout << "\n========== " << method.name << " ==========\n";

        // (a) Detect and compute descriptors
        cv::Mat leftGray, rightGray;
        cv::cvtColor(leftImage,  leftGray,  cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightImage, rightGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> kpLeft, kpRight;
        cv::Mat descLeft, descRight;
        method.extractor->detectAndCompute(leftGray,  cv::noArray(), kpLeft,  descLeft);
        method.extractor->detectAndCompute(rightGray, cv::noArray(), kpRight, descRight);
        if (descLeft.empty() || descRight.empty()) {
            std::cerr << "[WARN] " << method.name << ": no descriptors.\n";
            continue;
        }

        // (b) KNN match + ratio test
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(method.normType, false);
        std::vector<std::vector<cv::DMatch>> knn;
        matcher->knnMatch(descLeft, descRight, knn, 2);

        std::vector<cv::DMatch> goodMatches;
        double ratioThresh = 0.75;
        for (auto &m : knn) {
            if (m.size() == 2) {
                float dist1 = m[0].distance;
                float dist2 = m[1].distance;
                if (dist1 < ratioThresh * dist2) {
                    goodMatches.push_back(m[0]);
                }
            }
        }

        if (goodMatches.size() < 8) {
            std::cerr << "[WARN] " << method.name << ": Not enough good matches.\n";
            continue;
        }

        // (c) Collect matched points
        std::vector<cv::Point2f> ptsLeft, ptsRight;
        ptsLeft.reserve(goodMatches.size());
        ptsRight.reserve(goodMatches.size());
        for (auto &gm : goodMatches) {
            ptsLeft.push_back( kpLeft[gm.queryIdx].pt );
            ptsRight.push_back(kpRight[gm.trainIdx].pt );
        }

        // (d) RANSAC => Fundamental matrix
        int inlierCount = 0;
        cv::Mat F = ransacFundamentalMatrix(ptsLeft, ptsRight, 5000, 1.5f, inlierCount);
        if (F.empty()) {
            std::cerr << "[WARN] " << method.name << ": F is empty.\n";
            continue;
        }
        std::cout << "[INFO] " << method.name << " => #inliers=" << inlierCount << "\n";

        /********************************************************
         * (1) Uncalibrated Rectification
         ********************************************************/
        // cv::Mat H1, H2;
        // cv::stereoRectifyUncalibrated(ptsLeft, ptsRight, F, leftImage.size(), H1, H2);

        // // Warp perspective
        // cv::Mat rectLeftU, rectRightU;
        // cv::warpPerspective(leftImage,  rectLeftU,  H1, leftImage.size());
        // cv::warpPerspective(rightImage, rectRightU, H2, leftImage.size());

        // // Concatenate horizontally
        // cv::Mat stereoUncalib;
        // cv::hconcat(rectLeftU, rectRightU, stereoUncalib);

        // // Save the horizontally stitched rectified image
        // std::string uncalibName = "uncalib_stereo_concat_" + method.name + ".png";
        // cv::imwrite(uncalibName, stereoUncalib);
        
        // //disparity map
        // // Convert to gray 
        // cv::Mat rectLeftUGray, rectRightUGray;
        // cv::cvtColor(rectLeftU,   rectLeftUGray,   cv::COLOR_BGR2GRAY);
        // cv::cvtColor(rectRightU,  rectRightUGray,  cv::COLOR_BGR2GRAY);

        // // Optionally use CLAHE or other pre-processing
        // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
        // clahe->apply(rectLeftUGray,  rectLeftUGray);
        // clahe->apply(rectRightUGray, rectRightUGray);

        // // Compute disparity (SGBM)
        // cv::Mat dispU = computeDisparitySGBM(rectLeftUGray, rectRightUGray);

        // // Normalize to 8U for saving
        // double minv, maxv;
        // cv::minMaxLoc(dispU, &minv, &maxv);
        // cv::Mat dispU8;
        // dispU.convertTo(dispU8, CV_8U, 255.0/(maxv - minv + 1e-6));

        // // Save disparity
        // std::string dispUName = "uncalib_disparity_" + method.name + ".png";
        // cv::imwrite(dispUName, dispU8);


        /********************************************************
         * (2) Calibrated Rectification (Own Implementation)
         ********************************************************/
        // Compute the Essential matrix using the calibrated intrinsics and F
        // (1) Compute the homographies H_left, H_right as you do:
        cv::Mat E = K1.t() * F * K0;
        cv::Mat bestR, bestT;
        decomposeEssentialMatrix(E, ptsLeft, ptsRight, K0, bestR, bestT);

        // Left: P_left = K0 * [I|0], Right: P_right = K1 * [R|t]
        cv::Mat P_left = cv::Mat::zeros(3, 4, CV_64F);
        K0.copyTo(P_left(cv::Rect(0, 0, 3, 3)));  // top-left 3×3 submatrix

        cv::Mat RT;
        cv::hconcat(bestR, bestT, RT);
        cv::Mat P_right = K1 * RT;

        // Rectify homographies
        cv::Mat H_left  = P_left(cv::Rect(0, 0, 3, 3)) * K0.inv();
        cv::Mat H_right = P_right(cv::Rect(0, 0, 3, 3)) * K1.inv();

        // Warp
        cv::Mat rectLeftC, rectRightC;
        myWarpPerspective(leftImage,  rectLeftC,  H_left,  leftImage.size());
        myWarpPerspective(rightImage, rectRightC, H_right, leftImage.size());

        // (2) Convert to gray
        cv::Mat rectLeftCGray, rectRightCGray;
        cv::cvtColor(rectLeftC,  rectLeftCGray,  cv::COLOR_BGR2GRAY);
        cv::cvtColor(rectRightC, rectRightCGray, cv::COLOR_BGR2GRAY);

        // (3) If you want to save memory or speed up, downscale these *rectified* images:
        cv::Mat rectLeftCGraySmall, rectRightCGraySmall;
        cv::resize(rectLeftCGray,  rectLeftCGraySmall,  cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        cv::resize(rectRightCGray, rectRightCGraySmall, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

        // (4) Compute disparity on the smaller rectified images
        int dispRange = 64;
        int p1 = 2, p2 = 5;
        cv::Mat dispC = computeDisparityFastSGBM(rectLeftCGraySmall, rectRightCGraySmall, dispRange, p1, p2);

        // (5) Rescale disparity to 8-bit for saving
        double minv2, maxv2;
        cv::minMaxLoc(dispC, &minv2, &maxv2);
        cv::Mat dispC8;
        dispC.convertTo(dispC8, CV_8U, 255.0 / (maxv2 - minv2 + 1e-6));

        std::string dispCName = "calib_disparity_" + method.name + ".png";
        cv::imwrite(dispCName, dispC8);



    //     /********************************************************
    //      * (2) Calibrated Rectification - Opencv implementation
    //      ********************************************************/
    //     // E = K1^T * F * K0
    //     cv::Mat E = K1.t() * F * K0;

    //     // Decompose E => (R, t)
    //     cv::Mat bestR, bestT;
    //     decomposeEssentialMatrix(E, ptsLeft, ptsRight, K0, bestR, bestT);

    //     // stereoRectify
    //     cv::Size imageSize(leftImage.cols, leftImage.rows);
    //     cv::Mat RR1, RR2, PP1, PP2, Q;
    //     cv::stereoRectify(K0, D0, K1, D1, imageSize,
    //                       bestR, bestT, RR1, RR2, PP1, PP2, Q,
    //                       0, -1, imageSize);

    //     // initUndistortRectifyMap + remap
    //     cv::Mat mapxL, mapyL, mapxR, mapyR;
    //     cv::initUndistortRectifyMap(K0, D0, RR1, PP1, imageSize, CV_32FC1, mapxL, mapyL);
    //     cv::initUndistortRectifyMap(K1, D1, RR2, PP2, imageSize, CV_32FC1, mapxR, mapyR);

    //     cv::Mat rectLeftC, rectRightC;
    //     cv::remap(leftImage,  rectLeftC,  mapxL, mapyL, cv::INTER_LINEAR);
    //     cv::remap(rightImage, rectRightC, mapxR, mapyR, cv::INTER_LINEAR);

    //     // Concatenate horizontally
    //     cv::Mat stereoCalib;
    //     cv::hconcat(rectLeftC, rectRightC, stereoCalib);

    //     // Save the horizontally stitched rectified image
    //     std::string calibName = "calib_stereo_concat_" + method.name + ".png";
    //     cv::imwrite(calibName, stereoCalib);
        
    //     // --- Compute disparity from the calibrated rectified pair ---
    //     // Convert to gray
    //     cv::Mat rectLeftCGray, rectRightCGray;
    //     cv::cvtColor(rectLeftC,   rectLeftCGray,   cv::COLOR_BGR2GRAY);
    //     cv::cvtColor(rectRightC,  rectRightCGray,  cv::COLOR_BGR2GRAY);


    //     // Compute disparity
    //     cv::Mat dispC = computeDisparitySGBM(rectLeftCGray, rectRightCGray);

    //     // Normalize for saving
    //     // double minv, maxv;
    //     cv::minMaxLoc(dispC, &minv, &maxv);
    //     cv::Mat dispC8;
    //     dispC.convertTo(dispC8, CV_8U, 255.0/(maxv - minv + 1e-6));

    //     std::string dispCName = "calib_disparity_" + method.name + ".png";
    //     cv::imwrite(dispCName, dispC8);
    
    }

    std::cout << "\nDone. 6 images (3 methods x 2 rectification) saved.\n";
    return 0;
}